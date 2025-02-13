import os
from pathlib import Path
import common.constant as constant
import build_embedding_matrix as bem
from common.tool.progress import print_progress
import sent2vec
from code_embedding.qwenCoder_tokenize import tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

# 从 C 函数提取的 mid_graph 中获取 raw graph 和 sequence 类型特征向量
def get_vector_raw_and_sequence_by_mid_graph(function_path):
    # 保证函数为 .c，报错
    if function_path[-2:] != ".c":
        raise Exception("The file is not a C file!")
    
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(constant.represent_dir + "/raw_model.bin")

    # 提取函数图信息
    label, source_code, adj_ast, adj_cfg, \
    adj_pdg, ast_nodes, cfg_nodes, pdg_nodes, \
    pdg_start_nodes, pdg_dest_nodes, ast_start_nodes, \
    ast_dest_nodes, problem = bem.extract_graph_info(function_path)

    if problem:
        # 写个问题日志
        with open(constant.problem_log_path, "a") as f:
            f.write("提取 Mid Graph Information 时出错, function_path:"function_path + "\n")
        print("存在问题文件：", function_path)
        return None
    
    # 获取函数 PDG，并按行号排序
    new_line_nodes, new_line_edges = bem.get_func_pdg(pdg_nodes, pdg_start_nodes, pdg_dest_nodes, source_code)
    new_line_nodes = sorted(new_line_nodes) # 按行号排序

    raw_graph_list = []
    sequence_list = []

    for _, line_num in enumerate(new_line_nodes):
        # 获取 raw graph embedding
        # 获取子图
        sub_graph_nodes, sub_graph_edges = bem.get_sub_graph(new_line_nodes, new_line_edges, line_num, ast_nodes, ast_start_nodes, ast_dest_nodes, source_code, hop = 2)

        node_features = []
        # 遍历子图的节点，通过 sent2vec 将每个节点 content 转成 embedding
        for node in sub_graph_nodes:
            content_tokenized = tokenizer.tokenize(node["content"])
            # to know
            vector = sent2vec_model.embed_sentence(" ".join(content_tokenized))[0] # [0] 返回句子列表中的第一个句子的 embedding
            node_features.append(vector.tolist()) # 将 numpy.ndarray 转成 list

        line_graph = {"raw_node_features": node_features, "raw_edges": sub_graph_edges} # 得到该行的图 embedding
        raw_graph_list.append(line_graph)

        # 获取 sequence embedding
        line_content = source_code[line_num-1]
        line_content_tokenized = tokenizer.encode(line_content).tokens
        sequence = sent2vec_model.embed_sentence(" ".join(line_content_tokenized))[0].tolist()

        sequence_list.append(sequence)
    
    return sequence_list, raw_graph_list
    
# 从 C 函数提取的 mid_graph 中获取 normalized graph 类型特征向量
def get_vector_normalized_by_mid_graph(function_path):
    # 保证函数为 .c，报错
    if function_path[-2:] != ".c":
        raise Exception("The file is not a C file!")
    
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(constant.represent_dir + "/normalized_model.bin")

    # 提取函数图信息
    label, source_code, adj_ast, adj_cfg, \
    adj_pdg, ast_nodes, cfg_nodes, pdg_nodes, \
    pdg_start_nodes, pdg_dest_nodes, ast_start_nodes, \
    ast_dest_nodes, problem = bem.extract_graph_info(function_path)

    if problem:
        # 写个问题日志
        with open(constant.problem_log_path, "a") as f:
            f.write("提取 Mid Graph Information 时出错, function_path:"function_path + "\n")
        print("存在问题文件：", function_path)
        return None
    
    # 获取函数 PDG，并按行号排序
    new_line_nodes, new_line_edges = bem.get_func_pdg(pdg_nodes, pdg_start_nodes, pdg_dest_nodes, source_code)
    new_line_nodes = sorted(new_line_nodes) # 按行号排序

    normalized_line_embeddings_list = [] # 所有 line 的 embedding

    for _, line_num in enumerate(new_line_nodes):
        # 获取子图
        sub_graph_nodes, sub_graph_edges = bem.get_sub_graph(new_line_nodes, new_line_edges, line_num, ast_nodes, ast_start_nodes, ast_dest_nodes, source_code, hop = 2)

        node_features = []
        # 遍历子图的节点，通过 sent2vec 将每个节点 content 转成 embedding
        for node in sub_graph_nodes:
            content_tokenized = tokenizer.tokenize(node["content"])
            # to know
            vector = sent2vec_model.embed_sentence(" ".join(content_tokenized))[0] # [0] 返回句子列表中的第一个句子的 embedding
            node_features.append(vector.tolist()) # 将 numpy.ndarray 转成 list

        line_graph = {"normalized_node_features": node_features, "normalized_edges": sub_graph_edges} # 得到该行的图 embedding
        normalized_line_embeddings_list.append(line_graph)
    
    return normalized_line_embeddings_list

# 将图字典转换为 PyG 的 Data 对象
def convert_graph_to_data(graph):
    """
    graph: dict，格式例如：
      {
          "raw_node_features": [[...], [...], ...],
          "raw_edges": [[src1, dst1], [src2, dst2], ...]
      }
    """
    # 将节点特征转换为 tensor
    x = torch.tensor(graph["raw_node_features"], dtype=torch.float)
    
    # 假设 raw_edges 是列表，每个元素 [src, dst]
    # 转换为 tensor，并转置为 shape [2, num_edges]
    edge_index = torch.tensor(graph["raw_edges"], dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index)
    return data

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super().__init__()
        # 第一层 GATConv，输出维度为 hidden_channels，每个 head 的输出维度为 hidden_channels/num_heads（这里简化处理，不严格除头数）
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        # 第二层 GATConv，将多头输出拼接后的维度转为 out_channels
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)
        
    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

# 获取函数的 三个 特征向量：sequence, raw_graph vector, normalized_graph vector
def get_func_all_embeddings(function_path):
    sequence_list, raw_graph_list = get_vector_raw_and_sequence_by_mid_graph(function_path)
    normalized_graph_list = get_vector_normalized_by_mid_graph(function_path)
    
    raw_vector_list = []
    normalized_vector_list = []

    # 将 graph_list 转换为 PyG 的 Data 对象
    for graph in raw_graph_list:
        raw_vector_list.append(convert_graph_to_data(graph))
    batch_raw = Batch.from_data_list(raw_vector_list)

    for graph in normalized_graph_list:
        normalized_vector_list.append(convert_graph_to_data(graph))
    batch_normalized = Batch.from_data_list(normalized_vector_list)

    # 特征维度为 feature_dim
    feature_dim = len(raw_graph_list[0]["raw_node_features"][0])

    # 初始化 GAT 模型
    GAT_128 = GATEncoder(in_channels=feature_dim, hidden_channels=16, out_channels=128, num_heads=8)
    GAT_256 = GATEncoder(in_channels=feature_dim, hidden_channels=32, out_channels=256, num_heads=8)
    GAT_128.eval() # 评估模式
    GAT_256.eval()

    # 计算 raw_graph 和 normalized_graph 的特征向量
    with torch.no_grad():
        raw_pooled_128 = GAT_128(batch_raw.x, batch_raw.edge_index, batch_raw.batch)
        raw_pooled_mean_256 = GAT_256(batch_raw.x, batch_raw.edge_index, batch_raw.batch)
        normalized_pooled_mean_128 = GAT_128(batch_normalized.x, batch_normalized.edge_index, batch_normalized.batch)
        normalized_pooled_mean_256 = GAT_256(batch_normalized.x, batch_normalized.edge_index, batch_normalized.batch)

    # 对所有语句的特征向量再进行一次池化，得到一个向量代表函数
    raw_vec128_mean = torch.mean(raw_pooled_128, dim=0) 
    raw_vec256_mean = torch.mean(raw_pooled_mean_256, dim=0)
    normalized_vec128_mean = torch.mean(normalized_pooled_mean_128, dim=0)
    normalized_vec256_mean = torch.mean(normalized_pooled_mean_256, dim=0)

    raw_vec128_max = torch.max(raw_pooled_128, dim=0)
    raw_vec256_max = torch.max(raw_pooled_mean_256, dim=0)
    normalized_vec128_max = torch.max(normalized_pooled_mean_128, dim=0)
    normalized_vec256_max = torch.max(normalized_pooled_mean_256, dim=0)

    sequence_vec = torch.mean(torch.tensor(sequence_list), dim=0)

    return sequence_vec, raw_vec128_mean, raw_vec256_mean, normalized_vec128_mean, normalized_vec256_mean, raw_vec128_max, raw_vec256_max, normalized_vec128_max, normalized_vec256_max