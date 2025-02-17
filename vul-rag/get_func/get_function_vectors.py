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
import json
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sent2vec_raw_model = "/root/autodl-tmp/MGVUL-RAG/code_embedding/raw_model.bin"
sent2vec_normalized_model = "/root/autodl-tmp/MGVUL-RAG/code_embedding/normalized_model.bin"

# 从 C 函数提取的 mid_graph 中获取 raw graph 和 sequence 类型特征向量
def get_vector_raw_and_sequence_by_mid_graph(graph_path):
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(sent2vec_raw_model)

    # 提取函数图信息
    source_code, adj_ast, adj_cfg, \
    adj_pdg, ast_nodes, cfg_nodes, pdg_nodes, \
    pdg_start_nodes, pdg_dest_nodes, ast_start_nodes, \
    ast_dest_nodes, problem = bem.extract_graph_info(graph_path)

    if problem:
        # 写个问题日志
        with open(constant.problem_log_path, "a") as f:
            f.write(f"提取 Mid Graph Information 时出错, graph_path:{graph_path}\n")
        print("存在问题文件：", graph_path)
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
        line_content_tokenized = tokenizer.tokenize(line_content)
        sequence = sent2vec_model.embed_sentence(" ".join(line_content_tokenized))[0].tolist()
        sequence_list.append(sequence)
    
    return sequence_list, raw_graph_list
    
# 从 C 函数提取的 mid_graph 中获取 normalized graph 类型特征向量
def get_vector_normalized_by_mid_graph(graph_path):
    if not str(graph_path).endswith("mid_graph.txt"):
        raise Exception("The file is not a mid graph.")
    
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(sent2vec_normalized_model)

    # 提取函数图信息
    source_code, adj_ast, adj_cfg, \
    adj_pdg, ast_nodes, cfg_nodes, pdg_nodes, \
    pdg_start_nodes, pdg_dest_nodes, ast_start_nodes, \
    ast_dest_nodes, problem = bem.extract_graph_info(graph_path)

    if problem:
        # 写个问题日志
        with open(constant.problem_log_path, "a") as f:
            f.write(f"提取 Mid Graph Information 时出错, graph_path:{graph_path}\n")
        print("存在问题文件：", graph_path)
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
def convert_graph_to_data(graph, code_type):
    # 将节点特征转换为 tensor
    x = torch.tensor(graph[f"{code_type}_node_features"], dtype=torch.float)
    
    # 假设 raw_edges 是列表，每个元素 [src, dst]
    # 转换为 tensor，并转置为 shape [2, num_edges]
    edge_index = torch.tensor(graph[f"{code_type}_edges"], dtype=torch.long).t().contiguous()
    
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
def get_func_all_embeddings(fun_path):
    fun_name = fun_path.split("/")[-1].split(".")[0]
    CWE_ID = fun_path.split("/")[-2]
    raw_graph_path = Path(constant.vul_rag_normalized) / "raw_mid_graphs" / CWE_ID / (fun_name + "_mid_graph.txt")
    normalized_graph_path = Path(constant.vul_rag_normalized) / "normalized_mid_graphs" / CWE_ID / (fun_name + "_mid_graph.txt")

    if not os.path.exists(raw_graph_path) or not os.path.exists(normalized_graph_path):
        print("graph 文件不存在：", raw_graph_path, normalized_graph_path)
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(sent2vec_normalized_model)
        sequence_list = []
        with open(fun_path, "r") as f:
            for line in f:
                line_content = line.strip()
                line_content_tokenized = tokenizer.tokenize(line_content)
                sequence = sent2vec_model.embed_sentence(" ".join(line_content_tokenized))[0].tolist()
                sequence_list.append(sequence)

        sequence_vec = torch.mean(torch.tensor(sequence_list), dim=0)

        with open(constant.problem_log_path, "a") as f:
            f.write(f"graph 不存在:{raw_graph_path, normalized_graph_path}\n")

        return sequence_vec, None, None, None, None, None, None, None, None
    try:
        sequence_list, raw_graph_list = get_vector_raw_and_sequence_by_mid_graph(raw_graph_path)
        normalized_graph_list = get_vector_normalized_by_mid_graph(normalized_graph_path)
    except:
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(sent2vec_normalized_model)
        sequence_list = []
        with open(fun_path, "r") as f:
            for line in f:
                line_content = line.strip()
                line_content_tokenized = tokenizer.tokenize(line_content)
                sequence = sent2vec_model.embed_sentence(" ".join(line_content_tokenized))[0].tolist()
                sequence_list.append(sequence)
        sequence_vec = torch.mean(torch.tensor(sequence_list), dim=0)
        return sequence_vec, None, None, None, None, None, None, None, None

    raw_vector_list = []
    normalized_vector_list = []

    if not raw_graph_list:
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(sent2vec_normalized_model)
        sequence_list = []
        with open(fun_path, "r") as f:
            for line in f:
                line_content = line.strip()
                line_content_tokenized = tokenizer.tokenize(line_content)
                sequence = sent2vec_model.embed_sentence(" ".join(line_content_tokenized))[0].tolist()
                sequence_list.append(sequence)
        sequence_vec = torch.mean(torch.tensor(sequence_list), dim=0)
        return sequence_vec, None, None, None, None, None, None, None, None

    # 将 graph_list 转换为 PyG 的 Data 对象
    for graph in raw_graph_list:
        raw_vector_list.append(convert_graph_to_data(graph, "raw"))
    batch_raw = Batch.from_data_list(raw_vector_list)

    for graph in normalized_graph_list:
        normalized_vector_list.append(convert_graph_to_data(graph, "normalized"))
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

    # return sequence_vec, raw_vec128_mean, normalized_vec128_mean, raw_vec128_max, normalized_vec128_max
    return  sequence_vec, raw_vec128_mean, raw_vec256_mean, \
            normalized_vec128_mean, normalized_vec256_mean, \
            raw_vec128_max, raw_vec256_max, \
            normalized_vec128_max, normalized_vec256_max

def main():
    for dir in os.listdir(constant.vul_rag_normalized + "/normalized_code_files"):
        if not dir.startswith("CWE-"):
            continue
        print("开始转换文件夹: {}".format(dir))
        dir_path = constant.vul_rag_normalized + "/normalized_code_files" + "/" + dir
        if os.path.isdir(dir_path):
            count = 0
            if not dir.startswith("CWE-"):
                continue

            CWE_ID = dir
            if not (CWE_ID == "CWE-416"):
                continue
            json_path = constant.vul_rag_vul_knowledge_with_id_dir + "/" + f"gpt-3.5-turbo_{CWE_ID}_316_with_id.json"
            # save_path = Path(constant.vulnerability_knowledge_with_vectors_dir) / f"{CWE_ID}_with_vectors.json"
            temp_save_path = Path(constant.vulnerability_knowledge_with_vectors_dir) / f"temp_{CWE_ID}_with_vectors.csv"

            # 所有 id
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            id_list = []
            for CVE_ID, CVE_LIST in data.items():  
                for item in CVE_LIST:    
                    id = item["id"]
                    id_list.append(id)

        # 如果文件不存在，创建一个空的 CSV 文件并写入表头
        if not os.path.exists(temp_save_path):
            df_header = pd.DataFrame(columns=["id", "sequence_vec", "raw_vec128_mean", "raw_vec256_mean",
                                            "normalized_vec128_mean", "normalized_vec256_mean",
                                            "raw_vec128_max", "raw_vec256_max",
                                            "normalized_vec128_max", "normalized_vec256_max"])
            df_header.to_csv(temp_save_path, index=False)
        id_list = sorted(id_list)
        df = pd.read_csv(temp_save_path)

        # 遍历 id_list 并处理每个 ID
        for id in id_list:
            print("{} 进度: {}/{}".format(dir_path, id, len(id_list)))
            print("当前时间：", os.popen("date").read())

            if not df["id"].str.contains(str(id) + "_before").any():
                print("处理 before.c 文件")
                # 处理 before.c 文件
                fun_path_before = os.path.join(dir_path, f"{id}_before.c")
                
                sequence_vec, raw_vec128_mean, raw_vec256_mean, \
                normalized_vec128_mean, normalized_vec256_mean, \
                raw_vec128_max, raw_vec256_max, \
                normalized_vec128_max, normalized_vec256_max = get_func_all_embeddings(fun_path_before)

                # sequence_vec, raw_vec128_mean, normalized_vec128_mean, \
                # raw_vec128_max, normalized_vec128_max = get_func_all_embeddings(fun_path_before)

                # 创建新行数据
                new_row_before = {
                    "id": str(id) + "_before",
                    "sequence_vec": sequence_vec,
                    "raw_vec128_mean": raw_vec128_mean,
                    "raw_vec256_mean": raw_vec256_mean,
                    "normalized_vec128_mean": normalized_vec128_mean,
                    "normalized_vec256_mean": normalized_vec256_mean,
                    "raw_vec128_max": raw_vec128_max,
                    "raw_vec256_max": raw_vec256_max,
                    "normalized_vec128_max": normalized_vec128_max,
                    "normalized_vec256_max": normalized_vec256_max
                }
                # 将新行数据追加保存到文件
                pd.DataFrame([new_row_before]).to_csv(temp_save_path, mode='a', header=False, index=False)

            if not df["id"].str.contains(str(id) + "_after").any():
                # 处理 after.c 文件
                print("处理 after.c 文件")
                fun_path_after = os.path.join(dir_path, f"{id}_after.c")
                
                sequence_vec, raw_vec128_mean, raw_vec256_mean, \
                normalized_vec128_mean, normalized_vec256_mean, \
                raw_vec128_max, raw_vec256_max, \
                normalized_vec128_max, normalized_vec256_max = get_func_all_embeddings(fun_path_after)

                # sequence_vec, raw_vec128_mean, normalized_vec128_mean, \
                # raw_vec128_max, normalized_vec128_max = get_func_all_embeddings(fun_path_after)
                
                # 创建新行数据
                new_row_after = {
                    "id": str(id) + "_after",
                    "sequence_vec": sequence_vec,
                    "raw_vec128_mean": raw_vec128_mean,
                    "raw_vec256_mean": raw_vec256_mean,
                    "normalized_vec128_mean": normalized_vec128_mean,
                    "normalized_vec256_mean": normalized_vec256_mean,
                    "raw_vec128_max": raw_vec128_max,
                    "raw_vec256_max": raw_vec256_max,
                    "normalized_vec128_max": normalized_vec128_max,
                    "normalized_vec256_max": normalized_vec256_max
                }
                # 将新行数据追加保存到文件
                pd.DataFrame([new_row_after]).to_csv(temp_save_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    main()
    