from pathlib import Path
import subprocess
import shutil
import re
import codecs
from typing import List
import json
import os
import pandas as pd
import common.constant as constant
import subprocess
import build_embedding_matrix as bem
import sent2vec
from code_embedding.qwenCoder_tokenize import tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sent2vec_raw_model = constant.sent2vec_raw_model
sent2vec_normalized_model = constant.sent2vec_normalized_model

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
def get_func_all_embeddings(code_dir):
    raw_func_path = Path(code_dir) / 'raw' / 'code.c'
    raw_graph_path = Path(code_dir) / 'raw' / 'mid_graph.txt'
    normalized_graph_path = Path(code_dir) / 'normalized' / 'mid_graph.txt'

    if not os.path.exists(raw_graph_path) or not os.path.exists(normalized_graph_path):
        print("graph 文件不存在：", raw_graph_path, normalized_graph_path)
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(sent2vec_normalized_model)
        sequence_list = []
        with open(raw_func_path, "r") as f:
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
        with open(raw_func_path, "r") as f:
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
        with open(raw_func_path, "r") as f:
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

def nodeInformation(nodeList): 
    # 使用正则表达式按模式 "(数字," 分割字符串
    new_inf_list = re.split('[(]\d+,', nodeList)
    # 提取所有匹配的节点编号部分
    num_list = re.findall('[(]\d+,', nodeList)
    new_node_list = []  # 保存处理后的节点信息
    for i in range(0, len(num_list)): 
        if i == len(num_list) - 1:
            # 最后一个节点，直接拼接编号和内容
            new_node = num_list[i] + new_inf_list[i + 1]  # i+1 是因为 new_inf_list 的第一个元素是空字符串
        else:
            # 非最后一个节点，去掉末尾多余的字符
            new_node = num_list[i] + new_inf_list[i + 1][:-2] 
        new_node_list.append(new_node)
    return new_node_list

# 主函数，用于处理 joern_parse 生成的图数据
def joernGraph(code_dir):
    input_path = Path(code_dir) / 'graph_info.txt'
    output = Path(code_dir) / 'mid_graph.txt'

    # 初始化存储 AST、CFG 和 PDG 边的列表
    ast_node_relation = []  # 存储 AST 的边
    cfg_node_relation = []  # 存储 CFG 的边
    pdg_node_relation = []  # 存储 PDG 的边

    path = input_path
    with open(path, 'r', encoding='utf-8') as f:  # 打开文件
        lines = str(f.read())  # 读取文件内容
        alllist = lines.split("),List(")  # 按 "),List(" 分割内容

        # 初始化存储节点和边的列表
        ast_node1 = []
        ast_node2 = []
        ast_relation = []
        cfg_node1 = []
        cfg_node2 = []
        cfg_relation = []
        pdg_node1 = []
        pdg_node2 = []
        pdg_realtion = []

        # 判断数据是否为空，且文件名符合要求，也就是看最主要的函数
        if (alllist[1] != "" and alllist[0].find(".c-<global>") < 0 and alllist[0].find(".c") > 0 and alllist[0].find(".c-VAR") < 0):
            filename = alllist[0].split("/")[-1].split(".c")[0]
            
            # AST 添加边
            ast_node_relation.append(alllist[1])
            # AST 添加节点
            ast_node_info = alllist[2]

            # CFG 添加边
            cfg_node_relation.append(alllist[3])
            # CFG 添加节点
            cfg_node_info = alllist[4]

            # PDG 添加边
            pdg_node_relation.append(alllist[5])
            # PDG 添加节点
            pdg_node_info = alllist[6][:-3]

            # 使用正则表达式提取边信息
            ast_node_relation = re.findall(r"\(\d*,\d*,\d*\)", str(ast_node_relation))
            ast_node_info = nodeInformation(ast_node_info)
            cfg_node_relation = re.findall(r"\(\d*,\d*,\d*\)", str(cfg_node_relation))
            cfg_node_info = nodeInformation(cfg_node_info)
            pdg_node_relation = re.findall(r"\(\d*,\d*,\d*\)", str(pdg_node_relation))
            pdg_node_info = nodeInformation(pdg_node_info)

            # 将三种边信息拆分为起点、终点和关系类型，操作都相同
            ast_node_relation = ' '.join(ast_node_relation) # 将列表转换为字符串，以空格分隔
            ast_batch = re.findall('\d+', ast_node_relation) # 使用正则表达式提取所有边(a, b, c)的数字
            for i in range(0, len(ast_batch), 3): # 每三个数字为一组
                ast_node1.append(ast_batch[i]) # 第一个数字为起点
                ast_node2.append(ast_batch[i + 1]) # 第二个数字为终点
                ast_relation.append(ast_batch[i + 2]) # 第三个数字为关系类型

            cfg_node_relation = ' '.join(cfg_node_relation)
            cfgBatch = re.findall('\d+', cfg_node_relation)
            for i in range(0, len(cfgBatch), 3):
                cfg_node1.append(cfgBatch[i])
                cfg_node2.append(cfgBatch[i + 1])
                cfg_relation.append(cfgBatch[i + 2])

            pdg_node_relation = ' '.join(pdg_node_relation)
            pdgBatch = re.findall('\d+', pdg_node_relation)
            for i in range(0, len(pdgBatch), 3):
                pdg_node1.append(pdgBatch[i])
                pdg_node2.append(pdgBatch[i + 1])
                pdg_realtion.append(pdgBatch[i + 2])

            # 提取节点信息（编号、语义、行号）
            ast_nodes = []
            ast_means = []
            ast_lines = []

            cfg_nodes = []
            cfg_means = []
            cfg_lines = []

            pdg_nodes = []
            pdg_means = []
            pdg_lines = []

            # 遍历节点信息，提取节点编号、语义和行号
            # e.g. ast_node_info = ['(18,1)', '(16,b+1)', '(15,malloc(b+1))', '(14,char*)']
            for i in range(0, len(ast_node_info)):
                # e.g. ast_node_info[i] = '(15,malloc(b+1))'
                ast_node = re.match('[(]\d+', ast_node_info[i]).group()[1:]  # 提取节点编号
                # e.g. ast_ml = 'malloc(b+1)'
                ast_ml = ast_node_info[i][len(ast_node) + 2:-1]  # 提取节点内容，去掉编号和括号
                ast_line = ast_ml.split(",")[-1]  # 提取行号
                ast_mean = ast_ml[:len(ast_ml) - len(ast_line) - 1]  # 提取语义
                ast_nodes.append(ast_node)
                ast_means.append(ast_mean)
                ast_lines.append(ast_line)

            for i in range(0, len(cfg_node_info)):
                cfg_node = re.match('[(]\d+', cfg_node_info[i]).group()[1:]
                cfg_ml = cfg_node_info[i][len(cfg_node) + 2:-1]
                cfg_line = cfg_ml.split(",")[-1]
                cfg_mean = cfg_ml[:len(cfg_ml) - len(cfg_line) - 1]
                cfg_nodes.append(cfg_node)
                cfg_means.append(cfg_mean)
                cfg_lines.append(cfg_line)

            for i in range(0, len(pdg_node_info)):
                pdg_node = re.match('[(]\d+', pdg_node_info[i]).group()[1:]
                pdg_ml = pdg_node_info[i][len(pdg_node) + 2:-1]
                pdg_line = pdg_ml.split(",")[-1]
                pdg_mean = pdg_ml[:len(pdg_ml) - len(pdg_line) - 1]
                pdg_nodes.append(pdg_node)
                pdg_means.append(pdg_mean)
                pdg_lines.append(pdg_line)

            # 重新编号节点, node1 和 node2 分别为边的起点和终点
            # ast_new_node1 和 ast_new_node2 为重新编号后的边的起点和终点, 新编号为节点在节点列表中的位置！！！！！！！！！
            # 点 ast_node1[i] 的新编号为 ast_new_node1[i]
            ast_new_node1 = []
            ast_new_node2 = []
            ast_new_nodes = list(range(0, len(ast_nodes))) # 生成节点编号列表
            for x in ast_node1: # 遍历边的起点
                for i in range(len(ast_nodes)): # 遍历节点编号列表
                    if x == ast_nodes[i]: # 同一个
                        ast_new_node1.append(str(i)) # 找到边的起点在节点编号列表中的位置
                        break
            for x in ast_node2:
                for i in range(len(ast_nodes)):
                    if x == ast_nodes[i]:
                        ast_new_node2.append(str(i)) # 找到边的终点在节点编号列表中的位置
                        break

            cfg_new_node1 = []
            cfg_new_node2 = []
            cfg_new_nodes = list(range(0, len(cfg_nodes)))
            for x in cfg_node1:
                for i in range(len(cfg_nodes)):
                    if x == cfg_nodes[i]:
                        cfg_new_node1.append(str(i))
                        break
            for x in cfg_node2:
                for i in range(len(cfg_nodes)):
                    if x == cfg_nodes[i]:
                        cfg_new_node2.append(str(i))
                        break

            pdg_new_node1 = []
            pdg_new_node2 = []
            pdg_new_nodes = list(range(0, len(pdg_nodes)))
            for x in pdg_node1:
                for i in range(len(pdg_nodes)):
                    if x == pdg_nodes[i]:
                        pdg_new_node1.append(str(i))
                        break
            for x in pdg_node2:
                for i in range(len(pdg_nodes)):
                    if x == pdg_nodes[i]:
                        pdg_new_node2.append(str(i))
                        break

            code_path = Path(code_dir) / 'code.c'
            with open(code_path, 'r', encoding='utf-8') as f1:
                code = f1.read()

            # 创建输出目录
            os.makedirs(code_dir, exist_ok=True)

            # 写入文件
            with open(output, 'w', encoding='utf-8') as f2:
                
                f2.write("-----Code-----\n")
                f2.write(code + "\n")
                f2.write("-----AST-----\n")
                for x, y in zip(ast_new_node1, ast_new_node2): 
                    f2.write(x + ',' + y + "\n")
                f2.write("-----AST_Node-----\n")
                for x, y, z in zip(ast_new_nodes, ast_means, ast_lines):
                    f2.write(str(x) + '|||' + y.replace('\n', '') + '|||' + str(z) + "\n")
                f2.write("-----CFG-----\n")
                for x, y in zip(cfg_new_node1, cfg_new_node2):
                    f2.write(x + ',' + y + "\n")
                f2.write("-----CFG_Node-----\n")
                for x, y in zip(cfg_new_nodes, cfg_means):
                    f2.write(str(x) + '|||' + y.replace('\n', '') + "\n")
                f2.write("-----PDG-----\n")
                for x, y in zip(pdg_new_node1, pdg_new_node2):
                    f2.write(x + ',' + y + "\n")
                f2.write("-----PDG_Node-----\n")
                for x, y, z in zip(pdg_new_nodes, pdg_means, pdg_lines):
                    f2.write(str(x) + '|||' + y.replace('\n', '') + '|||' + str(z) + "\n")
                f2.write("-----End-----\n")
        

def to_code_file(source_code, save_dir):
    save_path = save_dir / "code.c"
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(source_code)

def joern_parse(code_dir): 
    print("开始转换 code.c 到 CPGs")

    input = str(code_dir / 'code.c')

    output = str(Path(code_dir) / f"cpg.bin")
    os.makedirs(code_dir, exist_ok=True)

    joern_path = constant.joern_path
    os.chdir(joern_path)
    
    os.environ['input'] = input
    os.environ['output'] = output

    print("input: {}\noutput: {}".format(input, output))

    process = subprocess.Popen('sh joern-parse $input --out $output',    
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=True, close_fds=True)
    output = process.communicate()

keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL', 'printf', 'STR'})
# holds known non-user-defined functions; immutable set
main_set = frozenset({'main'})
# arguments in main function; immutable set
main_args = frozenset({'argc', 'argv'})

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~'
}

def to_regex(lst): # 将 lst 中的元素转换为正则表达式
    return r'|'.join([f"({re.escape(el)})" for el in lst])

regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)

def _removeComments(source): # 移除源代码每一行的注释 // 和 /* */
    in_block = False # 是否在注释块中
    new_source = []
    # source = source.split('\n')
    for line in source:
        i = 0
        if not in_block: # 如果不在注释块中
            newline = []
        while i < len(line): # 遍历该行
            if line[i:i + 2] == '/*' and not in_block: # 如果遇到 /*
                in_block = True
                i += 1
            elif line[i:i + 2] == '*/' and in_block: # 如果遇到 */
                in_block = False
                i += 1
            elif not in_block and line[i:i + 2] == '//': # 如果遇到 //，就停止存储剩余部分
                break
            elif not in_block:
                newline.append(line[i])
            i += 1
        if newline and not in_block:
            new_source.append("".join(newline))
    return new_source

def clean_gadget(gadget): 
    # dictionary; map function name to symbol name + number
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to find function name candidates
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()') # 匹配函数名
    # regular expression to find variable name candidates
    # rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b((?!\s*\**\w+))(?!\s*\()') # 匹配变量名，第一个匹配变量名，第二个匹配指针

    # final cleaned gadget output to return to interface
    cleaned_gadget = []

    for line in gadget: # 遍历每一行
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', line) # 将 line 中的非 ASCII 字符替换为空字符串
        hex_line = re.sub(r'0[xX][0-9a-fA-F]+', "HEX", ascii_line) # 将 line 中的十六进制数字替换为 HEX
        user_fun = rx_fun.findall(hex_line) # 在 hex_line 中匹配函数名
        user_var = rx_var.findall(hex_line) # 在 hex_line 中匹配变量名

        
        for fun_name in user_fun:
            # 如果不是 main 函数和关键字
            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                if fun_name not in fun_symbols.keys(): # 如果函数名不在字典中，在字典中就说明之前存过了
                    fun_symbols[fun_name] = 'FUN' + str(fun_count) # 将函数名映射为 FUN#，# 为计数
                    fun_count += 1 # 计数加一
                hex_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], hex_line) # 将函数名替换为 FUN#

        for var_name in user_var: # 遍历变量名
            # next line is the nuanced difference between fun_name and var_name
            if len({var_name[0]}.difference(keywords)) != 0 and len({var_name[0]}.difference(main_args)) != 0:
                # check to see if variable name already in dictionary
                if var_name[0] not in var_symbols.keys(): # var_name 的部分是变量名，如果变量名不在字典中
                    var_symbols[var_name[0]] = 'VAR' + str(var_count) # 将变量名映射为 VAR#，# 为计数
                    var_count += 1 # 计数加一
                # ensure that only variable name gets replaced (no function name with same
                # identifier); uses negative lookforward
                # print(var_name, gadget, user_var)
                hex_line = re.sub(r'\b(' + var_name[0] + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', 
                                  var_symbols[var_name[0]], hex_line) # 将变量名替换为 VAR#

        cleaned_gadget.append(hex_line) # 将处理后的行添加到 cleaned_gadget 中
    # return the list of cleaned lines
    return cleaned_gadget 

def normalize_code(source_code, need_normalize):
    gadget: List[str] = []

    if need_normalize:
        try:
            no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '"STR"', source_code) # 将字符串替换为 STR
        except:
            print("Error in string literal replacement")
            print(clean)
        no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line) # 将字符替换为""
        source_code = no_char_lit_line
    for line in source_code.splitlines(): # 遍历处理完字符和字符串的代码的每一行
        if line == '':
            continue
        stripped = line.strip() # 去除首尾空格
        gadget.append(stripped) # 将处理完字符和字符串的代码行添加到 gadget 中
    clean = _removeComments(gadget)
    if need_normalize:
        clean = clean_gadget(clean)
    
    dest_code = ""
    for line in clean:
        dest_code += line + '\n'

    return dest_code

def extract_graph_from_CPG(code_dir):
    joern_path = constant.joern_path
    script_file = constant.all_script

    print("开始转换 CPGs 到 graphs")

    # 设置输入参数
    input_path = Path(code_dir) / 'cpg.bin'

    if not input_path.exists():
        with open(constant.problem_log_path, "a") as f:
            f.write(f"{code_dir} 的 cpg.bin not exists\n")
        return
    
    os.chdir(joern_path)
    print("current input file:", input_path)

    # 设置输出参数，创建临时输出文件夹，提取要的主函数
    out_dir = Path(code_dir) / f"temp_graph_info"
    os.makedirs(out_dir, exist_ok=True)

    params = f"cpgFile={input_path},outDir={out_dir}"
    os.environ['params'] = str(params)
    os.environ['script_file'] = str(script_file)

    process = subprocess.Popen('sh joern --script $script_file --params $params',    
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=True, close_fds=True)
    output = process.communicate()
    print(output)

    # 筛选
    for root, _, files in os.walk(out_dir):
        for file in files:
            with open(Path(root) / file, 'r', encoding='utf-8') as f:  # 打开文件
                lines = str(f.read())  # 读取文件内容
            if lines.startswith("(Some(/Users"):
                shutil.move(Path(out_dir) / file, code_dir / f"graph_info.txt")
    shutil.rmtree(out_dir) # 删除临时文件夹



# 得到代码的向量，并将所有数据都保存到 save_dir 中，save_dir 下有 raw 和 normalized 两个文件夹
def get_func_vector(source_code, save_dir):
    # 初始化保存文件夹
    normalized_dir = save_dir / "normalized"
    raw_dir = save_dir / "raw"
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # 先 normalize 代码
    normalized_code = normalize_code(source_code, True)
    raw_code = normalize_code(source_code, False)

    # 保存到文件夹中，并获取 CPG
    to_code_file(normalized_code, normalized_dir)
    to_code_file(raw_code, raw_dir)
    joern_parse(normalized_dir)
    joern_parse(raw_dir)

    # 提取 CPG 的 AST、PDG、CFG
    extract_graph_from_CPG(normalized_dir)
    extract_graph_from_CPG(raw_dir)

    # 从 graph_info.txt 中提取 mid graph
    try:
        joernGraph(normalized_dir)
        # joernGraph(raw_dir)
    except:
        with open(constant.problem_log_path, "a") as f:
            f.write(f"{normalized_dir} 从 graph_info.txt 中提取 mid graph 失败\n")

    try:
        # joernGraph(normalized_dir)
        joernGraph(raw_dir)
    except:
        with open(constant.problem_log_path, "a") as f:
            f.write(f"{raw_dir} 从 graph_info.txt 中提取 mid graph 失败\n")

    # 从 mid_graph.txt 中提取向量
    try:
        sequence_vec, raw_vec128_mean, raw_vec256_mean, \
        normalized_vec128_mean, normalized_vec256_mean, \
        raw_vec128_max, raw_vec256_max, \
        normalized_vec128_max, normalized_vec256_max = get_func_all_embeddings(save_dir)

        # print("sequence_vec: ", sequence_vec)
        # print("raw_vec128_mean: ", raw_vec128_mean)
        # print("raw_vec256_mean: ", raw_vec256_mean)
        # print("normalized_vec128_mean: ", normalized_vec128_mean)
        # print("normalized_vec256_mean: ", normalized_vec256_mean)
        # print("raw_vec128_max: ", raw_vec128_max)
        # print("raw_vec256_max: ", raw_vec256_max)
        # print("normalized_vec128_max: ", normalized_vec128_max)
        # print("normalized_vec256_max: ", normalized_vec256_max)

        # 保存到 json
        json_path = Path(save_dir)  / "func_vectors.json"
        with open(json_path, "w") as f:
            json.dump({
                "sequence_vec": sequence_vec.tolist(),
                "raw_vec128_mean": raw_vec128_mean.tolist(),
                "raw_vec256_mean": raw_vec256_mean.tolist(),
                "normalized_vec128_mean": normalized_vec128_mean.tolist(),
                "normalized_vec256_mean": normalized_vec256_mean.tolist(),
                "raw_vec128_max": raw_vec128_max.values.tolist(),
                "raw_vec256_max": raw_vec256_max.values.tolist(),
                "normalized_vec128_max": normalized_vec128_max.values.tolist(),
                "normalized_vec256_max": normalized_vec256_max.values.tolist()
            }, f)
    except:
        with open(constant.problem_log_path, "a") as f:
            f.write(f"testset 的 {save_dir} 解析失败\n")

def main():
    test_dir = Path(constant.vul_rag_test_set)
    for testset in os.listdir(test_dir):
        if not testset.endswith('_testset.json'):
            continue
        CWE_ID = testset.split('_')[0]
        save_dir = Path(constant.vul_rag_test_set) / 'vectors' / CWE_ID
        os.makedirs(save_dir, exist_ok=True)

        # 获取测试集代码的向量
        with open(test_dir / testset, 'r') as f:
            test_data = json.load(f)

        total = len(test_data['non_vul_data']) + len(test_data['vul_data'])
        count = 0
        for _, list in test_data.items():
            for item in list:
                source_code = item['code_snippet']
                code_dir = save_dir / str(item['id'])
                count += 1
                dest_path = code_dir / 'func_vectors.json'
                if dest_path.exists():
                    continue
                print("testset: {}".format(CWE_ID))
                print("进度：{}/{}".format(count, total))
                os.makedirs(code_dir, exist_ok=True)
                get_func_vector(source_code, code_dir)

if __name__ == "__main__":
    main()