import os
import sent2vec
import jsonlines
from common.tool.progress import print_progress
from code_embedding.qwenCoder_tokenize import tokenizer
import common.constant as constant
from pathlib import Path

# process graph further
# Author https://github.com/Lifeasarain/MGVD/tree/75abb76f1a89a472bc539a0e2d29195331580ba1
# refered to the code of MGVD，and fixed some bugs in function `get_sub_graph`
# TODO test `get_sub_graph` function

def extract_graph_info(file_path):
    problem = False

    source_code = []  # 源代码内容
    adj_ast = []  # AST 边，得到：(start, dest)
    adj_cfg = []  # CFG 边
    adj_pdg = []  # PDG 边
    ast_nodes = []  # AST 节点，得到：("content", "line_num")
    cfg_nodes = []  # CFG 节点
    pdg_nodes = []  # PDG 节点
    ast_start_nodes = []  # AST 起始节点
    ast_dest_nodes = []  # AST 目标节点
    pdg_start_nodes = []  # PDG 起始节点
    pdg_dest_nodes = []  # PDG 目标节点
    label = ""  # 标签
    label_label = False  # 是否正在读取标签
    label_code = False  # 是否正在读取代码
    label_ast = False  # 是否正在读取 AST 边
    label_ast_node = False  # 是否正在读取 AST 节点
    label_cfg = False  # 是否正在读取 CFG 边
    label_cfg_node = False  # 是否正在读取 CFG 节点
    label_pdg = False  # 是否正在读取 PDG 边
    label_pdg_node = False  # 是否正在读取 PDG 节点

    with open(file_path, "r") as f:
        data = f.readlines()
        for line in data:
            
            # 当前在读什么，按顺序一个一个读
            if line.find("-----Label-----") >= 0:
                label_label = True
                continue
            if label_label:
                label = line.replace('\n', "") # 将这行的换行符去掉，并把这行的内容赋值给 label
                label_label = False

            if line.find("-----Code-----") >= 0:
                label_code = True
                continue

            if label_code:
                if line.find("-----AST-----") >= 0:
                    label_code = False
                    label_ast = True
                    continue
                else:
                    source_code.append(line)
                    continue

            if label_ast:
                if line.find("-----AST_Node-----") >= 0:
                    label_ast = False
                    label_ast_node = True
                    continue
                else:
                    edges = line.split(",")
                    adj_ast.append((int(edges[0]), int(edges[1])))
                    ast_start_nodes.append(int(edges[0]))
                    ast_dest_nodes.append(int(edges[1]))
                    continue

            if label_ast_node:
                if line.find("-----CFG-----") >= 0:
                    label_ast_node = False
                    label_cfg = True
                    continue
                else:
                    nodes = line.split('\n')[0].split('|||')
                    try:
                        ast_nodes.append({"content": nodes[1], "line_num": nodes[2]})
                    except:
                        # print(file_path)
                        problem = True
                        error_node = {"content": "<ERROR>", "line_num": "<ERROR>"}
                        ast_nodes.append(error_node)
                    continue

            if label_cfg:
                if line.find("-----CFG_Node-----") >= 0:
                    label_cfg = False
                    label_cfg_node = True
                    continue
                else:
                    edges = line.strip().split(",")
                    adj_cfg.append((int(edges[0]), int(edges[1])))
                    continue

            if label_cfg_node:
                if line.find("-----PDG-----") >= 0:
                    label_cfg_node = False
                    label_pdg = True
                    continue
                else:
                    nodes = line.split('\n')[0].split('|||')
                    cfg_nodes.append(nodes[1])
                    continue

            if label_pdg:
                if line.find("-----PDG_Node-----") >= 0:
                    label_pdg = False
                    label_pdg_node = True
                    continue
                else:
                    edges = line.strip().split(",")
                    adj_pdg.append((int(edges[0]), int(edges[1])))
                    pdg_start_nodes.append(int(edges[0]))
                    pdg_dest_nodes.append(int(edges[1]))
                    continue

            if label_pdg_node:
                if line.find("-----End-----") >= 0: # 结束
                    label_pdg_node = False
                    break
                else:
                    nodes = line.split('\n')[0].split('|||')
                    try:
                        pdg_nodes.append({"content": nodes[1], "line_num": nodes[2]})
                    except:
                        problem = True
                        error_node = {"content": "<ERROR>", "line_num": "<ERROR>"}
                        pdg_nodes.append(error_node)
                    continue

    return label, source_code, adj_ast, adj_cfg, adj_pdg, \
           ast_nodes, cfg_nodes, pdg_nodes, pdg_start_nodes, pdg_dest_nodes, \
           ast_start_nodes, ast_dest_nodes, problem

# 获取某行的 AST
# line_ast_nodes = {idx: content}，idx 是 ast_nodes 原始索引，content 是节点的内容
# line_ast_edges = [(start, dest)]，start 和 dest 是原始索引
def get_line_ast(line_num, ast_nodes, ast_start_nodes, ast_dest_nodes, source_code):
    line_ast_nodes = {}
    line_ast_edges = [] 
    root_node = {}

    # 遍历所有的 AST 节点，找到属于该行的子树 和 根
    for idx, node in enumerate(ast_nodes):
        content = node["content"]
        # 此节点属于该行的子树【如果行号和节点的行号相同，或者源代码的该行包含此节点的内容】
        if str(line_num) == node["line_num"] or source_code[line_num - 1].find(content) > 0:
            line_ast_nodes[idx] = content
            # 索引 idx 是根节点【如果内容和源代码该行的内容相同（AST 节点的内容可能会不包含最后一个字符）】
            if content == source_code[line_num-1].strip().replace("\n","")[:-1] or content == source_code[line_num-1].strip().replace("\n",""):
                root_node[idx] = content

    # 遍历子树的节点 node_idx，找到所有包含 node_idx，且终点也在子树中的边
    for node_idx in line_ast_nodes:
        # 遍历所有的以 node_idx 为起始节点的边
        for sidx, sn in enumerate(ast_start_nodes):
            if sn == node_idx:
                to_node_idx = ast_dest_nodes[sidx]
                if to_node_idx in line_ast_nodes: # 如果终点也包含在子树中
                    line_ast_edges.append((node_idx, to_node_idx))
        
        # 遍历所有的以 node_idx 为目标节点的边
        for didx, dn in enumerate(ast_dest_nodes):
            if dn == node_idx:
                from_node_idx = ast_start_nodes[didx]
                if from_node_idx in line_ast_nodes:
                    line_ast_edges.append((from_node_idx, node_idx))

    line_ast_edges = list(set(line_ast_edges)) # 去重
    return line_ast_nodes, line_ast_edges, root_node

# 获得函数的 PDG
# new_line_nodes = {line_num: content}，line_num 是行号，content 是行内容
# new_line_edges = [(start, dest)]，start 和 dest 是行号
def get_func_pdg(pdg_nodes, pdg_start_nodes, pdg_dest_nodes, source_code):
    new_line_nodes = {}
    new_line_edges = []

    for idx, node in enumerate(pdg_nodes):
        from_node_idxs = [] # 以 idx 为终止节点的边的起始节点的 pdg 原始索引
        to_node_idxs = [] # 以 idx 为起始节点的边的终点节点的pdg 原始索引
        from_line_nums = [] # 以 idx 为终止节点的边的起始节点的行号
        to_line_nums = [] # 以 idx 为起始节点的边的终点节点的行号
        content = node["content"]
        line_num = node["line_num"]
        try:
            int(line_num) # 将行号转换为整数，如果不能转换，可能是无效数据
        except:
            continue
        source_content = source_code[int(line_num) - 1] # 源代码的该行内容
        # 遍历所有的 PDG 边，找到以 idx 为起始节点的边，保存终点节点的索引和行号
        for sidx, sn in enumerate(pdg_start_nodes):
            if sn == idx: # 以 idx 为起始节点的边
                to_node_idx = pdg_dest_nodes[sidx] # 对应的终点节点索引
                to_node_num = pdg_nodes[to_node_idx]["line_num"] # 对应的终点节点行号
                try:
                    int(to_node_num) # 将行号转换为整数，如果不能转换，可能是无效数据
                except:
                    continue
                to_node_idxs.append(to_node_idx)
                to_line_nums.append(to_node_num)

        for didx, dn in enumerate(pdg_dest_nodes):
            if dn == idx:
                from_node_idx = pdg_start_nodes[didx]
                from_node_num = pdg_nodes[from_node_idx]["line_num"]
                try:
                    int(from_node_num)
                except:
                    continue
                from_node_idxs.append(from_node_idx)
                from_line_nums.append(from_node_num)

        to_line_nums = set(to_line_nums) # 去重
        from_line_nums = set(from_line_nums) # 去重

        # 该行已经被保存过一次了，说明该行还有其他 pdg node。所以该行内容不用再保存，只需要存与该 node 相关的 pdg 边
        if line_num in new_line_nodes:
            if source_content.find(content) >= 0:
                # 将所有与该节点相关的边添加到 new_line_edges 中
                for tl in to_line_nums:
                    new_line_edges.append((int(line_num), int(tl)))
                for fl in from_line_nums:
                    new_line_edges.append((int(fl), int(line_num)))
            else:
                continue
        
        # 行号不在 new_line_nodes 中，且源代码的该行包含节点内容
        if line_num not in new_line_nodes:
            if source_content.find(content) >= 0:
                new_line_nodes[int(line_num)] = source_content.strip().replace("\n", "") # 将该节点的行号和内容添加到 new_line_nodes 中
                for tl in to_line_nums:
                    new_line_edges.append((int(line_num), int(tl))) # 新的边（起点行号，终点行号）
                for fl in from_line_nums:
                    new_line_edges.append((int(fl), int(line_num)))

    new_line_edges = list(set(new_line_edges))
    return new_line_nodes, new_line_edges

# 在 PDG 图中找到与当前节点集合 (nodes) 直接相连的所有邻居节点，并更新节点和边集合
# nodes = [node1, node2, ...]，当前节点（行号）集合
# edges = [[node1, node2], [node2, node3], ...]，当前边集合
def find_neighbor(nodes, edges, new_line_edges):
    neighbors = []
    neighbors.extend(nodes)
    for node in neighbors: # 遍历当前节点集合
        for edge in new_line_edges: # 遍历 PDG 函数图的所有的边
            # 当前节点是边的起始节点，且该边尚未存在于现有的边集合中
            if edge[0] == node and not any(e == [node, edge[1]] for e in edges): 
                nodes.append(edge[1]) # 保存邻居
                edges.append([node, edge[1]]) # 保存边
            # 当前节点是边的终点节点，且该边尚未存在于现有的边集合中
            if edge[1] == node and not any(e == [edge[0], node] for e in edges):
                nodes.append(edge[0])
                edges.append([edge[0], node])
    nodes = list(set(nodes)) # 去重
    return nodes, edges

# 根据指定行号 (line_num) 和跳数 (hop) 提取一个子图。默认上下文距离为 2
def get_sub_graph(new_line_nodes, new_line_edges, line_num, ast_nodes, ast_start_nodes, ast_dest_nodes, source_code, hop=2):
    pdg_node_list = []
    pdg_edge_list = []
    pdg_node_list.append(line_num)

    # 扩展子图
    for i in range(hop):
        pdg_node_list, pdg_edge_list = find_neighbor(pdg_node_list, pdg_edge_list, new_line_edges)

    sub_graph_edges = pdg_edge_list # 先存所有的 PDG 边
    sub_graph_nodes = []

    '''
        1、先将所有的 PDG 边和节点加入到子图中，并将 PDG 边的索引转换成新的子图节点索引
        2、将子图里的 PDG 行号转换成对应的 AST 根节点原始索引，并保存根节点对应的 AST
        -- 转换行号后，避免了行号与原始索引的混淆
        3、将每行 AST 加入到子图中，并构建对应的边
    '''

    # 先将所有的 PDG 边和节点加入到子图中，并将 PDG 边的索引转换成新的子图节点索引
    # 这里的 edge[i] 是 行号
    for i, edge in enumerate(sub_graph_edges):
        if edge[0] not in sub_graph_nodes:
            sub_graph_nodes.append(edge[0])
        sub_graph_edges[i][0] = sub_graph_nodes.index(edge[0])
        if edge[1] not in sub_graph_nodes:
            sub_graph_nodes.append(edge[1])
        sub_graph_edges[i][1] = sub_graph_nodes.index(edge[1])

    all_line_ast_edges = {} # 存储根节点对应的 AST，"根节点": [(start, dest), ...] 

    # 将子图里的 PDG 行号转换成对应的 AST 根节点原始索引，并保存根节点对应的 AST
    for i, node_line in enumerate(sub_graph_nodes):
        _, line_ast_edges, ast_root_node = get_line_ast(node_line, ast_nodes, ast_start_nodes, ast_dest_nodes, source_code)
        if not ast_root_node: # 如果没有根节点，跳过
            continue
        ast_root_idx = list(ast_root_node.keys())[0] # 得到根节点的原始索引
        sub_graph_nodes[i] = ast_root_idx # 行号转成对应的 AST 根原始索引
        all_line_ast_edges[ast_root_idx] = line_ast_edges # 预存根节点对应的 AST

    # 此时的 sub_graph_nodes 是 AST 根节点的原始索引，避免了 MGVD 代码里 AST 原始索引和 PDG 行号的混淆
    # 将每行 AST 加入到子图中，并构建对应的边
    # 这里的 edge[i] 是 AST 原始索引
    for i, root_idx in enumerate(sub_graph_nodes):
        # 将根节点的 AST 加入到子图中
        if root_idx in all_line_ast_edges:
            line_ast_edges = all_line_ast_edges[root_idx]
            for edge in line_ast_edges:
                # 将子树的每个节点加入到子图中
                if edge[0] not in sub_graph_nodes:
                    sub_graph_nodes.append(edge[0])
                if edge[1] not in sub_graph_nodes:
                    sub_graph_nodes.append(edge[1])
                # 构建对应的边
                sub_graph_edges.append([sub_graph_nodes.index(edge[0]), sub_graph_nodes.index(edge[1])])

    # 将 AST 原始索引转换成对应的 AST node 内容 { "line_num": x, "content": y }
    for i, ast_node_idx in enumerate(sub_graph_nodes):
        sub_graph_nodes[i] = ast_nodes[ast_node_idx]

    return sub_graph_nodes, sub_graph_edges

# 返回 max_length 个 padding 后的图
def padding_graph(line_embeddings_list, max_length):
    if len(line_embeddings_list) == max_length:
        return line_embeddings_list
    if len(line_embeddings_list) < max_length:
        blank_nodes_content = [[0 for _ in range(100)]] # 内容填 100 个 0
        blank_nodes_edge = []
        blank_line_dict = {"node_features": blank_nodes_content, "edges": blank_nodes_edge}
        num_added_line = max_length - len(line_embeddings_list)
        for i in range(num_added_line):
            line_embeddings_list.append(blank_line_dict)
        return line_embeddings_list
    else:
        return line_embeddings_list[:max_length]

# 获取 fold 下 code_type 的所有 mid_graph 的图 embedding，每个函数的图 embedding 保存在一个 jsonl 文件中
def get_fun_graph(fold, code_type):
    problem_files = []

    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(constant.represent_dir + "/"+code_type+"_model.bin")

    for root, _, files in os.walk(fold):
        files_num = len(files)
        count = 0
        print("total files: ", files_num)
        
        # 按文件名 id.后缀 的 id 排序
        files = sorted(files, key=lambda x: int(x.split(".")[0]))

        # 遍历所有函数，得到每个函数的图 embedding
        for fname in files:
            count += 1
            print_progress(count, files_num)

            if not fname.endswith(".txt"):
                continue
            
            id = fname.split(".")[0]

            file_path = os.path.join(root, fname)
            
            # 提取函数图信息
            label, source_code, adj_ast, adj_cfg, \
            adj_pdg, ast_nodes, cfg_nodes, pdg_nodes, \
            pdg_start_nodes, pdg_dest_nodes, ast_start_nodes, \
            ast_dest_nodes, problem = extract_graph_info(file_path)
            
            if problem:
                problem_files.append(file_path)
                continue

            new_line_nodes, new_line_edges = get_func_pdg(pdg_nodes, pdg_start_nodes, pdg_dest_nodes, source_code)
            
            new_line_nodes = sorted(new_line_nodes) # 按行号排序
            line_embeddings_list = [] # 所有 line 的 embedding

            # 遍历每行，得到每行的图 embedding
            for _, line_num in enumerate(new_line_nodes):
                # 获取子图
                sub_graph_nodes, sub_graph_edges = get_sub_graph(new_line_nodes, new_line_edges, line_num, ast_nodes, ast_start_nodes, ast_dest_nodes, source_code, hop = 2)

                node_features = []
                # 遍历子图的节点，通过 sent2vec 将每个节点 content 转成 embedding
                for node in sub_graph_nodes:
                    content_tokenized = tokenizer.tokenize(node["content"])
                    # to know
                    vector = sent2vec_model.embed_sentence(" ".join(content_tokenized))[0] # [0] 返回句子列表中的第一个句子的 embedding
                    node_features.append(vector.tolist()) # 将 numpy.ndarray 转成 list

                line_graph = {"node_features": node_features, "edges": sub_graph_edges} # 得到该行的图 embedding
                line_embeddings_list.append(line_graph)
            
            # padding 并追加保存当前函数的图集表示到 jsonl 文件
            if len(line_embeddings_list) > 0:
                line_embeddings_list = padding_graph(line_embeddings_list, 128)
                save_path = constant.functions_graph_dir + "/" + code_type + "_function_embedding.jsonl"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with jsonlines.open(save_path, mode='a') as writer:
                    writer.write({"function_id": id, "label": label, f"{code_type}_graph": line_embeddings_list})
    
    # 保存问题文件
    print("\n{} problem_files: ".format(code_type))
    print(len(problem_files))
    problem_file_path = constant.functions_graph_dir + "/problem/" + code_type + "_problem_files.txt"
    os.makedirs(os.path.dirname(problem_file_path), exist_ok=True) 
    with open(problem_file_path, 'w', encoding='utf-8') as pf:
        pf.writelines(problem_files)

# def main(code_type): # TODO {code_type}_graph 包含三个信息，并 update padding_graph
#     fold = Path(constant.normalized_dir) / f"{code_type}_mid_graphs"
#     print("Start to get {} function graph".format(code_type))
#     os.makedirs(fold, exist_ok=True)
#     get_fun_graph(fold, code_type)

# if __name__ == '__main__':
#     # main("normalized")
#     main("raw")