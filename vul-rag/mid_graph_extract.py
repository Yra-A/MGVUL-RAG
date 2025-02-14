import os
import re
import pandas as pd
import common.constant as constant
from pathlib import Path
from common.tool.progress import print_progress

# 定义一个函数，用于分割节点信息为节点编号和节点内容
# e.g. input：nodeList = '[(18,1), (16,b+1), (15,malloc(b+1)), (14,char*)'
# e.g. output：new_node_list = ['(18,1)', '(16,b+1)', '(15,malloc(b+1))', '(14,char*)']
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
def joernGraph(code_type):
    input_dir = f"/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/vul-rag/normalized/{code_type}_graphs"
    output_dir = f"/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/vul-rag/normalized/{code_type}_mid_graphs"

    for dir in os.listdir(input_dir):  
        if not dir.startswith("CWE-"):
            continue
        print("开始处理文件夹: {}".format(dir))
        total = len(os.listdir(input_dir + '/' + dir))
        count = 0
        for file in os.listdir(input_dir + '/' + dir):
            if not file.endswith(".txt"):
                continue
            count += 1
            print("进度: {} / {}".format(count, total))

            # 初始化存储 AST、CFG 和 PDG 边的列表
            ast_node_relation = []  # 存储 AST 的边
            cfg_node_relation = []  # 存储 CFG 的边
            pdg_node_relation = []  # 存储 PDG 的边

            path = input_dir + '/' + dir + '/' + file
            try:
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

                        code_path = Path(constant.vul_rag_normalized) / f"{code_type}_code_files" / f"{dir}" / f"{filename}.c"
                        with open(code_path, 'r', encoding='utf-8') as f1:
                            code = f1.read()

                        output_path = output_dir + '/' + dir
                        # 创建输出目录
                        if os.path.exists(output_path) == False:
                            os.makedirs(output_path)

                        # 写入文件
                        with open(output_path + "/" + filename + "_mid_graph.txt", 'w', encoding='utf-8') as f2:
                            
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
            except Exception as e:
                print(path)
                print(e)
                print(filename)

if __name__ == '__main__':
    joernGraph("normalized")  # 处理 normalized 数据
    joernGraph("raw")  # 处理 raw 数据
    
    print("finish!!!!!!!!!!!!")