import os
from pathlib import Path
import common.constant as constant
from common.tool.progress import print_progress
import subprocess
import shutil

# 调用 joern 将 CPG 转换为 graph，保存在 normalized_graphs 和 raw_graphs 中
def extract_graph_from_CPG(code_type: str):
    joern_path = constant.joern_path
    script_file = constant.all_script

    work_dir = Path(constant.vul_rag_normalized) / f"{code_type}_CPGs"
    problem_file = Path(constant.problem_log_path) / f"vul_rag_{code_type}_problem_graph_info_extract.txt"

    print("开始转换 {}_CPGs 到 graphs".format(code_type))

    for dir in os.listdir(work_dir):
        print("开始转换文件夹: {}".format(dir))
        dir_path = work_dir / dir
        if os.path.isdir(dir_path):
            count = 0
            for file in os.listdir(dir_path):
                if not file.endswith(".bin"):
                    continue
                name = file.split("_cpg")[0]

                # 如果已经提取过，跳过
                if os.path.exists(Path(constant.vul_rag_normalized) / f"{code_type}_graphs" / dir / f"{name}_graph_info.txt"):
                    continue
                
                count += 1
                print("{} 进度: {}/{}".format(dir, count, len(os.listdir(dir_path))))

                # 设置输入参数
                input_path = Path(dir_path) / file
                if not input_path.exists():
                    with open(problem_file, "a") as f:
                        f.write(f"{name}_cpg.bin not exists\n")
                    continue
                os.chdir(joern_path)
                print("current input file:", input_path)

                # 设置输出参数，创建临时输出文件夹，提取要的主函数
                out_dir = Path(constant.vul_rag_normalized) / f"{code_type}_graphs" / dir / f"temp_{name}"
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
                for root, dirs, files in os.walk(out_dir):
                    for file in files:
                        with open(Path(root) / file, 'r', encoding='utf-8') as f:  # 打开文件
                            lines = str(f.read())  # 读取文件内容
                        if lines.startswith("(Some(/Users"):
                            shutil.move(Path(out_dir) / file, Path(constant.vul_rag_normalized) / f"{code_type}_graphs" / dir / f"{name}_graph_info.txt")
                shutil.rmtree(out_dir) # 删除临时文件夹
            
def main():
    extract_graph_from_CPG("normalized")
    extract_graph_from_CPG("raw")

if __name__ == "__main__":
    main()