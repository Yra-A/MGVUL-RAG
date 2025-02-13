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

    work_dir = Path(constant.normalized_dir) / f"{code_type}_CPGs"
    problem_file = Path(constant.problem_log_path) / f"{code_type}_problem_graph_info_extract.txt"

    print("开始转换 {}_CPGs 到 graphs".format(code_type))

    total = len(os.listdir(work_dir))


    for i in range(total):
        input_path = Path(work_dir) / f"{i}_cpg.bin"
        if not input_path.exists():
            with open(problem_file, "a") as f:
                f.write(f"{i}_cpg.bin not exists\n")
            continue
        os.chdir(joern_path)
        print("current input file:", input_path)

        out_dir = Path(constant.normalized_dir) / f"{code_type}_graphs" / f"temp_{i}"
        os.makedirs(out_dir, exist_ok=True)
        
        # 设置参数
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
                    print("current file:", file)
                    shutil.move(Path(out_dir) / file, Path(constant.normalized_dir) / f"{code_type}_graphs" / f"{i}_graph_info.txt")
        shutil.rmtree(out_dir)
    # try:
    #     shutil.rmtree(work_dir)
    # except Exception as e:
    #     print(e)
    #     print("remove error")
    #     return
            
def main():
    extract_graph_from_CPG("normalized")
    # extract_graph_from_CPG("raw")

if __name__ == "__main__":
    main()