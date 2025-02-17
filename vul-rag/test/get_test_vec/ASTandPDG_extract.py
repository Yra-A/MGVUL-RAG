import os
from pathlib import Path
import common.constant as constant
from common.tool.progress import print_progress
import subprocess
import shutil

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