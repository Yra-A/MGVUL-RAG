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
    out_dir = Path(constant.normalized_dir) / f"{code_type}_graphs"

    print("开始转换 {}_CPGs 到 graphs".format(code_type))
    
    if out_dir.exists():
        shutil.rmtree(out_dir)
    count = 0
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if not file.endswith("_cpg.bin"):
                continue
            count += 1
            print_progress(count, len(files))
            input_path = Path(root) / file
            os.chdir(joern_path)
            params = f"cpgFile={input_path},outDir={out_dir}"
            os.environ['params'] = str(params)
            os.environ['script_file'] = str(script_file)
            process = subprocess.Popen('sh joern --script $script_file --params $params',    
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       shell=True, close_fds=True)
            output = process.communicate()
            print(output)
            name = str(input_path).split('/')[-1].split('.')[0] # 得到文件名，先按 / 分割，取最后一个元素，再按 . 分割，取第一个元素，例如 0_cpg.bin 得到 0_cpg
    try:
        shutil.rmtree(work_dir)
    except Exception as e:
        print(e)
        print("remove error")
        return
            
def main():
    extract_graph_from_CPG("normalized")
    extract_graph_from_CPG("raw")

if __name__ == "__main__":
    main()