import os
import pandas as pd
import common.constant as constant
from pathlib import Path
import subprocess
from common.tool.progress import print_progress
import json

normalized_codes_path = Path(constant.vul_rag_normalized) / "normalized_code_files"
raw_codes_path = Path(constant.vul_rag_normalized)  / "raw_code_files"


def DF_to_code_file(code_type: str):
    codes_path = normalized_codes_path
    if code_type == "raw":
        codes_path = raw_codes_path

    for file in os.listdir(constant.vul_rag_vul_knowledge_with_id_dir):
        if file.endswith(".json"):  
            CWE_ID = file.split("_")[1]
            with open(Path(constant.vul_rag_normalized) / file, 'r', encoding='utf-8') as file:
                data = json.load(file)  # 将 JSON 文件内容解析为 Python 对象
            assert(isinstance(data, dict))

            codes_path_CWE = codes_path / CWE_ID
            os.makedirs(codes_path_CWE, exist_ok=True)
            os.makedirs(codes_path_CWE, exist_ok=True)

            for CVE_ID, CVE_LIST in data.items():  
                for item in CVE_LIST:
                    before_file_name = f"{item['id']}_before.c"
                    after_file_name = f"{item['id']}_after.c"
                    with open(Path(codes_path_CWE) / before_file_name, "w") as f:
                        if code_type == "normalized":
                            f.write(item["code_before_change_normalized"])
                        elif code_type == "raw":
                            f.write(item["code_before_change_raw"])
                    with open(Path(codes_path_CWE) / after_file_name, "w") as f:
                        if code_type == "normalized":
                            f.write(item["code_after_change_normalized"])
                        elif code_type == "raw":
                            f.write(item["code_after_change_raw"])

# 将 normalized_code 和 raw_code 转换为 CPG 保存在 normalized_CPGs 和 raw_CPGs 中
def joern_parse(code_type: str): 
    print("开始转换 {}_codes 到 CPGs".format(code_type))
    codes_path = normalized_codes_path
    if code_type == "raw":
        codes_path = raw_codes_path
    for dir in os.listdir(codes_path):
        print("开始转换文件夹: {}".format(dir))
        dir_path = codes_path / dir
        if os.path.isdir(dir_path):
            count = 0
            for file in os.listdir(dir_path):
                if not file.endswith(".c"):
                    continue
                count += 1
                print("{} 进度: {}/{}".format(dir, count, len(os.listdir(dir_path))))
                
                name = file.split(".")[0]
                if os.path.exists(Path(constant.vul_rag_normalized) / f"{code_type}_CPGs" / dir / f"{name}_cpg.bin"):
                    continue

                out_dir = constant.vul_rag_normalized + f"/{code_type}_CPGs" + f"/{dir}"
                os.makedirs(out_dir, exist_ok=True)
                joern_path = constant.joern_path
                os.chdir(joern_path)
                
                name, _ = os.path.splitext(file)
                output = str(Path(out_dir) / f"{name}_cpg.bin")

                input = str(dir_path / file)
                os.environ['input'] = input
                os.environ['output'] = output

                print("    input: {}; output: {}".format(input, output))

                process = subprocess.Popen('sh joern-parse $input --out $output',    
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                shell=True, close_fds=True)
                output = process.communicate() 
        


def main():
    # 将 normalized code 和 raw code 写入文件中
    # DF_to_code_file("normalized")
    # DF_to_code_file("raw")
    joern_parse("normalized")
    joern_parse("raw")

if __name__ == "__main__":
    main()