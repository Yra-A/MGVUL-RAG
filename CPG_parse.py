import os
import pandas as pd
import common.constant as constant
from pathlib import Path
import subprocess
from common.tool.progress import print_progress

normalized_codes_path = Path(constant.normalized_dir) / "normalized_code_files"
raw_codes_path = Path(constant.normalized_dir) / "raw_code_files"


# 将 normalized 下的 bigvul_normalized.csv 中的 normalized_code 和 raw_code 写入文件夹 normalized_code_files 和 raw_code_files 中
def DF_to_code_file(code_type: str):
    if code_type == "normalized":
        code_path = normalized_codes_path
    elif code_type == "raw":
        code_path = raw_codes_path

    os.makedirs(code_path, exist_ok=True)

    df = pd.read_csv(Path(constant.normalized_dir) / "bigvul_normalized.csv")
    
    for idx, row in df.iterrows():
        # 打印进度
        print_progress(row['id'], df.shape[0])

        file_name = f"{row['id']}.c"
        with open(Path(code_path) / file_name, "w") as f:
            if code_type == "normalized":
                f.write(row.normalized_code)
            elif code_type == "raw":
                f.write(row.raw_code)

# 将 normalized_code 和 raw_code 转换为 CPG 保存在 normalized_CPGs 和 raw_CPGs 中
def joern_parse(code_type: str): 
    codes_path = normalized_codes_path
    if code_type == "raw":
        codes_path = raw_codes_path

    out_dir = Path(constant.normalized_dir) / f"{code_type}_CPGs"
    os.makedirs(out_dir, exist_ok=True)

    cur_progress = 0

    print("开始转换 {}_codes 到 CPGs".format(code_type))

    for file in os.listdir(codes_path):
        # 打印进度
        cur_progress += 1
        print_progress(cur_progress, len(os.listdir(codes_path)))

        name, _ = os.path.splitext(file) # 得到 id
        out_dir = Path(constant.normalized_dir) / f"{code_type}_CPGs" # CPG 的文件夹
        os.makedirs(out_dir, exist_ok=True)

        joern_path = constant.joern_path
        os.chdir(joern_path)
        os.environ['input'] = str(codes_path / file)
        os.environ['output'] = str(out_dir / f"{name}_cpg.bin")

        process = subprocess.Popen('sh joern-parse $input --out $output',    
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        shell=True, close_fds=True)
        output = process.communicate() 

        


def main():
    # 将 normalized code 和 raw code 写入文件中
    # DF_to_code_file("normalized")
    DF_to_code_file("raw")
    # joern_parse("normalized")
    # joern_parse("raw")

if __name__ == "__main__":
    main()