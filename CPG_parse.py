import os
import pandas as pd
import common.constant as constant
from pathlib import Path

normalized_codes_path = Path(constant.normalized_dir) / "normalized_code_files"
raw_codes_path = Path(constant.normalized_dir) / "raw_code_files"

def print_progress(idx, total):
    print("\r", end="")
    print("Process progress: {}%: ".format(idx / total * 100), end="")

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
        print_progress(idx, df.shape[0])

        file_name = f"{idx}.c"
        with open(Path(code_path) / file_name, "w") as f:
            if code_type == "normalized":
                f.write(row.normalized_code)
            elif code_type == "raw":
                f.write(row.raw_code)

def main():
    DF_to_code_file("normalized")
    DF_to_code_file("raw")

if __name__ == "__main__":
    main()