import common.constant as constant
import pandas as pd
import os
import re
from pathlib import Path

# def remove_comments(text):
#     """Delete comments from code."""

#     def replacer(match):
#         s = match.group(0)
#         if s.startswith("/"):
#             return " "  # note: a space and not an empty string
#         else:
#             return s

#     pattern = re.compile(
#         r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
#         re.DOTALL | re.MULTILINE,
#     )
#     return re.sub(pattern, replacer, text)

def process_bigvul(pre_num=0):
    '''
    处理 bigvul 数据集
    pre_num: 加载预处理后的前 pre_num 个样本
    '''
    csv_file_path = constant.bigvul_origin
    
    df = pd.read_csv(
        csv_file_path,
        parse_dates=["Publish Date", "Update Date"],
        dtype={
            "commit_id": str,
            "del_lines": int,
            "file_name": str,
            "lang": str,
            "lines_after": str,
            "lines_before": str,
            "Unnamed: 0": int,
            "Access Gained": str,
            "Attack Origin": str,
            "Authentication Required": str,
            "Availability": str,
            "CVE ID": str,
            "CVE Page": str,
            "CWE ID": str,
            "Complexity": str,
            "Confidentiality": str,
            "Integrity": str,
            "Known Exploits": str,
            "Score": float,
            "Summary": str,
            "Vulnerability Classification": str,
            "add_lines": int,
            "codeLink": str,
            "commit_message": str,
            "files_changed": str,
            "func_after": str,
            "func_before": str,
            "parentID": str,
            "patch": str,
            "project": str,
            "project_after": str,
            "project_before": str,
            "vul": int,
            "vul_func_with_fix": str,
        },
        nrows=10,
    )
    df = df.rename(columns={"Unnamed: 0": "id", "func_before": "vulnerable_code", "func_after": "patched_code"})
    df["dataset"] = "bigvul"
    df["is_extended"] = False # 是否是扩展数据集

    dfv = df[df.vul == 1] # 保留有漏洞的样本

    # 删除异常结尾的代码样本
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]

    # 过滤掉被预处理掉的 vul 样本
    keep_vuln = set(dfv["id"].tolist()) # 将被预处理过的样本 id 转成 set
    df = df[(df.vul == 0) | (df["id"].isin(keep_vuln))].copy() # 保留 no-vul 样本和被预处理过的 vul 样本

    if pre_num > 0:
        df = df.head(pre_num)


    os.makedirs(constant.preprocessed_dir, exist_ok=True)

    df[
        [
            "id",
            "CVE ID",
            "CWE ID",
            "Vulnerability Classification",
            "patched_code",
            "vulnerable_code",
            "vul",
            "is_extended",
        ]
    ].to_csv(Path(constant.preprocessed_dir) / "bigvul_processed_metadata.csv", index = False)
    
# main 函数
if __name__ == "__main__":
    process_bigvul(10)