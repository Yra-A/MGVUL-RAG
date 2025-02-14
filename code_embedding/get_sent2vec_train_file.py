import os
from pathlib import Path
import common.constant as constant
from common.tool.progress import print_progress
from code_embedding.qwenCoder_tokenize import tokenizer

def to_train_file(code_type):
    input_path = Path(constant.normalized_dir) / f"{code_type}_code_files"

    # 构造输出文件路径
    output_file = Path(constant.sent2vec_train_file) / f"{code_type}_train.txt"
    os.makedirs(output_file.parent, exist_ok=True)

    problem_list = []
    
    with open(output_file, 'a', encoding='utf-8') as train_file:
        # 遍历输入路径下的所有文件
        total = len(os.listdir(input_path))
        count = 0
        for file in input_path.iterdir():
            # 打印进度
            count += 1
            print_progress(count, total)
            if file.name.endswith(".c"):
                # 打开每个文件并按行读取内容
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            # 去除行首尾的空白字符
                            line = line.strip()
                            # 如果行非空，则写入输出文件
                            if line:
                                line_tokenized = tokenizer.tokenize(line)
                                train_file.write(" ".join(line_tokenized) + "\n")
                except Exception as e:
                    problem_list.append(file)


    for problem in problem_list:
        print(f"Problem with: {problem}")

to_train_file("normalized")
# to_train_file("raw")