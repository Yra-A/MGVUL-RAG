import os
import pandas as pd
import common.constant as constant
from pathlib import Path
import subprocess
from common.tool.progress import print_progress
import multiprocessing
import os

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
def joern_parse(process_idx, code_type: str, file_list): 
    print("待生成的 CPGs 个数: ", len(file_list))

    cur_progress = 0

    for file in file_list:
        name, _ = os.path.splitext(file) # 得到 id
        output = str(out_dir / f"{name}_cpg.bin")

        # 打印进度
        cur_progress += 1
        print(f"进程 {process_idx} 当前进度: {cur_progress}/{len(file_list)}", end='')

        os.environ['input'] = str(codes_path / file)
        os.environ['output'] = output

        process = subprocess.Popen('sh joern-parse $input --out $output',    
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        shell=True, close_fds=True)
        output = process.communicate() 

# 定义一个函数，作为新进程的入口
def worker_function(name, file_list, codes_path, out_dir):
    process_idx = name
    print(f"子进程 {process_idx} (PID: {os.getpid()}) 开始执行")
    print("待生成的 CPGs 个数: ", len(file_list))
    cur_progress = 0
    total = len(file_list)

    for file in file_list:
        name, _ = os.path.splitext(file) # 得到 id
        output = str(out_dir / f"{name}_cpg.bin")

        # 打印进度
        cur_progress += 1
        print(f"进程 {process_idx} 当前进度: {cur_progress}/{total}")
        joern_path = constant.joern_path
        os.chdir(joern_path)
        os.environ['input'] = str(codes_path / file)
        os.environ['output'] = output

        process = subprocess.Popen('sh joern-parse $input --out $output',    
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        shell=True, close_fds=True)
        output = process.communicate() 



    print(f"子进程 {name} 结束")

if __name__ == "__main__":
    print(f"主进程 (PID: {os.getpid()}) 开始执行")
    code_type = "normalized"
    codes_path = Path(constant.normalized_dir) / f"{code_type}_code_files"
    
    file_list = os.listdir(codes_path) 
    
    out_dir = Path(constant.normalized_dir) / f"{code_type}_CPGs"
    processed_list = os.listdir(out_dir)

    for file in processed_list:
        name = file.split("_")[0] + ".c"
        if name in file_list:
            file_list.remove(name)


    process_num = 3
    print("剩余未处理的文件个数: ", len(file_list))
    print("开始处理剩余文件, 子进程数: {}".format(process_num))
    
    # 将剩余未处理的文件分成 process_num 份
    every_len = len(file_list) // process_num
    new_file_list = []
    for i in range(process_num):
        if i == process_num - 1:
            new_file_list.append(file_list[i * every_len:])
        else:
            new_file_list.append(file_list[i * every_len: (i + 1) * every_len])

    # 创建多个进程
    processes = []
    for i in range(process_num):
        p = multiprocessing.Process(target=worker_function, args=(f"{i}", new_file_list[i], codes_path, out_dir))
        processes.append(p)
        p.start()  # 启动进程

    # 等待所有子进程结束
    for p in processes:
        p.join()

    print("所有子进程执行完毕")       