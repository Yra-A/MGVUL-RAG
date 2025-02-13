import os
from pathlib import Path
import common.constant as constant
from common.tool.progress import print_progress
import subprocess
import shutil
import multiprocessing

# 调用 joern 将 CPG 转换为 graph，保存在 normalized_graphs 和 raw_graphs 中
def extract_graph_from_CPG(name, work_list, code_type: str):
    process_idx = name

    joern_path = constant.joern_path
    script_file = constant.all_script

    work_dir = Path(constant.normalized_dir) / f"{code_type}_CPGs"
    problem_file = Path(constant.problem_log_path) / f"{code_type}_problem_graph_info_extract.txt"
    total = len(work_list)

    cur_progress = 0

    for i in work_list:
        if os.path.exists(Path(constant.normalized_dir) / f"{code_type}_graphs" / f"{i}_graph_info.txt"):
            continue
        cur_progress += 1
        print(f"进程 {process_idx} 当前进度: {cur_progress}/{total}")

        input_path = Path(work_dir) / f"{i}_cpg.bin"
        if not input_path.exists():
            with open(problem_file, "a") as f:
                f.write(f"{i}_cpg.bin not exists\n")
            continue
        os.chdir(joern_path)
        # print("current input file:", input_path)

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

        # 筛选
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                with open(Path(root) / file, 'r', encoding='utf-8') as f:  # 打开文件
                    lines = str(f.read())  # 读取文件内容
                if lines.startswith("(Some(/Users"):
                    # print("current file:", file)
                    shutil.move(Path(out_dir) / file, Path(constant.normalized_dir) / f"{code_type}_graphs" / f"{i}_graph_info.txt")
        shutil.rmtree(out_dir)

    print(f"进程 {process_idx} 执行完毕")
    

if __name__ == "__main__":
    print(f"主进程 (PID: {os.getpid()}) 开始执行")
    code_type = "normalized"

    work_dir = Path(constant.normalized_dir) / f"{code_type}_CPGs"
    out_dir = Path(constant.normalized_dir) / f"{code_type}_graphs"

    total = len(os.listdir(work_dir))
    processed_list = []

    for i in range(total):
        if os.path.exists(Path(constant.normalized_dir) / f"{code_type}_graphs" / f"{i}_graph_info.txt"):
            continue
        processed_list.append(i)

    process_num = 3
    print("剩余未处理的文件个数: ", len(processed_list))
    print("开始处理剩余文件, 子进程数: {}".format(process_num))
    
    # 将剩余未处理的文件分成 process_num 份
    every_len = len(processed_list) // process_num
    new_file_list = []
    for i in range(process_num):
        if i == process_num - 1:
            new_file_list.append(processed_list[i * every_len:])
        else:
            new_file_list.append(processed_list[i * every_len: (i + 1) * every_len])

    # 创建多个进程
    processes = []
    for i in range(process_num):
        p = multiprocessing.Process(target=extract_graph_from_CPG, args=(f"{i}", new_file_list[i], code_type))
        processes.append(p)
        p.start()  # 启动进程

    # 等待所有子进程结束
    for p in processes:
        p.join()

    print("所有子进程执行完毕")       