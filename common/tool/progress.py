def print_progress(idx, total, print_step=None):
    if total <= 0:
        raise ValueError("参数 total 必须大于 0。")
    
    # 根据总数自动确定打印步长（每 1% 打印一次）
    if print_step is None:
        print_step = max(total // 100, 1)
    
    # 如果当前任务编号 idx 刚好是步长的倍数，或者已经处理完成，则打印进度
    if idx % print_step == 0 or idx == total:
        progress = idx / total * 100
        print("Process progress: {:.2f}%".format(progress))
