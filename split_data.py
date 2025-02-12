import os
import numpy as np
import jsonlines
import gzip
import random

def Split(train, valid, test, data_fold, save_path):
    """
    将数据集分割为训练集、验证集和测试集，并对训练集进行过采样。
    
    参数:
        train (float): 训练集所占比例
        valid (float): 验证集所占比例
        test (float): 测试集所占比例
        data_fold (str): 数据文件所在的目录路径
        save_path (str): 保存分割后数据的目录路径
    """
    print("\nbegin to split files...")
    tem_item = []
    emb_path = data_fold + "/function_embedded.jsonl"
    path_tem = data_fold + "/tem.jsonl"
    
    clean_tem = []
    vul_tem = []

    # 读取 embedding
    with open(emb_path, "r", encoding="utf8") as f:
        reader = jsonlines.Reader(f)
        total_lines = sum(1 for _ in reader)  # 统计总行数，用于显示进度
        f.seek(0)  # 回到文件开头重新读取
        count = 0
        for item in jsonlines.Reader(f):
            tem_item.append(item)  # 将每一行数据加载到 tem_item 中
            print("\r处理进度: {:.2f}%".format(count / total_lines * 100), end="")
            count += 1
    print("\n数据加载完成！")

    # 分类数据：根据标签将数据分为干净样本和漏洞样本
    for item in tem_item:
        if item["label"] == "0":
            clean_tem.append(item)
        else:
            vul_tem.append(item)
    
    clean_length = len(clean_tem)
    vul_length = len(vul_tem)

    # 训练集
    vul_tem_train = vul_tem[:int(vul_length * train)]
    clean_tem_train = clean_tem[:int(clean_length * train)]
    
    # 过采样：使漏洞样本数量与干净样本数量平衡，这里采用简单的复制样本的方法
    for _ in range(int(clean_length * train) - int(vul_length * train)):
        random_index = random.randrange(len(vul_tem_train)) # 随机选择一个漏洞样本
        vul_tem_train.append(vul_tem_train[random_index]) # 复制该样本
    
    tem_train = vul_tem_train + clean_tem_train
    np.random.shuffle(tem_train) # 打乱顺序


    # 验证集
    tem_valid = (
        vul_tem[int(vul_length * train):int(vul_length * (train + valid))] +
        clean_tem[int(clean_length * train):int(clean_length * (train + valid))]
    )
    np.random.shuffle(tem_valid)  # 打乱顺序

    # 测试集
    tem_test = (
        vul_tem[int(vul_length * (train + valid)):] +
        clean_tem[int(clean_length * (train + valid)):]
    )
    np.random.shuffle(tem_test)  # 打乱顺序

    # 保存分割后的数据
    def save_data(data, file_name):
        """辅助函数：将数据保存为压缩的 .jsonl.gz 文件"""
        with jsonlines.open(path_tem, mode='w') as writer:  # 写入临时文件
            for item in data:
                writer.write(item)
        
        # 压缩临时文件
        with open(path_tem, 'rb') as f_in, gzip.open(os.path.join(save_path, file_name), 'wb') as f_out:
            f_out.writelines(f_in)
        
        os.remove(path_tem)  # 删除临时文件
        print(f"{file_name.split('.')[0]} 数据已保存！")

    # 保存训练集、验证集和测试集
    save_data(tem_train, "train.jsonl.gz")
    save_data(tem_valid, "valid.jsonl.gz")
    save_data(tem_test, "test.jsonl.gz")

    print("所有数据分割完成！")

if __name__ == "__main__":
    json_path = 
    save_path = 
    Split(train=0.8, valid=0.1, test=0.1, data_fold=json_path, save_path=save_path)