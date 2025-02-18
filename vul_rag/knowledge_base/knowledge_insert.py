import os
import pandas as pd
import common.constant as constant
from pathlib import Path
import ast
import torch
import re
import pickle
from pymilvus import MilvusClient

def mean2tensor(data_str):
    values_pattern = r"tensor\((\[.*?\])\)"
    values_match = re.search(values_pattern, data_str, re.DOTALL)
    if not values_match:
        raise ValueError("无法解析字符串中的 tensor 数据")

    values_list = ast.literal_eval(values_match.group(1))  # 转换为列表
    tensor = torch.tensor(values_list)
    return tensor

def max2tensor(data_str):
    values_pattern = r"values=tensor\((\[.*?\])\)"
    values_match = re.search(values_pattern, data_str, re.DOTALL)
    if not values_match:
        raise ValueError("无法解析字符串中的 tensor 数据")
    
    values_list = ast.literal_eval(values_match.group(1))
    tensor = torch.tensor(values_list)
    return tensor

def extract_data_to_pickle():
    for CWE_ID in constant.CWE_ID_ENUM:
        df = pd.read_csv(Path(constant.vulnerability_knowledge_with_vectors_dir) / f'temp_CWE-{CWE_ID}_with_vectors.csv')

        data_list = df.to_dict(orient='records')
        for data in data_list:
            for key, value in data.items():
                if value is None or str(value) == 'nan':
                    data[key] = None
                    continue
                if key.endswith('mean') or key == 'sequence_vec':
                    data[key] = mean2tensor(str(value))
                elif key.endswith('max'):
                    data[key] = max2tensor(str(value))

        # 保存到 pickle 文件
        with open(Path(constant.vulnerability_knowledge_with_vectors_dir) / f'CWE_{CWE_ID}_with_vectors.pkl', 'wb') as f:
            pickle.dump(data_list, f)

# 将数据插入到 Milvus 中
def insert_data_from_pickle(db_uri):
    client = MilvusClient(uri=db_uri)

    for CWE_ID in constant.CWE_ID_ENUM:
        with open(Path(constant.vulnerability_knowledge_with_vectors_dir) / f'CWE_{CWE_ID}_with_vectors.pkl', 'rb') as f:
            data_list = pickle.load(f)
        
        data = []
        for item in data_list:
            # 仅存 vul code
            if item['id'].endswith('after'):
                continue
            # id = xx_before
            item['id'] = int(item['id'].split('_')[0])
            for key, value in item.items():
                if key == 'id':
                    continue
                if value is not None:
                    item[key] = value.tolist()
                else: # joern 检查不出来的，用零向量填充
                    if key.find('128') != -1:
                        item[key] = [0] * 128
                    elif key.find('256') != -1:
                        item[key] = [0] * 256
            data.append(item)
        res = client.insert(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            data=data
        )
        print(constant.vul_rag_collection_name.format(CWE_ID=CWE_ID))
        print(res)

if __name__ == '__main__':
    # extract_data_to_pickle()
    insert_data_from_pickle(constant.vul_rag_db_uri)