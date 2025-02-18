import os
import common.constant as constant
import json
import shutil

test_set_dir = constant.vul_rag_test_set

id = 0

for root, dirs, files in os.walk(test_set_dir):
    for file in files:
        if not file.endswith(".json"):
            continue
        json_path = os.path.join(root, file)
        if file.endswith("_testset.json"):
            continue
        CWE_ID = file.split("_")[0]
        name = CWE_ID + "_testset.json"
        save_path = os.path.join(constant.vul_rag_test_set, name)
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for key, data_list in data.items():
            for item in data_list:
                item['id'] = id
                id += 1
                del item['detect_result']
                del item['final_result']
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
        os.remove(json_path)