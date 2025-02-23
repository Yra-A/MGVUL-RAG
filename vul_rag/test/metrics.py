from common.constant import MetricsKeywords as mk
import os
import logging
from common.tool.data_util import DataUtils
from common import config as cfg
import json
from pathlib import Path

# Author: VUL_RAG
# https://github.com/KnowledgeRAG4LLMVulD/KnowledgeRAG4LLMVulD

def calculate_metrics(**confusion_matrix) -> dict:
    """
    Calculate various performance metrics based on the provided confusion matrix.
    Args:
        **confusion_matrix: A dictionary containing the confusion matrix values.
            Expected keys are:
            - 'True Negative': True Negative count
            - 'True Positive': True Positive count
            - 'False Negative': False Negative count
            - 'False Positive': False Positive count
            - 'id_result_map' (optional): A list of tuples, where each tuple contains two dictionaries.
                Each dictionary should have the following keys:
                - 'id': Identifier for the item
                - 'Prediction': Predicted value
                - 'Ground Truth': Ground truth value
    Returns:
        dict: A dictionary containing the calculated metrics:
            - 'Precision': Average Precision
            - 'Recall': Average Recall
            - 'F1 Score': F1 Score
            - 'Accuracy': Accuracy
            - 'False Negative Rate': False Negative Rate
            - 'False Positive Rate': False Positive Rate
            - 'True Negative Rate': True Negative Rate
            - 'True Positive Rate': True Positive Rate
            - 'Valid Pair Count' (if 'id_result_map' is provided): Valid Pair Count
            - 'Accurate Pair Count' (if 'id_result_map' is provided): Accurate Pair Count
            - 'Pair Accuracy' (if 'id_result_map' is provided): Pair Accuracy
            - 'Pair_1 Rate' (if 'id_result_map' is provided): Pair_1 Rate
            - 'Pair_0 Rate' (if 'id_result_map' is provided): Pair_0 Rate
    Raises:
        ValueError: If the confusion matrix does not contain TN, TP, FN, and FP.
    """
    TN = confusion_matrix.get(mk.TN.value)
    TP = confusion_matrix.get(mk.TP.value)
    FN = confusion_matrix.get(mk.FN.value)
    FP = confusion_matrix.get(mk.FP.value)

    if TN is None or TP is None or FN is None or FP is None:
        logging.error("The confusion matrix must contain TN, TP, FN, and FP.")
        raise ValueError("The confusion matrix must contain TN, TP, FN, and FP.")

    pos_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    pos_recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    neg_precision = TN / (TN + FN) if (TN + FN) > 0 else 0
    neg_recall = TN / (TN + FP) if (TN + FP) > 0 else 0


    precision = (pos_precision + neg_precision) / 2
    recall = (pos_recall + neg_recall) / 2

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    FN_rate = FN / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else -1
    FP_rate = FP / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else -1
    TN_rate = TN / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else -1
    TP_rate = TP / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else -1

    result_map = {
        mk.PC.value: round(precision, cfg.METRICS_DECIMAL_PLACES_RESERVED),
        mk.RC.value: round(recall, cfg.METRICS_DECIMAL_PLACES_RESERVED),
        mk.F1.value: round(f1, cfg.METRICS_DECIMAL_PLACES_RESERVED),
        mk.AC.value: round(accuracy, cfg.METRICS_DECIMAL_PLACES_RESERVED),
        mk.FNR.value: round(FN_rate, cfg.METRICS_DECIMAL_PLACES_RESERVED),
        mk.FPR.value: round(FP_rate, cfg.METRICS_DECIMAL_PLACES_RESERVED),
        mk.TNR.value: round(TN_rate, cfg.METRICS_DECIMAL_PLACES_RESERVED),
        mk.TPR.value: round(TP_rate, cfg.METRICS_DECIMAL_PLACES_RESERVED)
    }

    id_result_map = confusion_matrix.get('id_result_map')
    if id_result_map:
        valid_pair_cnt = 0
        accurate_pair_cnt = 0
        pair_1_cnt = 0
        pair_0_cnt = 0
        for item_id, item_pair in id_result_map.items():
            assert len(item_pair) == 2, f"Invalid item pair for ID {item_id}."
            assert item_pair[0]['id'] == item_pair[1]['id'], f"ID mismatch for ID {item_id}."
            assert mk.PD.value in item_pair[0] and mk.PD.value in item_pair[1], f"PD value not found for ID {item_id}."
            assert mk.GT.value in item_pair[0] and mk.GT.value in item_pair[1], f"GT value not found for ID {item_id}."
            if item_pair[0][mk.PD.value] == item_pair[0][mk.GT.value] and \
                item_pair[1][mk.PD.value] == item_pair[1][mk.GT.value]:
                accurate_pair_cnt += 1
            if item_pair[0][mk.PD.value] == 1 and item_pair[1][mk.PD.value] == 1:
                pair_1_cnt += 1
            if item_pair[0][mk.PD.value] == 0 and item_pair[1][mk.PD.value] == 0:
                pair_0_cnt += 1
            valid_pair_cnt += 1

        result_map[mk.VPC.value] = valid_pair_cnt
        result_map[mk.APC.value] = accurate_pair_cnt
        result_map[mk.P1C.value] = pair_1_cnt
        result_map[mk.P0C.value] = pair_0_cnt
        result_map[mk.PAC.value] = accurate_pair_cnt / valid_pair_cnt if valid_pair_cnt > 0 else -1
        result_map[mk.P1R.value] = pair_1_cnt / valid_pair_cnt if valid_pair_cnt > 0 else -1
        result_map[mk.P0R.value] = pair_0_cnt / valid_pair_cnt if valid_pair_cnt > 0 else -1
        result_map[mk.PAC.value] = round(result_map[mk.PAC.value], cfg.METRICS_DECIMAL_PLACES_RESERVED)
        result_map[mk.P1R.value] = round(result_map[mk.P1R.value], cfg.METRICS_DECIMAL_PLACES_RESERVED)
        result_map[mk.P0R.value] = round(result_map[mk.P0R.value], cfg.METRICS_DECIMAL_PLACES_RESERVED)

    return result_map

def calculate_VD_metrics(result_file_or_dir: str, save_to_file: bool = True):
    target_result_file_list = []
    calculate_total_metrics_flag = False
    if os.path.exists(result_file_or_dir) and os.path.isdir(result_file_or_dir):
        target_result_file_list = [
            os.path.join(result_file_or_dir, file_name) for 
            file_name in os.listdir(result_file_or_dir) 
            if file_name.endswith(".json") and not file_name.endswith("_metrics.json")
        ]
        calculate_total_metrics_flag = True
    elif os.path.exists(result_file_or_dir) and os.path.isfile(result_file_or_dir):
        target_result_file_list.append(result_file_or_dir)
    else:
        raise ValueError("The result file or directory does not exist.")
    
    total_cfs_mat = {
        mk.TN.value: 0,
        mk.TP.value: 0,
        mk.FN.value: 0,
        mk.FP.value: 0
    }
    total_valid_pair_cnt = 0
    total_accurate_pair_cnt = 0
    total_pair_1_cnt = 0
    total_pair_0_cnt = 0

    for result_file in target_result_file_list:
        results = DataUtils.load_json(result_file)
        cfs_mat = {
            mk.TN.value: 0,
            mk.TP.value: 0,
            mk.FN.value: 0,
            mk.FP.value: 0
        }
        # the key in the result file is ether xx_data or xx_detect_data
        vul_data = results.get('vul_data', [])
        non_vul_data = results.get('non_vul_data', [])
        id_result_map = {}

        for vul in vul_data:
            try:
                if cfg.RESULT_UNIFORM_MAP[vul['final_result']] == 0:
                    cfs_mat[mk.FN.value] += 1
                else:
                    cfs_mat[mk.TP.value] += 1

            except Exception as e:
                logging.error(f"Error: {e}")
                logging.error(f"Cannot find the final_result for the ID {vul['id']} in the file {result_file_or_dir}.")

            id_result_map[vul['id']] = [{
                **vul,
                mk.PD.value: cfg.RESULT_UNIFORM_MAP[vul['final_result']],
                mk.GT.value: 1
            }]

        for non_vul in non_vul_data:
            try:
                if cfg.RESULT_UNIFORM_MAP[non_vul['final_result']] == 0:
                    cfs_mat[mk.TN.value] += 1
                else:
                    cfs_mat[mk.FP.value] += 1
                        
            except Exception as e:
                logging.error(f"Error: {e}")
                logging.error(f"Cannot find the final_result for the ID {vul['id']} in the file {result_file_or_dir}.")

            id_result_map[non_vul['id']].append({
                **non_vul,
                mk.PD.value: cfg.RESULT_UNIFORM_MAP[non_vul['final_result']],
                mk.GT.value: 0
            })

        metrics_data = calculate_metrics(**cfs_mat, id_result_map = id_result_map)
        
        logging.info(f"Result File: {result_file}")
        logging.info(f"{mk.TP.value}: {cfs_mat.get(mk.TP.value)}")
        logging.info(f"{mk.TN.value}: {cfs_mat.get(mk.TN.value)}")
        logging.info(f"{mk.FP.value}: {cfs_mat.get(mk.FP.value)}")
        logging.info(f"{mk.FN.value}: {cfs_mat.get(mk.FN.value)}")
        logging.info(f"{mk.FNR.value}: {metrics_data.get(mk.FNR.value)}")
        logging.info(f"{mk.FPR.value}: {metrics_data.get(mk.FPR.value)}")
        logging.info(f"{mk.PC.value}: {metrics_data.get(mk.PC.value)}")
        logging.info(f"{mk.RC.value}: {metrics_data.get(mk.RC.value)}")
        logging.info(f"{mk.F1.value}: {metrics_data.get(mk.F1.value)}")
        logging.info(f"{mk.AC.value}: {metrics_data.get(mk.AC.value)}")
        logging.info(f"{mk.VPC.value}: {metrics_data.get(mk.VPC.value)}")
        logging.info(f"{mk.APC.value}: {metrics_data.get(mk.APC.value)}")
        logging.info(f"{mk.PAC.value}: {metrics_data.get(mk.PAC.value)}")
        logging.info(f"{mk.P1R.value}: {metrics_data.get(mk.P1R.value)}")
        logging.info(f"{mk.P0R.value}: {metrics_data.get(mk.P0R.value)}")
        logging.info(f"{mk.P1C.value}: {metrics_data.get(mk.P1C.value)}")
        logging.info(f"{mk.P0C.value}: {metrics_data.get(mk.P0C.value)}")
        logging.info(f"--------------------------------------------------")

        total_cfs_mat[mk.TN.value] += cfs_mat.get(mk.TN.value)
        total_cfs_mat[mk.TP.value] += cfs_mat.get(mk.TP.value)
        total_cfs_mat[mk.FN.value] += cfs_mat.get(mk.FN.value)
        total_cfs_mat[mk.FP.value] += cfs_mat.get(mk.FP.value)

        total_valid_pair_cnt += metrics_data.get(mk.VPC.value)
        total_accurate_pair_cnt += metrics_data.get(mk.APC.value)
        total_pair_1_cnt += metrics_data.get(mk.P1C.value)
        total_pair_0_cnt += metrics_data.get(mk.P0C.value)

        if save_to_file:
            # try:
            #     check_result_file_legality(result_file)
            # except Exception as e:
            #     logging.warning(f"The result file {result_file} is incomplete. Error: {e}")

            metrics_data = {
                **cfs_mat,
                **metrics_data
            }
            result_file_name = os.path.basename(result_file)
            result_file_name = result_file_name.replace(".json", "_metrics.json")
            result_file_name = os.path.join(os.path.dirname(result_file), result_file_name)
            DataUtils.save_json(result_file_name, metrics_data)

    if calculate_total_metrics_flag:
        total_metrics_data = calculate_metrics(**total_cfs_mat)
        total_pair_accuracy = total_accurate_pair_cnt / total_valid_pair_cnt if total_valid_pair_cnt > 0 else -1
        total_pair_accuracy = round(total_pair_accuracy, cfg.METRICS_DECIMAL_PLACES_RESERVED)
        total_metrics_data[mk.VPC.value] = total_valid_pair_cnt
        total_metrics_data[mk.APC.value] = total_accurate_pair_cnt
        total_metrics_data[mk.PAC.value] = total_pair_accuracy
        total_metrics_data[mk.P1R.value] = total_pair_1_cnt / total_valid_pair_cnt if total_valid_pair_cnt > 0 else -1
        total_metrics_data[mk.P0R.value] = total_pair_0_cnt / total_valid_pair_cnt if total_valid_pair_cnt > 0 else -1
        total_metrics_data[mk.P1R.value] = round(total_metrics_data[mk.P1R.value], cfg.METRICS_DECIMAL_PLACES_RESERVED)
        total_metrics_data[mk.P0R.value] = round(total_metrics_data[mk.P0R.value], cfg.METRICS_DECIMAL_PLACES_RESERVED)
        logging.info(f"Total Metrics:")
        logging.info(f"{mk.TP.value}: {total_cfs_mat.get(mk.TP.value)}")
        logging.info(f"{mk.TN.value}: {total_cfs_mat.get(mk.TN.value)}")
        logging.info(f"{mk.FP.value}: {total_cfs_mat.get(mk.FP.value)}")
        logging.info(f"{mk.FN.value}: {total_cfs_mat.get(mk.FN.value)}")
        logging.info(f"{mk.FNR.value}: {total_metrics_data.get(mk.FNR.value)}")
        logging.info(f"{mk.FPR.value}: {total_metrics_data.get(mk.FPR.value)}")
        logging.info(f"{mk.PC.value}: {total_metrics_data.get(mk.PC.value)}")
        logging.info(f"{mk.RC.value}: {total_metrics_data.get(mk.RC.value)}")
        logging.info(f"{mk.F1.value}: {total_metrics_data.get(mk.F1.value)}")
        logging.info(f"{mk.AC.value}: {total_metrics_data.get(mk.AC.value)}")
        logging.info(f"{mk.VPC.value}: {total_valid_pair_cnt}")
        logging.info(f"{mk.APC.value}: {total_accurate_pair_cnt}")
        logging.info(f"{mk.PAC.value}: {total_pair_accuracy}")
        logging.info(f"{mk.P1R.value}: {total_metrics_data.get(mk.P1R.value)}")
        logging.info(f"{mk.P0R.value}: {total_metrics_data.get(mk.P0R.value)}")
        logging.info(f"--------------------------------------------------")

        if save_to_file:
            total_metrics_data = {
                **total_cfs_mat,
                **total_metrics_data
            }
            first_level_dir_name = os.path.basename(result_file_or_dir)
            second_level_dir_name = os.path.basename(os.path.dirname(result_file_or_dir))
            total_result_file_name = os.path.join(
                result_file_or_dir,
                f"{second_level_dir_name}_{first_level_dir_name}_all_CWE_metrics.json"
            )
            DataUtils.save_json(total_result_file_name, total_metrics_data)

# 将 non_vul_data 的 id 与 vul_data 的 id 对应起来，构成 pair
def make_pair_id():
    save_path = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/vul_rag/test/result/qwen_max_latest'

    for root, dirs, files in os.walk(save_path):
        for file in files:
            if file.endswith('.json'):
                with open(Path(root) / file, 'r') as f:
                    data = json.load(f)
                
                vul_data = data['vul_data']
                non_vul_data = data['non_vul_data']
                start_id = vul_data[0]['id']
                assert(len(vul_data) == len(non_vul_data))
                for i in range(len(vul_data)):
                    non_vul_data[i]['id'] = start_id
                    start_id += 1
                
                with open(Path(root) / file, 'w') as f:
                    json.dump(data, f, indent=2)

if __name__ == "__main__":
    calculate_VD_metrics('/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/vul_rag/test/result/qwen_max_latest', save_to_file = True)