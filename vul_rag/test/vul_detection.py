import os
from pathlib import Path
from tqdm import tqdm
import os
from common import config as cfg
from common.tool.path_util import PathUtil
from common.tool.data_util import DataUtils
from common import constant
import argparse
from vul_rag.test.MGVUL_RAG import MGVUL_RAG_Detector

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument( # 要检测的CWE列表
        '--cwe-list', 
        nargs = '*', 
        type = str, 
        help = 'The list of CWEs to detect.',
        default = []
    )

    parser.add_argument( # 用来vul_rag检测的模型名称
        "--model-name",
        type = str,
        required = True,
        help = "The name of the model used for VUL-RAG detection.",
    )

    parser.add_argument( # 是否从检查点恢复
        '--resume',
        action = 'store_true',
        help = 'Whether to resume from a checkpoint.'
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    '''
    测试数据集: storage/vul_rag/test/testset/CWE-{CWE_ID}_testset.json
    保存结果: storage/vul_rag/test/result/{model_name}_{prompt}.json
    包含检查点恢复机制
    从测试数据集中得到提前分类好的 vul 和 non-vul 数据，并对其进行检测，检测结果保存在每个 item 的 detect_result 中
    '''

    args = parse_command_line_arguments() # 解析命令行参数
    result_dir_list = [] # 结果目录列表

    for CWE_ID in args.cwe_list:
        print(f"Start detecting {CWE_ID}...")

        vul_knowledge_path = constant.vul_knowledge_file.format(
            CWE_ID = CWE_ID
        )
        VulD = MGVUL_RAG_Detector(
            model_name = args.model_name,
            vul_knowledge_path = vul_knowledge_path
        )

        result_file_name = f"CWE_{CWE_ID}_{args.model_name}" # 结果文件名
        checkpoint_path = PathUtil.checkpoint_data(result_file_name, "pkl") # 检查点路径
        output_path = PathUtil.vul_detection_output( # 最终结果文件 json
            filename=result_file_name,
            ext="json",
            detection_model_name=args.model_name,
        )
        print("result_file_name: ", result_file_name)
        print("output_path: ", output_path)
        print("checkpoint_path: ", checkpoint_path)

        result_dir_list.append(os.path.dirname(output_path)) # 结果目录列表

        test_data_path = Path(constant.vul_rag_test_data.format(CWE_ID = CWE_ID)) # 测试数据路径
        test_data = DataUtils.load_json(test_data_path) # 加载测试数据

        vul_list = test_data['vul_data'] # 待检测的漏洞数据
        non_vul_list = test_data['non_vul_data'] # 待检测的非漏洞数据
        ckpt_cve_list = []

        res_vul_list = [] # 检测结果的漏洞数据
        res_non_vul_list = [] # 检测结果的非漏洞数据
        
        if args.resume: # 从检查点恢复
            if os.path.exists(checkpoint_path): # 如果检查点文件存在
                ckpt_cve_list = list(DataUtils.load_data_from_pickle_file(checkpoint_path)) # 从检查点文件中加载数据
                if os.path.exists(output_path): # 如果输出文件存在
                    data = DataUtils.load_json(output_path) # 加载输出文件
                    res_vul_list = data['vul_data'] # 获取 vul_data 检测结果
                    res_non_vul_list = data['non_vul_data'] # 获取non_vul_data 检测结果
            else: 
                # to avoid overwriting the existing output file
                raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.") # 抛出断点文件未找到异常
        
        # 开始检测
        try:
            print(f"Start testing vulnerable samples")
            for item in tqdm(vul_list):
                if item['id'] in ckpt_cve_list:
                    continue
                
                vectors = constant.vul_rag_test_data_vector.format(CWE_ID = CWE_ID, id = item['id']) # 获取向量
                vul_detect_result = VulD.detection_pipeline(
                    id=item['id'],
                    cve_id=item['cve_id'],
                    cwe_id=CWE_ID,
                    code_snippet=item['code_snippet'],
                ) # 返回多了一个 detect_result 和 final_result 的 item

                res_vul_list.append(vul_detect_result) # 添加检测结果
                ckpt_cve_list.append(item['id'])
                DataUtils.save_json(output_path, {"vul_data": res_vul_list, "non_vul_data": res_non_vul_list})
                DataUtils.write_data_to_pickle_file(ckpt_cve_list, checkpoint_path) # 保存检查点

            print(f"Start testing non-vulnerable samples")

            for item in tqdm(non_vul_list):
                if item['id'] in ckpt_cve_list:
                    continue

                non_vul_detect_result = VulD.detection_pipeline(
                    id=item['id'],
                    cve_id=item['cve_id'],
                    cwe_id=CWE_ID,
                    code_snippet=item['code_snippet'],
                )

                res_non_vul_list.append(non_vul_detect_result) # 添加检测结果
                ckpt_cve_list.append(item['id'])
                DataUtils.save_json(output_path, {"vul_data": res_vul_list, "non_vul_data": res_non_vul_list})
                DataUtils.write_data_to_pickle_file(ckpt_cve_list, checkpoint_path) # 保存检查点

        except Exception as e:
            DataUtils.write_data_to_pickle_file(ckpt_cve_list, checkpoint_path) # 保存检查点
            print(f"ID: {item['id']}")
            print(f"CVE ID: {item['cve_id']}")
            print(f"Error: {e}")
            print(f"Detection for CWE_{CWE_ID} failed. Checkpoint saved.")

        if os.path.exists(checkpoint_path): 
            os.remove(checkpoint_path)
        print(f"Detection for CWE_{CWE_ID} finished.")
        DataUtils.save_json(output_path, {"vul_data": res_vul_list, "non_vul_data": res_non_vul_list})

    print(f"Detection for all CWEs finished.")