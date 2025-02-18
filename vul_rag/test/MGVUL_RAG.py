from common.tool.data_util import DataUtils
from common import config as cfg
from common.model_manager import ModelManager
from pymilvus import MilvusClient
from common import constant
import json
from common.prompt import VD_Prompt

class MGVUL_RAG_Detector:
    def __init__(
            self, 
            model_name: str,
            vul_knowledge_path: str,
        ):
        self.vul_knowledge = DataUtils.load_json(vul_knowledge_path) 
        self.model_instance = ModelManager.get_model_instance(model_name)
        self.client = MilvusClient(
            uri=constant.vul_rag_db_uri,
        )

    def rerank(self, sequence_res_list, raw_vec128_max_res_list, normalized_vec128_max_res_list, weight=cfg.RERANK_WEIGHT):
        sequence_weight = weight['sequence']
        raw_weight = weight['raw']
        normalized_weight = weight['normalized']

        combined_results = {}

        for hit in sequence_res_list:
            id = hit['id']
            score = hit['distance']
            if id not in combined_results:
                combined_results[id] = 0
            combined_results[id] += sequence_weight * score
        
        for hit in raw_vec128_max_res_list:
            id = hit['id']
            score = hit['distance']
            if id not in combined_results:
                combined_results[id] = 0
            combined_results[id] += raw_weight * score
        
        for hit in normalized_vec128_max_res_list:
            id = hit['id']
            score = hit['distance']
            if id not in combined_results:
                combined_results[id] = 0
            combined_results[id] += normalized_weight * score

        # 根据综合得分进行排序（降序）
        combined_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
        # 返回重新排序后的结果
        final_res = []
        for id, score in combined_results:
            final_res.append({
                "id": id,
                "score": score
            })
        return final_res

    # 从 Milvus 中检索相似代码，并对检索结果进行重排
    def retrieve_similar_code(self, id, cwe_id, top_N=cfg.DETECT_TOP_N):
        vector_path = constant.vul_rag_test_data_vector.format(CWE_ID = cwe_id, id = id)
        vector = DataUtils.load_json(vector_path)
        sequence_vec = vector['sequence_vec']
        raw_vec128_max = vector['raw_vec128_max']
        normalized_vec128_max = vector['normalized_vec128_max']
        
        sequence_res_list = []
        raw_vec128_max_res_list = []
        normalized_vec128_max_res_list = []

        res = self.client.search(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=cwe_id),
            anns_field="sequence_vec",
            data=[sequence_vec],
            limit=top_N,
            search_params={"metric_type": "COSINE"}
        )
        assert(len(res) > 0)
        for hits in res:
            for hit in hits:
                sequence_res_list.append(hit)
        
        if raw_vec128_max:
            res = self.client.search(
                collection_name=constant.vul_rag_collection_name.format(CWE_ID=cwe_id),
                anns_field="raw_vec128_max",
                data=[raw_vec128_max],
                limit=top_N,
                search_params={"metric_type": "COSINE"}
            )
            assert(len(res) > 0)
            for hits in res:
                for hit in hits:
                    raw_vec128_max_res_list.append(hit)

        if normalized_vec128_max:
            res = self.client.search(
                collection_name=constant.vul_rag_collection_name.format(CWE_ID=cwe_id),
                anns_field="normalized_vec128_max",
                data=[normalized_vec128_max],
                limit=top_N,
                search_params={"metric_type": "COSINE"}
            )
            assert(len(res) > 0)
            for hits in res:
                for hit in hits:
                    normalized_vec128_max_res_list.append(hit)

        return self.rerank(sequence_res_list, raw_vec128_max_res_list, normalized_vec128_max_res_list)[:top_N]

    def get_vul_knowledge(self, id):
        assert(len(self.vul_knowledge.items()) > 0)
        for CVE_ID, CVE_LIST in self.vul_knowledge.items():
            for item in CVE_LIST:
                if item['id'] == id:
                    return item['vulnerability_behavior']

    def detection_pipeline(self, id, cve_id, cwe_id, code_snippet):
        detect_result = []
        similar_code_list = self.retrieve_similar_code(id, cwe_id)
        print(f"similar_code_list: {similar_code_list}")
        count = 0
        assert(len(similar_code_list) > 0)
        for similar_code in similar_code_list:
            count += 1
            print(f"{count} / {len(similar_code_list)}")

            id = similar_code['id']
            # 获取漏洞知识
            vul_knowledge = self.get_vul_knowledge(id)
            print(vul_knowledge)
            
            # 确保 vul_knowledge 为 dict，且有三个 key，分别为 "preconditions_for_vulnerability"， trigger_condition， specific_code_behavior_causing_vulnerability
            assert(isinstance(vul_knowledge, dict) and 
                "preconditions_for_vulnerability" in vul_knowledge and 
                "trigger_condition" in vul_knowledge and 
                "specific_code_behavior_causing_vulnerability" in vul_knowledge)
            
            # 生成 prompt
            vul_detect_prompt = VD_Prompt.generate_detect_vul_prompt(
                code_snippet=code_snippet,
                vulnerability_knowledge=vul_knowledge
            )

            # 调用 LLM
            vul_messages = self.model_instance.get_messages( # 得到 message，用来发给 model
                user_prompt=vul_detect_prompt, 
                sys_prompt=VD_Prompt.default_sys_prompt
            ) 
            vul_output = self.model_instance.get_response_with_messages( # 获取 model 响应
                vul_messages
            )
            print("vul_output: \n", vul_output)
            assert(vul_output is not None)

            # 保存检测结果
            result = {
                "vul_knowledge": vul_knowledge,
                "vul_detect_prompt": vul_detect_prompt,
                "vul_output": vul_output,
            }
            detect_result.append(result)
            
            # 有漏洞
            if constant.vul_positive in vul_output:
                return {
                    "id": id,
                    "cve_id": cve_id,
                    "code_snippet": code_snippet, 
                    "detect_result": detect_result, 
                    "detection_model": self.model_instance.get_model_name(),
                    "final_result": 1
                }
            else:
                continue
        
        # 无漏洞
        return {
            "id": id,
            "cve_id": cve_id, 
            "code_snippet": code_snippet, 
            "detect_result": detect_result, 
            "detection_model": self.model_instance.get_model_name(),
            "final_result": 0
        }


