from common.tool.data_util import DataUtils
from common.tool.path_util import PathUtil

# ----------------------- config for OpenAI API -----------------------
deepseek_api_base = "https://api.deepseek.com/v1"
deepseek_api_key = DataUtils.load_data_from_pickle_file(PathUtil.api_keys_data("deepseek_api_key", "pkl"))

qwen_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
qwen_api_key = DataUtils.load_data_from_pickle_file(PathUtil.api_keys_data("qwen_api_key", "pkl"))

# 百炼平台除了 qwen 模型以外的调用
bailian_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
bailian_api_key = DataUtils.load_data_from_pickle_file(PathUtil.api_keys_data("bailian_api_key", "pkl"))

OPENAI_API_CONNECTION_PROXY = "http://127.0.0.1:58390"

DETECT_TOP_N = 3 # RAG 检索 TOP N 个结果

RERANK_WEIGHT = { # 检索重排权重
    'sequence': 0.3,
    'normalized': 0.3,
    'raw': 0.4
}

RESULT_UNIFORM_MAP = {
    1: 1,
    0: 0,
    -1: 0,
    "yes": 1,
    "no": 0,
    "Yes": 1,
    "No": 0,
    "1": 1,
    "0": 0,
    "-1": 0,
    "VUL_YES": 1,
    "VUL_NO": 0
}

METRICS_DECIMAL_PLACES_RESERVED = 4