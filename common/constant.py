# bigvul 原始数据集路径
bigvul_origin = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/original/MSR_data_cleaned.csv'

# 预处理后的数据集文件夹
preprocessed_dir = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/preprocessed'

# normalized 文件夹
normalized_dir = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/normalized'

# 训练 sent2vec 语料的文件夹
sent2vec_train_file = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/sent2vec_train'

# sent2vec raw model
sent2vec_raw_model = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/code_embedding/raw_model.bin'

# sent2vec normalized model
sent2vec_normalized_model = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/code_embedding/normalized_model.bin'

# joern 路径
joern_path = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/joern-cli'

# joern 脚本路径
all_script = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/joern-cli/graph/all.sc'

# 存函数 embedding 文件夹
embeddings_dir = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/embeddings'

# embedding model 与 tool 文件夹
represent_dir = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/code_embedding'

# functions_graph 文件夹, 保存 normalized 和 raw 的所有函数图级表示
functions_graph_dir = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/functions_graph'

# 保存问题函数日志的路径
problem_log_path = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/logs/problem_fun.txt'

# vul_rag dir
vul_rag_dir = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/storage/vul_rag'

# vul_rag vunerability knowledge 数据
vul_rag_vul_knowledge_dir = vul_rag_dir + '/vulnerability_knowledge'

# vul_rag vunerability knowledge with id 数据
vul_rag_vul_knowledge_with_id_dir = vul_rag_dir + '/vulnerability_knowledge_with_id'

# vul_rag normalized dir
vul_rag_normalized = vul_rag_dir + '/normalized'

# vulnerability knowledge with vectors
vulnerability_knowledge_with_vectors_dir = vul_rag_dir + '/vulnerability_knowledge_with_vectors'

# CWE-ID ENUM
CWE_ID_ENUM = {
    '119', '362', '416', '476', '787'
}

# vul_rag test set
vul_rag_test_set = vul_rag_dir + "/test/testset"

# vul_rag db uri
vul_rag_db_uri = vul_rag_dir + "/test/vul_rag.db"

# vul_rag db collection
vul_rag_collection_name="vul_code_CWE_{CWE_ID}"

# common dir
common_dir = '/Users/yra/Yra/graduation_project/vulnerability_detection/MGVUL-RAG/common'

# vul_rag test result
vul_rag_test_result = vul_rag_dir + '/test/result'

# vul_rag test data(input)
vul_rag_test_data = vul_rag_test_set + '/CWE-{CWE_ID}_testset.json'
vul_rag_test_data_vector = vul_rag_test_set + '/vectors/CWE-{CWE_ID}/{id}/func_vectors.json'

# vul_rag vul knowledge with id file
vul_knowledge_file = vul_rag_vul_knowledge_with_id_dir + '/gpt-3.5-turbo_CWE-{CWE_ID}_316_with_id.json'

# vul positive
vul_positive = 'VUL_YES'
# vul negative
vul_negative = 'VUL_NO'