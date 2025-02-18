from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema, Collection, connections
import common.constant as constant

client = MilvusClient(constant.vul_rag_db_uri)

# 创建 schema 和 collection

id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
sequence_vec = FieldSchema(name="sequence_vec", dtype=DataType.FLOAT_VECTOR, dim=700, description="sequence vector")
raw_vec128_mean = FieldSchema(name="raw_vec128_mean", dtype=DataType.FLOAT_VECTOR, dim=128, description="raw vector mean")
raw_vec256_mean = FieldSchema(name="raw_vec256_mean", dtype=DataType.FLOAT_VECTOR, dim=256, description="raw vector mean")
normalized_vec128_mean = FieldSchema(name="normalized_vec128_mean", dtype=DataType.FLOAT_VECTOR, dim=128, description="normalized vector mean")
normalized_vec256_mean = FieldSchema(name="normalized_vec256_mean", dtype=DataType.FLOAT_VECTOR, dim=256, description="normalized vector mean")
raw_vec128_max = FieldSchema(name="raw_vec128_max", dtype=DataType.FLOAT_VECTOR, dim=128, description="raw vector max")
raw_vec256_max = FieldSchema(name="raw_vec256_max", dtype=DataType.FLOAT_VECTOR, dim=256, description="raw vector max")
normalized_vec128_max = FieldSchema(name="normalized_vec128_max", dtype=DataType.FLOAT_VECTOR, dim=128, description="normalized vector max")
normalized_vec256_max = FieldSchema(name="normalized_vec256_max", dtype=DataType.FLOAT_VECTOR, dim=256, description="normalized vector max")

schema = CollectionSchema(fields=[id_field, sequence_vec, raw_vec128_mean, \
         raw_vec256_mean, normalized_vec128_mean, normalized_vec256_mean, \
         raw_vec128_max, raw_vec256_max, normalized_vec128_max, normalized_vec256_max], \
         auto_id=False, enable_dynamic_field=True, description="Vul Code Collection")

for CWE_ID in constant.CWE_ID_ENUM:
    if client.has_collection(collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID)):
        client.drop_collection(collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID))

    client.create_collection(
        collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID), 
        schema=schema, 
    )

# 创建索引
def construct_index():
    # 创建索引
    # sequence 索引
    index_params_sequence_vec = MilvusClient.prepare_index_params()
    index_params_sequence_vec.add_index(
        field_name="sequence_vec",
        index_name="sequence_vec_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_sequence_vec,
            sync=False
        )

    # raw_vec128_mean 索引
    index_params_raw_vec128_mean = MilvusClient.prepare_index_params()
    index_params_raw_vec128_mean.add_index(
        field_name="raw_vec128_mean",
        index_name="raw_vec128_mean_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_raw_vec128_mean,
            sync=False
        )

    # raw_vec256_mean 索引
    index_params_raw_vec256_mean = MilvusClient.prepare_index_params()
    index_params_raw_vec256_mean.add_index(
        field_name="raw_vec256_mean",
        index_name="raw_vec256_mean_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_raw_vec256_mean,
            sync=False
        )

    # normalized_vec128_mean 索引
    index_params_normalized_vec128_mean = MilvusClient.prepare_index_params()
    index_params_normalized_vec128_mean.add_index(
        field_name="normalized_vec128_mean",
        index_name="normalized_vec128_mean_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_normalized_vec128_mean,
            sync=False
        )

    # normalized_vec256_mean 索引
    index_params_normalized_vec256_mean = MilvusClient.prepare_index_params()
    index_params_normalized_vec256_mean.add_index(
        field_name="normalized_vec256_mean",
        index_name="normalized_vec256_mean_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_normalized_vec256_mean,
            sync=False
        )

    # raw_vec128_max 索引
    index_params_raw_vec128_max = MilvusClient.prepare_index_params()
    index_params_raw_vec128_max.add_index(
        field_name="raw_vec128_max",
        index_name="raw_vec128_max_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_raw_vec128_max,
            sync=False
        )

    # raw_vec256_max 索引
    index_params_raw_vec256_max = MilvusClient.prepare_index_params()
    index_params_raw_vec256_max.add_index(
        field_name="raw_vec256_max",
        index_name="raw_vec256_max_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_raw_vec256_max,
            sync=False
        )

    # normalized_vec128_max 索引
    index_params_normalized_vec128_max = MilvusClient.prepare_index_params()
    index_params_normalized_vec128_max.add_index(
        field_name="normalized_vec128_max",
        index_name="normalized_vec128_max_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_normalized_vec128_max,
            sync=False
        )

    # normalized_vec256_max 索引
    index_params_normalized_vec256_max = MilvusClient.prepare_index_params()
    index_params_normalized_vec256_max.add_index(
        field_name="normalized_vec256_max",
        index_name="normalized_vec256_max_index",
        index_type="FLAT",
        metric_type="COSINE"
    )
    for CWE_ID in constant.CWE_ID_ENUM:
        client.create_index(
            collection_name=constant.vul_rag_collection_name.format(CWE_ID=CWE_ID),
            index_params=index_params_normalized_vec256_max,
            sync=False
        )

construct_index()