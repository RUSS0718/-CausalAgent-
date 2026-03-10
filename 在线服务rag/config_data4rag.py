md5_path = "D:\项目\causal_agent项目考核----数据分析agent\离线知识库服务\md5.text"

collection_name = "rag"
# persist_directory="./chroma_db"
persist_directory="D:\项目\causal_agent项目考核----数据分析agent\离线知识库服务\chroma_db"

chunk_size=1000
chunk_overlap=100
separator=["\n\n","\n",".","",",","，","。","!"]
max_splitter_num = 1000 #文本分割阈值


similarity_threshold = 2  # 检索返回的文档数量


embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"
dashscope_api_key="sk-1d356b7349e3405790812ec286c58483"


   # session id 配置
session_config = {
    "configurable": {
        "session_id": "user_001",
    }
}