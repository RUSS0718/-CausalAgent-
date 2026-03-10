md5_path = "./md5.text"

collection_name = "rag"
persist_directory="./chroma_db"
chunk_size=1000
chunk_overlap=100
separator=["\n\n","\n",".","",",","，","。","!"]
max_splitter_num = 1000 #文本分割阈值


similarity_threshold = 2  # 检索返回的文档数量