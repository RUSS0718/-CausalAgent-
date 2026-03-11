import os


md5_path = os.getenv("MD5_PATH", "离线知识库服务/md5.text")

collection_name = os.getenv("RAG_COLLECTION", "rag")
persist_directory = os.getenv("RAG_PERSIST_DIR", "离线知识库服务/chroma_db")

chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
separator = ["\n\n", "\n", ".", "", ",", "，", "。", "!"]
max_splitter_num = int(os.getenv("RAG_MAX_SPLITTER_NUM", "1000"))

similarity_threshold = int(os.getenv("RAG_SIMILARITY_THRESHOLD", "2"))

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
chat_model_name = os.getenv("CHAT_MODEL_NAME", "qwen3-max")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")

# MCP 配置
mcp_enabled = os.getenv("MCP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
mcp_transport = os.getenv("MCP_TRANSPORT", "stdio")
mcp_server_cmd = os.getenv("MCP_SERVER_CMD", "")
mcp_timeout = float(os.getenv("MCP_TIMEOUT", "15"))

session_config = {
    "configurable": {
        "session_id": os.getenv("RAG_SESSION_ID", "user_001"),
    }
}
