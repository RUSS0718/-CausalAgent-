from langchain_chroma import Chroma
import config_data4rag as config

# 配置检索返回的文档数量（注释含义：检索返回匹配的文档数量）
similarity_threshold = 2


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的传入
        """
        self.embedding = embedding

        # 初始化持久化 Chroma 向量库
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入chain"""
        return self.vector_store.as_retriever(
            search_kwargs={"k": config.similarity_threshold}
        )


if __name__ == '__main__':
    from langchain_community.embeddings import DashScopeEmbeddings

    # 初始化向量检索器
    retriever = VectorStoreService(
        DashScopeEmbeddings(model="text-embedding-v4",dashscope_api_key="sk-1d356b7349e3405790812ec286c58483")
    ).get_retriever()

    # 执行检索
    res = retriever.invoke("EDA的目标")
    print(res)