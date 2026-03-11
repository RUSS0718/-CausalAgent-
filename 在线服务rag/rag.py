# rag_service.py

from vector_store import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data4rag as config
from file_history_store import get_history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory

from tools import maybe_calculate_from_text


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(
                model=config.embedding_model_name,
                dashscope_api_key=config.dashscope_api_key,
            )
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是智能助手cil,是一名进行数据分析的智能体助手。"
                    "以我提供的已知参考资料为主，简洁和专业的回答用户问题。"
                    "参考资料:{context}。"
                    "如果存在工具计算结果，请优先结合工具计算结果回答：{tool_result}。"
                    "请你每次回答用户问题用'你好我是您的数据分析助手cil'为开头，"
                    "你的输出格式请严格使用JSON",
                ),
                ("system", "并且我提供当前用户与你的历史记录"),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问: {input}"),
            ]
        )

        self.chat_model = ChatTongyi(
            model=config.chat_model_name,
            dashscope_api_key=config.dashscope_api_key,
        )
        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for doc in docs:
                formatted_str += f"文档片段：{doc.page_content}\n文档元数据:{doc.metadata}\n\n"
            return formatted_str

        def dict2str(value: dict) -> str:
            return value["input"]

        def maybe_tool(value: dict) -> str:
            return maybe_calculate_from_text(value["input"])

        def new_dict(value):
            return {
                "input": value["input"]["input"],
                "context": value["context"],
                "history": value["input"]["history"],
                "tool_result": value["tool_result"] or "无",
            }

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(dict2str) | retriever | format_document,
                "tool_result": RunnableLambda(maybe_tool),
            }
            | RunnableLambda(new_dict)
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain


if __name__ == "__main__":
    res = RagService().chain.invoke(
        {"input": "我有成绩点(1,450),(2,470),(3,485),(4,500)，下次是多少"},
        config=config.session_config,
    )
    print(res)
