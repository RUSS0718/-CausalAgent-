# rag_service.py

from dashscope import History

from vector_store import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data4rag as config
from file_history_store import FileChatMessageHistory,get_history
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory




class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name,dashscope_api_key=config.dashscope_api_key)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", 
                 "你是智能助手cil,是一名进行数据分析的智能体助手"
                 "以我提供的已知参考资料为主，"
                           "简洁和专业的回答用户问题。参考资料:{context}。"
                "请你每次回答用户问题用'你好我是您的数据分析助手cil'为开头，"
                "你的输出格式请严格使用JSON"),
                ("system","并且我提供当前用户与你的历史记录"),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问: {input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name,dashscope_api_key=config.dashscope_api_key)
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
        def print_prompt(v):
            print(v.to_string())
            return v
        def dict2str(value:dict)->str:
            return value["input"]
        def temp1(value):
            print("="*20,value)
            return value
        def new_dict(value):
            new_v = {}
            new_v["input"] = value["input"]["input"]
            new_v["context"] = value["context"]
            new_v["history"] = value["input"]["history"]
            return new_v
        chain = (
            {
                "input": RunnablePassthrough(),
                "context": 
                # RunnableLambda(temp1)| ##:{'input': '我是一名小白,请你向我介绍如何入门', 'history': []},retriever只接收字符串
                RunnableLambda(dict2str) 
                |retriever | format_document
            }
            # |RunnableLambda(temp1)  input[history是空的]
            |RunnableLambda(new_dict)
            
            | self.prompt_template|print_prompt
            | self.chat_model
            | StrOutputParser()
        )
      

        conversation_chain = RunnableWithMessageHistory(
                    chain,
                    get_history,
                    input_messages_key="input",
                    history_messages_key="history",

        )
        return conversation_chain #增强链不能传入字符串 必须传入字典了

if __name__ == '__main__':
 

    res = RagService().chain.invoke({"input": "我已经入门了 ，可以告诉怎么才能变成数据分析高手吗"}, config=config.session_config)
    # res = RagService().chain.invoke("我是一名小白,请你向我介绍如何入门")
    # print(res)我是一名小白,请你向我介绍如何入