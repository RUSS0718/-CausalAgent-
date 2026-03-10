import os, json
import config_data4rag as config

from langchain_core.messages import message_to_dict, messages_from_dict,BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import Sequence, List

# from langchain_core.chat_history import InMemoryChatMessageHistory

# message_to_dict：单个消息对象（BaseMessage类实例） -> 字典
# messages_from_dict：[字典、字典...]  -> [消息、消息...]
# AIMessage、HumanMessage、SystemMessage 都是BaseMessage的子类


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id          # 会话id
        self.storage_path = storage_path      # 不同会话id的存储文件，所在的文件夹路径
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)

        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # Sequence序列 类似list、tuple
        all_messages = list(self.messages)        # 已有的消息列表
        all_messages.extend(messages)             # 新的和已有的融合成一个list

        # 将数据同步写入到本地文件中
        # 类对象写入文件 -> 一堆二进制
        # 为了方便，可以将BaseMessage消息转为字典（借助json模块以json字符串写入文件）
        # 官方message_to_dict：单个消息对象（BaseMessage类实例） -> 字典
 

        new_messages = [message_to_dict(message) for message in all_messages]
        # 将数据写入文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f)

    @property
    # @property装饰器将messages方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        # 当前文件内：list[字典]
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)      # 返回值就是：list[字典]
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)




model =  ChatTongyi(model=config.chat_model_name,dashscope_api_key=config.dashscope_api_key)

prompt = PromptTemplate.from_template("你需要根据会话历史回应用户问题。对话历史:{chat_history},用户提问:{input}，请回答")

str_parser = StrOutputParser()

def prompt_printer(prompt):
    print("="*20,prompt,"="*20)

    return prompt
base_chain = prompt | prompt_printer |model | str_parser
his_dic = {}
def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")



conversation_chain = RunnableWithMessageHistory(
base_chain,
get_history,
input_messages_key="input",
history_messages_key="chat_history"

)