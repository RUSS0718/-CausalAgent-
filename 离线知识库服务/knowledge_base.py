from hashlib import md5
import os
import hashlib
from matplotlib.pylab import f
from numpy import save
import config_data as config
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
def check_md5(md5_str:str): #检查传入的文件 是否已经存入过
    if not os.path.exists(config.md5_path):
        #md5文件不存在
        open(config.md5_path,"w",encoding='utf-8').close()
        return False  #表示md5未被处理过
    else:
        for  line in open(config.md5_path,'r',encoding='utf-8').readlines():
            line = line.strip() #处理字符串之间的空格和回车
            if line == md5_str:
                return True #已经处理过
        return  False




def save_md5(md5_str:str):
    """""将传入的md5字符串记录到文件内保存"""
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5_str+'\n')


def get_string_md5(input_str:str , encoding='utf-8'):
    """将传入的字符串转换为md5字符串"""
    str_bytes = input_str.encode(encoding=encoding) #字符串转换为bytes字节数组
    md5_obj = hashlib.md5() #得到MD5对象
    md5_obj.update(str_bytes) #更新内容 传入最开始的字符串
    md5_hex = md5_obj.hexdigest() #得到md5的十六进制字符串
    return md5_hex

class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory,exist_ok=True)
        self.chroma=Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(
                model="text-embedding-v4",dashscope_api_key=config.dashscope_api_key
                )
            ,persist_directory=config.persist_directory
        ) #向量库实例
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,
            chunk_overlap = config.chunk_overlap,
            separators=config.separator,
            length_function = len,
        )
        

    def upload_by_str(self,data:str,filename):
        #先得到md5字符串
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "[跳过]内容已经存在于知识库中"
        if len(data)>config.max_splitter_num:
            knowledge_chunks:list[str] = self.spliter.split_text(data) #分割字符串
        else:
            knowledge_chunks=[data]
        metadata = {
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"才桥西"        
        }
        self.chroma.add_texts( #内容加载到向量库中
#iterable -> list / tuple
        knowledge_chunks,
        metadatas = [metadata for _ in knowledge_chunks]
        )  
        save_md5(md5_hex)

        return "[成功]内容已经载入向量库"



if __name__ == '__main__':
    # r = get_string_md5("才桥西")
    # r1 = get_string_md5("才桥西")
    # r2 = get_string_md5("才桥西1")
    # print(r)
    # print(r1)
    # print(r2)
    # b7035b58e78ef8809163f5fd5185282f
# b7035b58e78ef8809163f5fd5185282f
# ad042d8bec3d012019b150df037b9b7b
#字符串有一点不一样 md5就完全不一样 不管原字符串多大 md5都会生成等长度的字符串 节省空间
#    save_md5("b7035b58e78ef8809163f5fd5185282f")
# print(check_md5("b7035b58e78ef8809163f5fd5185282f"))
    service = KnowledgeBaseService()
    r = service.upload_by_str("cqx","testfile")
    print(r)