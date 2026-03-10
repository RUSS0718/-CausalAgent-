#网页上传文件
import streamlit as st
import time
from knowledge_base import KnowledgeBaseService
#streamlit机制:当web页面元素变化，则代码重新执行一遍
st.title("知识库更新服务")  #绝对路径streamlit run app_file_uploader.py
#file_uploader
upload_file = st.file_uploader(
    "请上传txt文件",
    type=['txt'],
    accept_multiple_files=False
)
#session_state 一个不会随着页面刷新而执行的字典
if"service"not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if upload_file is not None:
    #提取文件信息
    file_name = upload_file.name
    file_type = upload_file.type
    file_size = upload_file.size

    st.subheader(f"文件名:{file_name}")
    st.write(f"格式:{file_type}|大小:{file_size}")

    text = upload_file.getvalue().decode('utf-8')
    # st.write(text)  

    with st.spinner("载入知识库中..."):
        time.sleep(1)
        result  = st.session_state["service"].upload_by_str(text,file_name)
        st.write(result)  

# print(f'上传了{st.session_state["counter"]}个文件')