# 📚 RAG 智能问答系统项目说明

本项目实现了 **基于 RAG（Retrieval-Augmented Generation） 的 AI 智能问答系统**，整体由两部分组成：

- **离线知识库更新服务**
- **在线 AI 问答服务**

---

# 一、离线知识库更新服务
1.使用streamlit库实现前端页面 让用户能轻松载入知识库 
2.采用chroma轻量级向量库存储知识库
3.在载入新的知识库的过程中,本次我做的程序能够通过比较新的知识库与md5.text中的字段来选择是否载入新知识库(离线知识库服务/knowledge_base.py/upload_by_str) 
---

# 二、在线 AI 问答服务
1.使用streamlit库实现前端页面 用户能够轻松与cil交互
2.通过rag,让AI再回答问题前,会先检索知识库内容，进行相似度匹配 
3.每次输出都是JSON格式
4.根据session_id的不同，cil会记住不同用户与cil前面对话的内容,调用file_history_store中的方法添加历史消息列表,返回历史消息列表,以及删除所有历史记录 


春节后电脑坏了一段时间,后面快过完年才拿去修,学习节奏没有跟上,开学才完成本次任务。做的十分匆忙,代码估计有的地方注释不清。我没有实现工具调用功能,还没有来得及学已经要截至了,没有做出tool模块。十分抱歉

本次用的conda环境请看requirement.txt
可以用
pip install -r requirements.txt 
命令下载

