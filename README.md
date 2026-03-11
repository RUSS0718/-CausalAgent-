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



## 三、工具调用层（新增）
在线服务已新增 `在线服务rag/tools.py` 工具层，提供了一个 `calculate` 工具（LangChain `@tool`）。

- 输入：`[(x, y), ...]` 或 JSON 字符串 / 文本中的 `(x,y)` 序列
- 计算：使用 `numpy.linalg.lstsq` 做最小二乘线性拟合（不依赖 scikit-learn）
- 输出：线性方程、预测的下一个 `x` 与对应 `y`（JSON 字符串）
- 额外产物：拟合图将保存到 `在线服务rag/artifacts/` 并返回 `plot_path`

`RagService` 在生成回答前会尝试从用户输入抽取 `(x,y)` 数据，若成功会调用 `calculate` 并把结果注入 Prompt，形成“RAG + 工具计算”的组合回答。


## 四、MCP（Model Context Protocol）补充
当前已补充一个可用的 MCP 客户端层（stdio 方式），用于在对话中按命令触发远程工具：

- 新增 `在线服务rag/mcp_client.py`：实现 `initialize / tools/list / tools/call`。
- 在聊天中输入：`/mcp <tool_name> {json参数}`，即可尝试调用 MCP Tool。
- `RagService` 会把 MCP 返回结果注入到提示词中的 `mcp_result` 字段。

### 环境变量配置示例
```bash
export DASHSCOPE_API_KEY="你的key"
export MCP_ENABLED=true
export MCP_TRANSPORT=stdio
export MCP_SERVER_CMD="python your_mcp_server.py"
export MCP_TIMEOUT=15
```

> 说明：目前 MCP 入口为命令触发式（`/mcp ...`），是“先可用”的实现；后续可继续升级为由模型自动选择 MCP 工具。
