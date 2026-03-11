from rag import RagService
import streamlit as st
import config_data4rag as config

st.title("智能客服")
st.divider()

with st.sidebar:
    st.markdown("### Tool / MCP 使用说明")
    st.caption("趋势拟合: 在消息中输入如 `(1,450),(2,470),(3,485)` 会自动触发 calculate 工具")
    st.caption("MCP调用: 输入 `/mcp <tool_name> {json参数}`，例如 `/mcp ping {\"name\":\"cil\"}`")
    st.caption(f"MCP enabled: {config.mcp_enabled}, transport: {config.mcp_transport}")

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    ai_res_list = []
    with st.spinner("AI思考中..."):
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})
