import json
import subprocess
import threading
import time
from typing import Any


class MCPClientError(RuntimeError):
    pass


class MCPClient:
    """轻量MCP stdio客户端，支持tools/list和tools/call。"""

    def __init__(self, server_cmd: str, timeout: float = 15.0):
        if not server_cmd:
            raise MCPClientError("MCP_SERVER_CMD 未配置")
        self.server_cmd = server_cmd
        self.timeout = timeout
        self._id = 0
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            server_cmd,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _send(self, payload: dict[str, Any]) -> None:
        if not self._proc.stdin:
            raise MCPClientError("MCP stdin 不可用")
        body = json.dumps(payload, ensure_ascii=False)
        msg = f"Content-Length: {len(body.encode('utf-8'))}\r\n\r\n{body}"
        self._proc.stdin.write(msg)
        self._proc.stdin.flush()

    def _read_response(self) -> dict[str, Any]:
        if not self._proc.stdout:
            raise MCPClientError("MCP stdout 不可用")

        start = time.time()
        content_len = 0
        while True:
            if time.time() - start > self.timeout:
                raise MCPClientError("读取MCP响应超时(头部)")
            line = self._proc.stdout.readline()
            if not line:
                raise MCPClientError("MCP连接已断开")
            line = line.strip()
            if not line:
                break
            if line.lower().startswith("content-length:"):
                content_len = int(line.split(":", 1)[1].strip())

        if content_len <= 0:
            raise MCPClientError("MCP响应缺少Content-Length")

        body = self._proc.stdout.read(content_len)
        if not body:
            raise MCPClientError("MCP响应体为空")
        return json.loads(body)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = self._next_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
        )
        res = self._read_response()
        if "error" in res:
            raise MCPClientError(f"MCP调用失败: {res['error']}")
        return res.get("result", {})

    def initialize(self) -> None:
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "causal-agent-rag", "version": "0.1.0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def list_tools(self) -> list[dict[str, Any]]:
        result = self.request("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})
