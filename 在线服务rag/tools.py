import json
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
from langchain_core.tools import tool

import config_data4rag as config
from mcp_client import MCPClient, MCPClientError

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

POINT_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")
MCP_CALL_PATTERN = re.compile(r"^\s*/mcp\s+([a-zA-Z0-9_\-\.]+)\s*(\{.*\})?\s*$", re.S)


def _normalize_points(data_points: Any) -> list[tuple[float, float]]:
    parsed: Any = data_points
    if isinstance(data_points, str):
        data_points = data_points.strip()
        try:
            parsed = json.loads(data_points)
        except json.JSONDecodeError:
            pairs = POINT_PATTERN.findall(data_points)
            if not pairs:
                raise ValueError("无法解析数据点，请传入JSON数组或(x, y)格式列表")
            parsed = [[float(x), float(y)] for x, y in pairs]

    if not isinstance(parsed, list) or len(parsed) < 2:
        raise ValueError("数据点至少需要2个")

    points: list[tuple[float, float]] = []
    for point in parsed:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("每个数据点都必须是长度为2的列表或元组")
        x, y = float(point[0]), float(point[1])
        points.append((x, y))

    if len({p[0] for p in points}) < 2:
        raise ValueError("x值需要至少两个不同取值，否则无法拟合直线")

    return points


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    x_matrix = np.vstack([x, np.ones(len(x))]).T
    w, b = np.linalg.lstsq(x_matrix, y, rcond=None)[0]
    next_x = float(max(x) + 1)
    pred_y = float(w * next_x + b)
    return float(w), float(b), next_x, pred_y


def _save_plot(points: list[tuple[float, float]], w: float, b: float, next_x: float, pred_y: float) -> str:
    if plt is None:
        return ""
    output_dir = os.path.join("在线服务rag", "artifacts")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"trend_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")

    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    fit_x = np.linspace(min(x), max(next_x, max(x)), 100)
    fit_y = w * fit_x + b

    plt.figure(figsize=(7, 4))
    plt.scatter(x, y, label="原始数据点", color="#1f77b4")
    plt.plot(fit_x, fit_y, label=f"拟合直线: y={w:.3f}x+{b:.3f}", color="#ff7f0e")
    plt.scatter([next_x], [pred_y], label=f"预测点({next_x:.1f}, {pred_y:.2f})", color="#2ca02c")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("线性拟合结果")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


@tool("calculate", return_direct=False)
def calculate(data_points: Any) -> str:
    points = _normalize_points(data_points)
    w, b, next_x, pred_y = _fit_line(points)
    plot_path = _save_plot(points, w, b, next_x, pred_y)
    return json.dumps(
        {
            "tool": "calculate",
            "equation": f"y = {w:.4f}x + {b:.4f}",
            "next_x": next_x,
            "predicted_y": round(pred_y, 4),
            "sample_size": len(points),
            "plot_path": plot_path,
            "plot_available": bool(plot_path),
        },
        ensure_ascii=False,
    )


def maybe_calculate_from_text(text: str) -> str:
    pairs = POINT_PATTERN.findall(text)
    if len(pairs) < 2:
        return ""
    tool_input = [[float(x), float(y)] for x, y in pairs]
    return calculate.invoke({"data_points": tool_input})


def maybe_call_mcp_tool(text: str) -> str:
    """支持输入: /mcp <tool_name> {json_args}"""
    if not config.mcp_enabled or config.mcp_transport != "stdio":
        return ""

    match = MCP_CALL_PATTERN.match(text)
    if not match:
        return ""

    tool_name = match.group(1)
    args_text = match.group(2) or "{}"
    try:
        arguments = json.loads(args_text)
    except json.JSONDecodeError as exc:
        return json.dumps({"tool": "mcp", "ok": False, "error": f"参数JSON解析失败: {exc}"}, ensure_ascii=False)

    try:
        client = MCPClient(config.mcp_server_cmd, config.mcp_timeout)
        client.initialize()
        result = client.call_tool(tool_name, arguments)
        tools = client.list_tools()
        client.close()
        return json.dumps(
            {
                "tool": "mcp",
                "ok": True,
                "called": tool_name,
                "arguments": arguments,
                "result": result,
                "tool_count": len(tools),
            },
            ensure_ascii=False,
        )
    except MCPClientError as exc:
        return json.dumps({"tool": "mcp", "ok": False, "called": tool_name, "error": str(exc)}, ensure_ascii=False)
