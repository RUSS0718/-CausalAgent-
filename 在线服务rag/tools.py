import json
import os
import re
from datetime import datetime
from typing import Any

import numpy as np
from langchain_core.tools import tool

try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib在部分环境可能不可用
    plt = None


POINT_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")


def _normalize_points(data_points: Any) -> list[tuple[float, float]]:
    """将输入归一化为[(x, y), ...]格式，支持JSON字符串或Python列表。"""
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

    unique_x = {p[0] for p in points}
    if len(unique_x) < 2:
        raise ValueError("x值需要至少两个不同取值，否则无法拟合直线")

    return points


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """最小二乘拟合 y = wx + b，并预测下一个x点。"""
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)

    x_matrix = np.vstack([x, np.ones(len(x))]).T
    w, b = np.linalg.lstsq(x_matrix, y, rcond=None)[0]

    next_x = float(max(x) + 1)
    pred_y = float(w * next_x + b)
    return float(w), float(b), next_x, pred_y


def _save_plot(points: list[tuple[float, float]], w: float, b: float, next_x: float, pred_y: float) -> str:
    """保存拟合图，返回本地路径；如当前环境无matplotlib则返回空字符串。"""
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
    """对二维数据点做最小二乘线性拟合，预测下一个x的y值，并输出拟合图路径。"""
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
    """从用户文本中尝试提取(x, y)序列，成功则返回工具结果。"""
    pairs = POINT_PATTERN.findall(text)
    if len(pairs) < 2:
        return ""
    tool_input = [[float(x), float(y)] for x, y in pairs]
    return calculate.invoke({"data_points": tool_input})
