import json
import re
from typing import Any

import numpy as np
from langchain_core.tools import tool


POINT_PATTERN = re.compile(r"\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)")


def _normalize_points(data_points: Any) -> list[tuple[float, float]]:
    """将输入归一化为[(x, y), ...]格式，支持JSON字符串或Python列表。"""
    parsed: Any = data_points
    if isinstance(data_points, str):
        data_points = data_points.strip()
        try:
            parsed = json.loads(data_points)
        except json.JSONDecodeError:
            # 允许用户输入如"(1, 100), (2, 120)"
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
    return points


@tool("calculate", return_direct=False)
def calculate(data_points: Any) -> str:
    """对二维数据点做最小二乘线性拟合，预测下一个x的y值。"""
    points = _normalize_points(data_points)
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)

    x_matrix = np.vstack([x, np.ones(len(x))]).T
    w, b = np.linalg.lstsq(x_matrix, y, rcond=None)[0]

    next_x = float(max(x) + 1)
    pred_y = float(w * next_x + b)

    return json.dumps(
        {
            "tool": "calculate",
            "equation": f"y = {w:.4f}x + {b:.4f}",
            "next_x": next_x,
            "predicted_y": round(pred_y, 4),
            "sample_size": len(points),
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
