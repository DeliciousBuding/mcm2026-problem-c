"""
基础冒烟测试：验证核心依赖可导入、pipeline 模块语法正确。
"""
import sys
from pathlib import Path

# 将 src 添加到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """测试核心依赖是否可导入"""
    import numpy as np
    import pandas as pd
    import matplotlib
    import scipy
    assert np.__version__
    assert pd.__version__


def test_pipeline_syntax():
    """测试 pipeline.py 语法正确（可导入）"""
    # 仅检查语法，不实际运行
    import ast
    pipeline_path = Path(__file__).parent.parent / "src" / "pipeline.py"
    with open(pipeline_path, "r", encoding="utf-8") as f:
        code = f.read()
    # 如果语法错误，ast.parse 会抛出 SyntaxError
    ast.parse(code)


def test_data_file_exists():
    """测试数据文件存在（CI 环境可能跳过）"""
    data_path = Path(__file__).parent.parent / "data" / "2026_MCM_Problem_C_Data.csv"
    # 数据文件可能在 .gitignore 中，仅作提示
    if not data_path.exists():
        import pytest
        pytest.skip("数据文件未提交到仓库（正常，已在 .gitignore）")
    assert data_path.stat().st_size > 0


def test_outputs_dir():
    """测试 outputs 目录结构"""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    # outputs 可能被忽略，仅检查目录存在或可创建
    if outputs_dir.exists():
        assert outputs_dir.is_dir()
