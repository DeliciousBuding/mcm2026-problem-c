"""
基础冒烟测试：验证核心依赖可导入、pipeline 模块语法正确。
"""
import sys
import json
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


# =========================
# 口径回归测试（防止 A/V 分离污染复发）
# =========================

def test_yellow_action_not_activate_judge_save():
    """
    防回归测试：Yellow 的 Action 不应该是 'Activate Judge Save'
    任务清单要求：Yellow 显示 Warning，不写 Activate Judge Save
    """
    daws_tiers_path = Path(__file__).parent.parent / "outputs" / "daws_tiers.csv"
    if not daws_tiers_path.exists():
        import pytest
        pytest.skip("daws_tiers.csv 未生成，跳过回归测试")
    
    with open(daws_tiers_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Yellow 行不应包含 "Activate Judge Save"
    lines = content.strip().split("\n")
    for line in lines[1:]:  # 跳过 header
        if "Yellow" in line:
            assert "Activate Judge Save" not in line, \
                f"口径回归：Yellow 的 Action 不应该是 'Activate Judge Save'，发现: {line}"


def test_feasibility_no_delta_min_gap():
    """
    Hard-5 防回归测试：可行域约束清单中明确声明无 Delta/min-gap
    """
    gate_path = Path(__file__).parent.parent / "outputs" / "audit_block5_gate.json"
    if not gate_path.exists():
        import pytest
        pytest.skip("audit_block5_gate.json 未生成，跳过回归测试")
    
    with open(gate_path, "r", encoding="utf-8") as f:
        gate_info = json.load(f)
    
    # 检查 feasibility_constraints 存在且声明了 NOT USED
    assert "feasibility_constraints" in gate_info, "缺少 feasibility_constraints 字段"
    fc = gate_info["feasibility_constraints"]
    assert "delta_min_gap" in fc, "缺少 delta_min_gap 声明"
    assert "NOT USED" in fc["delta_min_gap"], "delta_min_gap 应声明 NOT USED"
    assert "hard5_verification" in gate_info, "缺少 hard5_verification 字段"


def test_lp_milp_scope_diagnostic_only():
    """
    Hard-2 防回归测试：LP/MILP 必须声明为 diagnostic tool ONLY
    
    口径锁定：
    - lp_milp_scope.role = "diagnostic tool ONLY"
    - LP/MILP 不用于采样、可行性检查、后验估计、机制指标
    - 论文叙事应与此一致
    """
    gate_path = Path(__file__).parent.parent / "outputs" / "audit_block5_gate.json"
    scope_path = Path(__file__).parent.parent / "outputs" / "audit_lp_milp_scope.json"
    
    if not gate_path.exists():
        import pytest
        pytest.skip("audit_block5_gate.json 未生成，跳过回归测试")
    
    with open(gate_path, "r", encoding="utf-8") as f:
        gate_info = json.load(f)
    
    # 检查 lp_milp_scope 存在且 role 为 diagnostic tool ONLY
    assert "lp_milp_scope" in gate_info, "缺少 lp_milp_scope 字段"
    lp_scope = gate_info["lp_milp_scope"]
    assert lp_scope["role"] == "diagnostic tool ONLY, not main model", \
        f"lp_milp_scope.role 应为 'diagnostic tool ONLY, not main model'，实际为 {lp_scope['role']}"
    
    # 检查 NOT_used_for 列表包含关键项
    not_used = lp_scope.get("NOT_used_for", [])
    assert any("sampling" in item.lower() for item in not_used), "NOT_used_for 应包含 sampling"
    assert any("feasibility" in item.lower() for item in not_used), "NOT_used_for 应包含 feasibility"
    
    # 检查 hard2_verification 字段
    assert "hard2_verification" in gate_info, "缺少 hard2_verification 字段"
    
    # 检查专用 audit JSON（如果存在）
    if scope_path.exists():
        with open(scope_path, "r", encoding="utf-8") as f:
            scope_audit = json.load(f)
        assert scope_audit.get("verification_passed") == True, "audit_lp_milp_scope.json verification_passed 应为 True"
        assert scope_audit.get("lp_milp_role", {}).get("status") == "DIAGNOSTIC TOOL ONLY", \
            "lp_milp_role.status 应为 DIAGNOSTIC TOOL ONLY"


def test_beta_sensitivity_output_exists():
    """
    Hard-7 防回归测试：beta 敏感性分析产物存在
    """
    csv_path = Path(__file__).parent.parent / "outputs" / "beta_sensitivity.csv"
    json_path = Path(__file__).parent.parent / "outputs" / "audit_beta_sensitivity.json"
    
    if not csv_path.exists() or not json_path.exists():
        import pytest
        pytest.skip("beta_sensitivity 产物未生成，跳过回归测试")
    
    # 检查 CSV 不为空
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) > 1, "beta_sensitivity.csv 应有数据行"
    
    # 检查 JSON 包含验收字段
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert "all_integrity_pass" in summary, "缺少 all_integrity_pass 字段"
    assert "conclusion_stable" in summary, "缺少 conclusion_stable 字段"
    assert "beta_values_tested" in summary, "缺少 beta_values_tested 字段"
