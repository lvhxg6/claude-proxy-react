# Compaction 修复报告

**日期**: 2026-03-02
**修复版本**: v1.0
**审查人**: codex-reviewer

## 1. 问题摘要

根据 codex 的审查报告 (review-20260302-1302.md)，发现 compaction 链路存在三个关键问题：

1. **高危**: summary 为空时仍然执行边界裁剪，导致历史上下文大量丢失
2. **高危**: 对客户端回传的 compaction block 缺少保护性验证
3. **中危**: usage 统计口径与 Anthropic 官方协议不一致

## 2. 根本原因分析

### 问题链路

```
第一轮 (成功):
  代理生成摘要 → summary_len=1971 ✅
  转换为 SSE → compaction_delta 发送 ✅

第二轮 (客户端问题):
  Claude Code 回传 → {type: "compaction", content: null} ❌

第三轮 (灾难):
  代理收到 null → 仍然裁剪历史 (dropped 103 messages) ❌
  转换消息 → 不添加摘要文本 (content=null) ❌
  结果: 上下文严重丢失 ❌
```

### 代码证据

**问题代码** (修复前):
```python
# request_converter.py:171-173
boundary_index, summary = _find_compaction_boundary(messages)
if boundary_index >= 0:
    messages = _prune_messages_at_boundary(messages, boundary_index, summary)
    # ❌ 无论 summary 是否为空都会裁剪
```

## 3. 修复方案

### 3.1 安全保护 (P0 - 已完成)

**文件**: `src/conversion/request_converter.py`

**修改内容**:
```python
# 第 171-185 行
boundary_index, summary = _find_compaction_boundary(messages)
if boundary_index >= 0:
    if summary and summary.strip():  # ✅ 只有摘要非空才裁剪
        messages = _prune_messages_at_boundary(messages, boundary_index, summary)
        logger.info(
            f"Compaction boundary applied: boundary_index={boundary_index}, "
            f"summary_len={len(summary)}"
        )
    else:  # ✅ 摘要为空时保留全量历史
        logger.warning(
            f"Compaction boundary found at index {boundary_index} but summary is "
            f"empty/null. Skipping pruning to preserve context. This may indicate "
            f"a client-side issue with compaction_delta handling or persistence."
        )
```

**效果**: 防止在摘要丢失时执行危险的历史裁剪操作

### 3.2 可观测性增强 (P0 - 已完成)

**文件 1**: `src/api/endpoints.py`

**修改内容** (第 331-340 行):
```python
# 生成摘要时记录哈希
import hashlib
summary_hash = hashlib.md5(summary.encode()).hexdigest()[:8] if summary else "null"

logger.info(
    f"[{short_id}] Compaction summary generated: {summary_elapsed:.3f}s, "
    f"input={compaction_input_tokens}, output={compaction_output_tokens}, "
    f"summary_len={len(summary)}, hash={summary_hash}"  # ✅ 新增 hash
)
```

**文件 2**: `src/conversion/request_converter.py`

**修改内容** (_find_compaction_boundary 函数):
```python
# 第 123-149 行
import hashlib

for i in range(len(messages) - 1, -1, -1):
    # ... 查找逻辑 ...
    summary = getattr(block, "content", None)

    # ✅ 记录收到的 compaction block
    if summary:
        summary_hash = hashlib.md5(summary.encode()).hexdigest()[:8]
        logger.info(
            f"Compaction block received: index={i}, "
            f"summary_len={len(summary)}, hash={summary_hash}"
        )
    else:
        logger.warning(
            f"Compaction block received with null/empty content at index={i}"
        )
    return i, summary
```

**效果**:
- 可以对比"发出去的 hash"和"收回来的 hash"
- 快速定位摘要丢失发生在哪个环节

### 3.3 Usage 统计修正 (P1 - 已完成)

**文件**: `src/core/compaction.py`

**修改内容** (第 270-272 行):
```python
# 修复前
"usage": {
    "input_tokens": message_input_tokens,
    "output_tokens": compaction_output_tokens + message_output_tokens,  # ❌
    ...
}

# 修复后
"usage": {
    "input_tokens": message_input_tokens,
    "output_tokens": message_output_tokens,  # ✅ 只统计 message iteration
    "iterations": [
        {
            "type": "compaction",
            "output_tokens": compaction_output_tokens,  # ✅ compaction 单独统计
        },
        ...
    ],
}
```

**效果**: 符合 Anthropic 官方协议，顶层 usage 不包含 compaction token

### 3.4 回归测试 (P1 - 已完成)

**文件**: `tests/test_compaction.py` (新建)

**测试覆盖**:
1. ✅ `test_find_compaction_boundary_with_valid_summary` - 正常摘要检测
2. ✅ `test_find_compaction_boundary_with_null_summary` - 空摘要检测
3. ✅ `test_find_compaction_boundary_not_found` - 无 compaction block
4. ✅ `test_prune_messages_preserves_from_boundary` - 裁剪逻辑正确性
5. ✅ `test_convert_skips_pruning_when_summary_null` - **核心**: 空摘要不裁剪
6. ✅ `test_convert_applies_pruning_when_summary_valid` - 有效摘要才裁剪
7. ✅ `test_usage_excludes_compaction_from_top_level` - usage 统计正确性

**测试结果**:
```
============================= test session starts ==============================
collected 7 items

tests/test_compaction.py::TestCompactionBoundaryPruning::test_find_compaction_boundary_with_valid_summary PASSED [ 14%]
tests/test_compaction.py::TestCompactionBoundaryPruning::test_find_compaction_boundary_with_null_summary PASSED [ 28%]
tests/test_compaction.py::TestCompactionBoundaryPruning::test_find_compaction_boundary_not_found PASSED [ 42%]
tests/test_compaction.py::TestCompactionBoundaryPruning::test_prune_messages_preserves_from_boundary PASSED [ 57%]
tests/test_compaction.py::TestCompactionSafetyProtection::test_convert_skips_pruning_when_summary_null PASSED [ 71%]
tests/test_compaction.py::TestCompactionSafetyProtection::test_convert_applies_pruning_when_summary_valid PASSED [ 85%]
tests/test_compaction.py::TestUsageStatistics::test_usage_excludes_compaction_from_top_level PASSED [100%]

============================== 7 passed in 0.24s
```

## 4. 修改文件清单

| 文件 | 修改类型 | 行数变化 | 说明 |
|------|---------|---------|------|
| `src/conversion/request_converter.py` | 修改 | +27 | 安全保护 + 可观测性 |
| `src/api/endpoints.py` | 修改 | +4 | 可观测性 (摘要生成时) |
| `src/core/compaction.py` | 修改 | -1 | Usage 统计修正 |
| `tests/test_compaction.py` | 新建 | +193 | 完整回归测试 |

## 5. 验证结果

### 5.1 单元测试
- ✅ 新增 7 个 compaction 测试全部通过
- ✅ 现有测试 (test_main.py) 全部通过
- ✅ 代码格式化 (black + isort) 通过

### 5.2 关键场景验证

**场景 1: 摘要为空时的保护**
```python
# 输入: compaction block with content=null
# 预期: 保留全量历史，记录 warning
# 实际: ✅ 测试通过，历史未被裁剪
```

**场景 2: 摘要有效时的正常裁剪**
```python
# 输入: compaction block with content="Valid summary"
# 预期: 裁剪历史，保留边界之后的消息
# 实际: ✅ 测试通过，裁剪逻辑正确
```

**场景 3: Usage 统计正确性**
```python
# 预期: 顶层 output_tokens 只包含 message iteration
# 实际: ✅ 测试通过，compaction token 在 iterations 中
```

## 6. 风险评估

### 已消除的风险
- ✅ **高危**: 上下文丢失风险 (summary 为空时不再裁剪)
- ✅ **中危**: Usage 统计不准确 (已修正为官方协议)

### 残留风险
- ⚠️ **客户端侧**: Claude Code 为何丢失 compaction content 仍需进一步诊断
  - 建议: 抓包验证 Claude Code 的 SSE 处理逻辑
  - 当前: 服务端已有足够保护，即使客户端有 bug 也不会导致灾难

## 7. 后续建议

### 短期 (P2)
1. 改进 compaction prompt 为结构化模板
2. 添加 SSE 协议兼容性测试
3. 监控生产环境中 "summary=null" 的出现频率

### 中期 (P3)
1. 与 Claude Code 团队沟通 compaction_delta 处理逻辑
2. 考虑添加 `pause_after_compaction=true` 作为兜底模式
3. 优化摘要质量 (保留关键文件路径、函数名等)

## 8. 协议一致性确认

| 协议要求 | 修复前 | 修复后 |
|---------|-------|-------|
| 只有摘要非空才裁剪历史 | ❌ | ✅ |
| 顶层 usage 不含 compaction token | ❌ | ✅ |
| SSE 格式: content_block_start/delta/stop | ✅ | ✅ |
| 摘要为空时保留上下文 | ❌ | ✅ |

## 9. 总结

本次修复完全解决了 codex 审查报告中指出的三个核心问题：

1. **安全保护**: 摘要为空时不再执行危险的历史裁剪
2. **可观测性**: 增加摘要哈希日志，便于诊断丢失问题
3. **协议一致性**: Usage 统计符合 Anthropic 官方规范

所有修改均通过完整的单元测试验证，代码质量符合项目规范。

---

**修复完成时间**: 2026-03-02 14:30
**测试状态**: ✅ 全部通过
**代码审查**: 待 codex-reviewer 确认
