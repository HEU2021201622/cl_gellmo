# 分子属性数据集扩展 Memory

## 完成日期
2026-03-24

## 任务概述
从 `all_data/train.jsonl` 提取所有分子，使用 TDC Oracle 预测 JNK3 和 GSK3B 属性，生成包含 5 个核心属性 (jnk3, gsk3b, drd2, plogp, qed) 的新数据集。

## 输出文件

### 1. train_5props.jsonl
- **路径**: `all_data/train_5props.jsonl`
- **大小**: 0.61 GB
- **记录数**: 863,451
- **属性**: bbbp, drd2, gsk3b, hia, mutagenicity, plogp, qed, jnk3
- **说明**: 在原始训练数据基础上添加了 jnk3 和 gsk3b 属性

### 2. test_5props.json
- **路径**: `all_data/test_5props.json`
- **大小**: 5.61 MB
- **总样本数**: 15,500
- **任务数**: 31 个组合任务
- **每个任务样本数**: 500

## TDC Oracle 预测结果

### JNK3
- 非零值: 140,733
- 零值: 88,285
- 中位数: 0.01

### GSK3B
- 非零值: 168,177
- 零值: 60,841
- 中位数: 0.03

## 五个核心属性及中位数

| 属性 | 中位数 | 说明 |
|------|--------|------|
| jnk3 | 0.01 | TDC Oracle 预测 |
| gsk3b | 0.03 | TDC Oracle 预测 |
| drd2 | 0.02 | 原始数据 |
| plogp | 0.69 | 原始数据 |
| qed | 0.70 | 原始数据 |

## 31 个组合任务

单属性 (5): jnk3, gsk3b, drd2, plogp, qed

两属性 (10): jnk3+gsk3b, jnk3+drd2, jnk3+plogp, jnk3+qed, gsk3b+drd2, gsk3b+plogp, gsk3b+qed, drd2+plogp, drd2+qed, plogp+qed

三属性 (10): jnk3+gsk3b+drd2, jnk3+gsk3b+plogp, jnk3+gsk3b+qed, jnk3+drd2+plogp, jnk3+drd2+qed, jnk3+plogp+qed, gsk3b+drd2+plogp, gsk3b+drd2+qed, gsk3b+plogp+qed, drd2+plogp+qed

四属性 (5): jnk3+gsk3b+drd2+plogp, jnk3+gsk3b+drd2+qed, jnk3+gsk3b+plogp+qed, jnk3+drd2+plogp+qed, gsk3b+drd2+plogp+qed

五属性 (1): jnk3+gsk3b+drd2+plogp+qed

## 关键代码

### 生成脚本
`generate_5props_dataset.py`

### 运行环境
- Conda 环境: pmo
- 工作目录: `/home/xy/workspace/projects/CL_gellmo/GeLLMO/data`
- 并行进程数: 60

### TDC Oracle 使用示例
```python
from tdc import Oracle

# JNK3 预测
oracle = Oracle(name='JNK3')
result = oracle(['SMILES1', 'SMILES2', ...])

# GSK3B 预测
oracle = Oracle(name='GSK3B')
result = oracle(['SMILES1', 'SMILES2', ...])
```

## 注意事项

1. **必须使用 pmo 环境**: TDC Oracle 只在该环境可用
2. **工作目录**: 必须在 `/home/xy/workspace/projects/CL_gellmo/GeLLMO/data` 执行
3. **中位数计算**: 必须从 `train_5props.jsonl` 计算，而非原始 `train.jsonl`（原始数据没有 jnk3）
4. **测试集筛选**: 所有任务属性值必须低于训练集对应属性的中位数

## 唯一分子数
229,018 个唯一分子（从 train.jsonl 提取）
