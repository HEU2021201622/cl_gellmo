#!/usr/bin/env python3
"""
分子属性数据集扩展脚本

从 train.jsonl 提取分子，使用 TDC Oracle 预测 JNK3 和 GSK3B，
生成包含 5 个属性 (jnk3, gsk3b, drd2, plogp, qed) 的新数据集。
"""

import json
import os
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from tdc import Oracle
from tqdm import tqdm
import numpy as np

# 配置
N_WORKERS = 60
BATCH_SIZE = 5000
INPUT_TRAIN = 'all_data/train.jsonl'
INPUT_TEST = 'all_data/test.json'
OUTPUT_TRAIN = 'all_data/train_5props.jsonl'
OUTPUT_TEST = 'all_data/test_5props.json'
UNIQUE_SMILES_FILE = 'all_data/unique_smiles.txt'

# 5个核心属性
CORE_PROPS = ['jnk3', 'gsk3b', 'drd2', 'plogp', 'qed']


def predict_oracle_batch(smiles_batch, oracle_name):
    """使用 Oracle 预测一个批次的分子"""
    oracle = Oracle(name=oracle_name)
    return oracle(smiles_batch)


def parallel_predict(smiles_list, oracle_name, n_workers=N_WORKERS, batch_size=BATCH_SIZE):
    """并行预测所有分子的属性"""
    print(f"Predicting {oracle_name} for {len(smiles_list)} molecules with {n_workers} workers...")

    # 分批
    batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]

    results = [None] * len(smiles_list)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(predict_oracle_batch, batch, oracle_name): i
                   for i, batch in enumerate(batches)}

        for future in tqdm(as_completed(futures), total=len(futures), desc=oracle_name):
            batch_idx = futures[future]
            batch_start = batch_idx * batch_size
            try:
                batch_results = future.result()
                for j, score in enumerate(batch_results):
                    results[batch_start + j] = score
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                # 标记失败
                for j in range(len(batches[batch_idx])):
                    results[batch_start + j] = 0.0

    return results


def generate_all_tasks():
    """生成所有31个组合任务 (2^5 - 1)"""
    tasks = []
    for r in range(1, len(CORE_PROPS) + 1):
        for combo in combinations(CORE_PROPS, r):
            tasks.append('+'.join(combo))
    return tasks


def main():
    # 1. 读取唯一分子
    print("Loading unique SMILES...")
    with open(UNIQUE_SMILES_FILE, 'r') as f:
        smiles_list = [line.strip() for line in f]
    print(f"Loaded {len(smiles_list)} unique molecules")

    # 创建 SMILES 到索引的映射
    smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}

    # 2. 并行预测 JNK3 和 GSK3B
    jnk3_scores = parallel_predict(smiles_list, 'JNK3')
    gsk3b_scores = parallel_predict(smiles_list, 'GSK3B')

    # 创建属性字典
    prop_dict = {
        'jnk3': {s: jnk3_scores[i] for i, s in enumerate(smiles_list)},
        'gsk3b': {s: gsk3b_scores[i] for i, s in enumerate(smiles_list)}
    }

    # 统计预测结果
    print(f"\nJNK3 - Non-zero: {sum(1 for v in jnk3_scores if v > 0)}, Zero: {sum(1 for v in jnk3_scores if v == 0)}")
    print(f"GSK3B - Non-zero: {sum(1 for v in gsk3b_scores if v > 0)}, Zero: {sum(1 for v in gsk3b_scores if v == 0)}")

    # 3. 处理训练数据
    print("\nProcessing training data...")
    with open(INPUT_TRAIN, 'r') as f_in, open(OUTPUT_TRAIN, 'w') as f_out:
        for line in tqdm(f_in, desc="Processing train"):
            data = json.loads(line)
            source = data['source_smiles']
            target = data['target_smiles']

            # 添加 jnk3 和 gsk3b 属性
            if 'properties' not in data:
                data['properties'] = {}

            # 添加 jnk3
            jnk3_source = prop_dict['jnk3'].get(source, 0.0)
            jnk3_target = prop_dict['jnk3'].get(target, 0.0) if target else 0.0
            data['properties']['jnk3'] = {
                'source': jnk3_source,
                'target': jnk3_target,
                'change': round(jnk3_target - jnk3_source, 2)
            }

            # 添加 gsk3b
            gsk3b_source = prop_dict['gsk3b'].get(source, 0.0)
            gsk3b_target = prop_dict['gsk3b'].get(target, 0.0) if target else 0.0
            data['properties']['gsk3b'] = {
                'source': gsk3b_source,
                'target': gsk3b_target,
                'change': round(gsk3b_target - gsk3b_source, 2)
            }

            # 保留原有属性，只更新这两个
            f_out.write(json.dumps(data) + '\n')

    print(f"Saved training data to {OUTPUT_TRAIN}")

    # 4. 计算训练集中5个属性的中位数（用于测试集筛选）
    print("\nCalculating property medians...")
    prop_values = {p: [] for p in CORE_PROPS}

    with open(INPUT_TRAIN, 'r') as f:
        for line in f:
            data = json.loads(line)
            props = data.get('properties', {})
            for p in CORE_PROPS:
                if p in props:
                    # 使用 source 值
                    val = props[p].get('source', 0.0)
                    if val is not None:
                        prop_values[p].append(val)

    medians = {p: np.median(vals) for p, vals in prop_values.items()}
    print(f"Property medians: {medians}")

    # 5. 生成所有31个组合任务
    all_tasks = generate_all_tasks()
    print(f"\nGenerated {len(all_tasks)} combination tasks")

    # 6. 生成测试数据
    print("\nProcessing test data...")

    # 读取所有测试分子
    with open(INPUT_TEST, 'r') as f:
        test_data = json.load(f)

    # 收集所有唯一的测试分子及其属性
    test_molecules = {}  # smiles -> properties dict
    for item in test_data:
        smiles = item['source_smiles']
        if smiles not in test_molecules:
            test_molecules[smiles] = item.get('properties', {})

    # 为测试分子添加 jnk3 和 gsk3b
    print("Adding JNK3 and GSK3B to test molecules...")
    for smiles, props in tqdm(test_molecules.items()):
        props['jnk3'] = {'source': prop_dict['jnk3'].get(smiles, 0.0)}
        props['gsk3b'] = {'source': prop_dict['gsk3b'].get(smiles, 0.0)}

    # 重新计算正确的中位数（从新的训练数据中）
    print("\nRecalculating property medians from train_5props.jsonl...")
    prop_values = {p: [] for p in CORE_PROPS}

    with open(OUTPUT_TRAIN, 'r') as f:
        for line in f:
            data = json.loads(line)
            props = data.get('properties', {})
            for p in CORE_PROPS:
                if p in props:
                    val = props[p].get('source', 0.0)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        prop_values[p].append(val)

    medians = {p: np.median(vals) if vals else 0.0 for p, vals in prop_values.items()}
    print(f"Property medians: {medians}")

    # 为每个任务筛选分子
    test_output = {}

    for task in tqdm(all_tasks, desc="Generating test tasks"):
        task_props = task.split('+')

        # 筛选条件：所有任务属性值都低于中位数
        candidates = []
        for smiles, props in test_molecules.items():
            valid = True
            changes = {}
            for p in task_props:
                if p in props:
                    val = props[p].get('source', 0.0)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        val = 0.0
                    median_val = medians.get(p, 0.0)
                    if np.isnan(median_val):
                        median_val = 0.0
                    if val >= median_val:
                        valid = False
                        break
                    # 计算 change（目标是提升属性）
                    change_val = round(median_val - val, 2)
                    if np.isnan(change_val) or np.isinf(change_val):
                        change_val = 0.0
                    changes[p] = {
                        'source': val,
                        'change': change_val
                    }
                else:
                    valid = False
                    break

            if valid:
                candidates.append({
                    'task': task,
                    'source_smiles': smiles,
                    'properties': changes,
                    'target_smiles': None,
                    'split': 'test'
                })

        # 最多取500个
        test_output[task] = candidates[:500]

    # 保存测试数据 - 使用自定义编码器处理 NaN
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return 0.0
                return obj
            return super().default(obj)

    # 转换为列表格式（与原始test.json格式一致）
    test_list = []
    for task, items in test_output.items():
        for item in items:
            test_list.append(item)

    with open(OUTPUT_TEST, 'w') as f:
        json.dump(test_list, f, indent=2, cls=NumpyEncoder)

    print(f"Saved {len(test_list)} test samples to {OUTPUT_TEST}")

    # 统计每个任务的分子数量
    print("\nTest samples per task:")
    for task in all_tasks:
        count = len(test_output[task])
        print(f"  {task}: {count}")


if __name__ == '__main__':
    main()
