#!/usr/bin/env python3
"""
生成新的 train_5props.jsonl

1. 保留原始 train.jsonl 中不包含 jnk3 和 gsk3b 的任务
2. 删除原始的 gsk3b 任务
3. 重新生成包含 jnk3 和/或 gsk3b 的所有任务
"""

import json
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

# 配置
INPUT_TRAIN = 'all_data/train.jsonl'
INPUT_5PROPS = 'all_data/train_5props.jsonl'  # 当前文件，用于读取 jnk3/gsk3b 属性
OUTPUT_TRAIN = 'all_data/train_5props_new.jsonl'

# 5个核心属性
CORE_PROPS = ['jnk3', 'gsk3b', 'drd2', 'plogp', 'qed']

# 阈值设置
THRESHOLDS = {
    'jnk3': 0.05,
    'gsk3b': 0.10,
    'drd2': 0.20,
    'plogp': 1.00,
    'qed': 0.10
}


def generate_new_tasks():
    """生成包含 jnk3 和/或 gsk3b 的所有任务组合"""
    tasks = []
    # 遍历所有可能的组合
    for r in range(1, len(CORE_PROPS) + 1):
        for combo in combinations(CORE_PROPS, r):
            task = '+'.join(combo)
            # 只保留包含 jnk3 或 gsk3b 的任务
            if 'jnk3' in task or 'gsk3b' in task:
                tasks.append(task)
    return sorted(tasks)


def load_smiles_to_5props(input_5props):
    """构建 SMILES 到 5个核心属性值的映射"""
    print("Building SMILES to 5props mapping...")
    smiles_to_props = {}

    with open(input_5props, 'r') as f:
        for line in tqdm(f, desc="Loading 5props"):
            data = json.loads(line)
            source = data['source_smiles']
            target = data['target_smiles']

            # 存储 source 和 target 的属性
            if source not in smiles_to_props:
                smiles_to_props[source] = {}
            if target not in smiles_to_props:
                smiles_to_props[target] = {}

            # 从 properties 中提取 5 个核心属性
            for prop in CORE_PROPS:
                if prop in data['properties']:
                    smiles_to_props[source][prop] = data['properties'][prop]['source']
                    smiles_to_props[target][prop] = data['properties'][prop]['target']

    print(f"Loaded {len(smiles_to_props)} unique molecules")
    return smiles_to_props


def main():
    # 1. 生成包含 jnk3 和/或 gsk3b 的任务
    new_tasks = generate_new_tasks()
    print(f"Generated {len(new_tasks)} tasks (containing jnk3 and/or gsk3b)")

    # 2. 加载 SMILES 到属性的映射
    smiles_to_props = load_smiles_to_5props(INPUT_5PROPS)

    # 3. 读取原始 train.jsonl
    print("\nProcessing train.jsonl...")

    # 用于存储新任务的候选分子对
    new_task_to_pairs = defaultdict(list)

    # 统计
    total_pairs = 0
    original_kept = 0  # 保留的原始任务（不包含 jnk3 和 gsk3b）
    new_added = 0  # 新增的任务样本

    with open(INPUT_TRAIN, 'r') as f:
        for line in tqdm(f, desc="Filtering pairs"):
            data = json.loads(line)
            source = data['source_smiles']
            target = data['target_smiles']
            total_pairs += 1

            # 获取 source 和 target 的属性
            source_props = smiles_to_props.get(source, {})
            target_props = smiles_to_props.get(target, {})

            # 检查是否有缺失的属性
            missing = False
            for prop in CORE_PROPS:
                if prop not in source_props or prop not in target_props:
                    missing = True
                    break

            if missing:
                continue

            # 计算 change
            changes = {}
            for prop in CORE_PROPS:
                change = target_props[prop] - source_props[prop]
                changes[prop] = round(change, 2)

            # 检查原始任务是否包含 jnk3 或 gsk3b
            original_task = data['task']
            has_jnk3_or_gsk3b = 'jnk3' in original_task or 'gsk3b' in original_task

            # 统计保留的原始任务
            if not has_jnk3_or_gsk3b:
                original_kept += 1

            # 对每个新任务检查是否满足条件
            for task in new_tasks:
                task_props = task.split('+')

                # 检查该任务的所有属性 change 是否都大于阈值
                valid = True
                for prop in task_props:
                    if changes[prop] <= THRESHOLDS[prop]:
                        valid = False
                        break

                if valid:
                    # 构建输出数据
                    output_data = {
                        'task': task,
                        'source_smiles': source,
                        'target_smiles': target,
                        'properties': {},
                        'instr_idx': data.get('instr_idx', 0),
                        'split': 'train',
                        'instr_setting': data.get('instr_setting', 'seen'),
                        'scaffold': data.get('scaffold', '')
                    }

                    # 添加所有 5 个核心属性
                    for prop in CORE_PROPS:
                        output_data['properties'][prop] = {
                            'source': source_props[prop],
                            'target': target_props[prop],
                            'change': changes[prop]
                        }

                    new_task_to_pairs[task].append(output_data)
                    new_added += 1

    print(f"\nStatistics:")
    print(f"  Total pairs in train.jsonl: {total_pairs}")
    print(f"  Original tasks (no jnk3/gsk3b): {original_kept}")
    print(f"  New task samples: {new_added}")

    # 4. 输出结果
    print(f"\nWriting to {OUTPUT_TRAIN}...")

    total_samples = 0

    with open(OUTPUT_TRAIN, 'w') as f_out:
        # 首先写入原始 train.jsonl 中不包含 jnk3 和 gsk3b 的任务
        print("Writing original tasks (without jnk3/gsk3b)...")
        with open(INPUT_TRAIN, 'r') as f_in:
            for line in tqdm(f_in, desc="Original tasks"):
                data = json.loads(line)
                original_task = data['task']

                # 跳过包含 jnk3 或 gsk3b 的任务
                if 'jnk3' in original_task or 'gsk3b' in original_task:
                    continue

                source = data['source_smiles']
                target = data['target_smiles']

                # 获取属性
                source_props = smiles_to_props.get(source, {})
                target_props = smiles_to_props.get(target, {})

                # 添加 jnk3 和 gsk3b 属性（如果不存在）
                for prop in ['jnk3', 'gsk3b']:
                    if prop not in data['properties']:
                        data['properties'][prop] = {
                            'source': source_props.get(prop, 0.0),
                            'target': target_props.get(prop, 0.0),
                            'change': round(target_props.get(prop, 0.0) - source_props.get(prop, 0.0), 2)
                        }

                f_out.write(json.dumps(data) + '\n')
                total_samples += 1

        # 然后写入新任务
        print("Writing new tasks (containing jnk3 and/or gsk3b)...")
        for task in new_tasks:
            pairs = new_task_to_pairs[task]
            for pair in pairs:
                f_out.write(json.dumps(pair) + '\n')
                total_samples += 1

    print(f"\nTotal samples written: {total_samples}")

    # 5. 统计每个任务的样本数
    print(f"\nNew task samples:")
    for task in new_tasks:
        count = len(new_task_to_pairs[task])
        print(f"  {task}: {count}")


if __name__ == '__main__':
    main()
