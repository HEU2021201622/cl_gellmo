#!/usr/bin/env python3
"""
合并 test.json 和 test_5props.json 生成完整的测试集

1. 从 test.json 中删除包含 jnk3 和 gsk3b 的任务
2. 从 test_5props.json 中只保留包含 jnk3 和 gsk3b 的任务
3. 合并两部分数据
"""

import json
from tqdm import tqdm

INPUT_TEST = 'all_data/test.json'
INPUT_TEST_5PROPS = 'all_data/test_5props.json'
OUTPUT = 'all_data/test_merged.json'


def main():
    # 第一部分：从 test.json 筛选不含 jnk3/gsk3b 的任务
    print("Processing test.json (removing jnk3/gsk3b tasks)...")
    part1 = []
    with open(INPUT_TEST, 'r') as f:
        test_data = json.load(f)

    for item in tqdm(test_data, desc="Filtering test.json"):
        task = item['task']
        if 'jnk3' not in task and 'gsk3b' not in task:
            part1.append(item)

    print(f"  Part 1 (test.json without jnk3/gsk3b): {len(part1)} samples")

    # 第二部分：从 test_5props.json 筛选含 jnk3/gsk3b 的任务
    print("\nProcessing test_5props.json (keeping jnk3/gsk3b tasks only)...")
    part2 = []
    with open(INPUT_TEST_5PROPS, 'r') as f:
        test_5props_data = json.load(f)

    for item in tqdm(test_5props_data, desc="Filtering test_5props.json"):
        task = item['task']
        if 'jnk3' in task or 'gsk3b' in task:
            part2.append(item)

    print(f"  Part 2 (test_5props.json with jnk3/gsk3b): {len(part2)} samples")

    # 合并
    merged = part1 + part2
    print(f"\nTotal merged samples: {len(merged)}")

    # 输出
    print(f"\nWriting to {OUTPUT}...")
    with open(OUTPUT, 'w') as f:
        json.dump(merged, f, indent=2)

    # 统计任务类型
    tasks = set(item['task'] for item in merged)
    print(f"\nTotal unique tasks: {len(tasks)}")

    return merged


if __name__ == '__main__':
    main()
