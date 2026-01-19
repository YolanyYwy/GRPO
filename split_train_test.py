"""
Script to split tasks.json into train and test sets for all domains.

This script will:
1. Read tasks.json from each domain directory
2. Split tasks into train (80%) and test (20%) sets
3. Save train_tasks.json and test_tasks.json in each domain directory
4. Update split_tasks.json to include train and test splits
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Train/test split ratio
TRAIN_RATIO = 0.8

# Base directory
BASE_DIR = Path(r"/data/yuweiyao/AGentCL/data/tau2/domains")

# Domains to process
DOMAINS = ["delivery", "instore", "ota", "airline", "retail", "telecom"]


def split_tasks(tasks: List[Dict[Any, Any]], train_ratio: float = 0.8) -> tuple[List[Dict], List[Dict]]:
    """Split tasks into train and test sets.

    Args:
        tasks: List of task dictionaries
        train_ratio: Ratio of tasks to use for training (default: 0.8)

    Returns:
        Tuple of (train_tasks, test_tasks)
    """
    # Shuffle tasks
    shuffled_tasks = tasks.copy()
    random.shuffle(shuffled_tasks)

    # Calculate split point
    split_idx = int(len(shuffled_tasks) * train_ratio)

    # Split
    train_tasks = shuffled_tasks[:split_idx]
    test_tasks = shuffled_tasks[split_idx:]

    return train_tasks, test_tasks


def process_domain(domain_name: str, domain_dir: Path):
    """Process a single domain directory.

    Args:
        domain_name: Name of the domain
        domain_dir: Path to the domain directory
    """
    tasks_file = domain_dir / "tasks.json"

    if not tasks_file.exists():
        print(f"  [SKIP] {domain_name}: tasks.json not found")
        return

    # Read tasks
    print(f"  [INFO] {domain_name}: Reading tasks.json...")
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    total_tasks = len(tasks)
    print(f"  [INFO] {domain_name}: Found {total_tasks} tasks")

    # Split tasks
    train_tasks, test_tasks = split_tasks(tasks, TRAIN_RATIO)

    print(f"  [INFO] {domain_name}: Split into {len(train_tasks)} train and {len(test_tasks)} test tasks")

    # Save train tasks
    train_file = domain_dir / "train_tasks.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_tasks, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] {domain_name}: Saved train_tasks.json")

    # Save test tasks
    test_file = domain_dir / "test_tasks.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_tasks, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] {domain_name}: Saved test_tasks.json")

    # Update split_tasks.json
    split_file = domain_dir / "split_tasks.json"

    # Read existing splits if available
    if split_file.exists():
        with open(split_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
    else:
        splits = {}

    # Add train and test splits
    splits['train'] = [task['id'] for task in train_tasks]
    splits['test'] = [task['id'] for task in test_tasks]

    # Keep base split (all tasks) if it exists
    if 'base' not in splits:
        splits['base'] = [task['id'] for task in tasks]

    # Save updated splits
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"  [SAVE] {domain_name}: Updated split_tasks.json")

    print(f"  [DONE] {domain_name}: Successfully processed")
    print()


def main():
    """Main function to process all domains."""
    print("=" * 60)
    print("Task Train/Test Split Script")
    print("=" * 60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Train ratio: {TRAIN_RATIO} ({TRAIN_RATIO*100}%)")
    print(f"Test ratio: {1-TRAIN_RATIO} ({(1-TRAIN_RATIO)*100}%)")
    print(f"Base directory: {BASE_DIR}")
    print("=" * 60)
    print()

    # Process each domain
    for domain in DOMAINS:
        domain_dir = BASE_DIR / domain

        if not domain_dir.exists():
            print(f"  [SKIP] {domain}: Directory not found")
            print()
            continue

        print(f"Processing domain: {domain}")
        print("-" * 60)

        try:
            process_domain(domain, domain_dir)
        except Exception as e:
            print(f"  [ERROR] {domain}: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print("All domains processed!")
    print("=" * 60)

    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for domain in DOMAINS:
        domain_dir = BASE_DIR / domain
        train_file = domain_dir / "train_tasks.json"
        test_file = domain_dir / "test_tasks.json"

        if train_file.exists() and test_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                train_count = len(json.load(f))
            with open(test_file, 'r', encoding='utf-8') as f:
                test_count = len(json.load(f))

            print(f"{domain:15s}: {train_count:3d} train, {test_count:3d} test, {train_count+test_count:3d} total")
        else:
            print(f"{domain:15s}: Not processed")

    print("-" * 60)


if __name__ == "__main__":
    main()
