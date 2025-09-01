#!/usr/bin/env python3
"""
Rush Hour Data Converter for veRL Training
Converts individual puzzle folders to veRL-compatible parquet format
"""

import pandas as pd
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

def extract_solution_from_file(solution_path: str) -> str:
    """Extract clean solution from solution.txt file"""
    with open(solution_path, 'r') as f:
        content = f.read()
    
    # Find solution steps
    lines = content.split('\n')
    solution_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('Step '):
            solution_lines.append(line)
    
    if solution_lines:
        return "<solution>\n" + "\n".join(solution_lines) + "\n</solution>"
    return ""

def convert_puzzles_to_parquet(
    data_dir: str = "data/rl-data", 
    train_output: str = "data/rushhour_train.parquet",
    val_output: str = "data/rushhour_val.parquet",
    puzzle_range: Tuple[int, int] = (151, 3150),
    val_split_ratio: float = 0.2
) -> Tuple[str, str]:
    """
    Convert puzzle folders to veRL parquet format with balanced difficulty split
    
    Args:
        data_dir: Directory containing puzzle folders
        train_output: Output path for training parquet
        val_output: Output path for validation parquet  
        puzzle_range: Tuple of (start, end) puzzle IDs to process
        val_split_ratio: Fraction of data to use for validation (0.2 = 20%)
    
    Returns:
        Tuple of (train_path, val_path)
    """
    
    print(f"Processing puzzles {puzzle_range[0]}-{puzzle_range[1]}...")
    
    # Collect all puzzles first
    all_records = []
    
    for puzzle_id in range(puzzle_range[0], puzzle_range[1] + 1):
        puzzle_folder = Path(data_dir) / f"puzzle{puzzle_id}"
        
        if not puzzle_folder.exists() or not puzzle_folder.is_dir():
            continue
        
        try:
            # Load prompt
            prompt_path = puzzle_folder / "prompt.txt"
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
            
            # Load puzzle state
            state_path = puzzle_folder / "puzzle_state.json"  
            with open(state_path, 'r') as f:
                puzzle_state = json.load(f)
            
            # Load solution
            solution_path = puzzle_folder / "solution.txt"
            optimal_solution = extract_solution_from_file(solution_path)
            
            if not optimal_solution:
                print(f"Warning: No valid solution found for puzzle{puzzle_id}, skipping...")
                continue
            
            # Create record in veRL format
            record = {
                "prompt": prompt,
                "data_source": "rushhour",  # For reward function dispatch
                # Validation data (expected by veRL framework)
                "reward_model": {
                    "ground_truth": optimal_solution  # Reference solution for validation
                },
                # Reward function data (passed as extra_info to compute_score)
                "extra_info": {
                    "puzzle_state": json.dumps(puzzle_state),  # JSON string for reward function
                    "optimal_moves": puzzle_state["puzzle_info"]["total_moves_in_solution"]  # Integer for comparison
                },
                # Additional metadata (optional)
                "puzzle_id": puzzle_id,
                "difficulty": puzzle_state["puzzle_info"]["difficulty"],
                "grid_size": puzzle_state["puzzle_info"]["grid_size"],
                "car_position": json.dumps(puzzle_state["car_position"]),
                "exit_position": json.dumps(puzzle_state["exit_position"])
            }
            
            all_records.append(record)
            
        except Exception as e:
            print(f"Error processing puzzle{puzzle_id}: {e}")
            continue
    
    if not all_records:
        print("✗ No puzzles found!")
        return "", ""
    
    # Group by difficulty
    from collections import defaultdict
    import random
    
    difficulty_groups = defaultdict(list)
    for record in all_records:
        difficulty_groups[record['difficulty']].append(record)
    
    print(f"Total puzzles loaded: {len(all_records)}")
    for difficulty, records in difficulty_groups.items():
        print(f"  {difficulty}: {len(records)} puzzles")
    
    # Split each difficulty group proportionally
    train_records = []
    val_records = []
    
    for difficulty, records in difficulty_groups.items():
        # Shuffle for random split
        random.shuffle(records)
        
        # Calculate split point
        val_count = int(len(records) * val_split_ratio)
        train_count = len(records) - val_count
        
        # Split the records
        val_records.extend(records[:val_count])
        train_records.extend(records[val_count:])
        
        print(f"  {difficulty}: {train_count} train + {val_count} val")
    
    # Create output directories
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    os.makedirs(os.path.dirname(val_output), exist_ok=True)
    
    # Save training data
    if train_records:
        train_df = pd.DataFrame(train_records)
        train_df.to_parquet(train_output, index=False)
        print(f"✓ Training data: {len(train_records)} puzzles → {train_output}")
    else:
        print("✗ No training records found!")
    
    # Save validation data
    if val_records:
        val_df = pd.DataFrame(val_records)
        val_df.to_parquet(val_output, index=False)
        print(f"✓ Validation data: {len(val_records)} puzzles → {val_output}")
    else:
        print("✗ No validation records found!")
    
    # Print final statistics
    if train_records:
        train_difficulties = pd.DataFrame(train_records)['difficulty'].value_counts().sort_index()
        print(f"Training difficulty distribution: {dict(train_difficulties)}")
    
    if val_records:
        val_difficulties = pd.DataFrame(val_records)['difficulty'].value_counts().sort_index()
        print(f"Validation difficulty distribution: {dict(val_difficulties)}")
    
    return train_output, val_output

def verify_parquet_files(train_path: str, val_path: str):
    """Verify the generated parquet files are valid"""
    print("\n=== Verification ===")
    
    try:
        # Check training file
        if os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            print(f"✓ Training file loaded: {len(train_df)} rows")
            print(f"  Columns: {list(train_df.columns)}")
            print(f"  Sample prompt length: {len(train_df.iloc[0]['prompt'])} chars")
            
        # Check validation file  
        if os.path.exists(val_path):
            val_df = pd.read_parquet(val_path)
            print(f"✓ Validation file loaded: {len(val_df)} rows")
            print(f"  Columns: {list(val_df.columns)}")
            
        print("✓ Parquet files are valid!")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Rush Hour puzzles to veRL parquet format')
    parser.add_argument('--data_dir', default='data/rl-data', 
                       help='Directory containing puzzle folders')
    parser.add_argument('--train_output', default='data/rushhour_train.parquet',
                       help='Output path for training parquet')
    parser.add_argument('--val_output', default='data/rushhour_val.parquet',  
                       help='Output path for validation parquet')
    parser.add_argument('--puzzle_start', type=int, default=151,
                       help='Start puzzle ID to process')
    parser.add_argument('--puzzle_end', type=int, default=3150,
                       help='End puzzle ID to process')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (0.2 = 20%)')
    
    args = parser.parse_args()
    
    # Convert puzzles
    train_path, val_path = convert_puzzles_to_parquet(
        data_dir=args.data_dir,
        train_output=args.train_output, 
        val_output=args.val_output,
        puzzle_range=(args.puzzle_start, args.puzzle_end),
        val_split_ratio=args.val_split
    )
    
    # Verify results
    verify_parquet_files(train_path, val_path)
    
    print(f"\n🎯 Ready for veRL training!")
    print(f"   Training: {train_path}")
    print(f"   Validation: {val_path}")