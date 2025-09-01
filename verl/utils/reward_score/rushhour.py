#!/usr/bin/env python3
"""
Rush Hour Reward Function for veRL Integration
Provides compute_score function for Rush Hour puzzle domain
"""

import json
import re
import os
import sys
from typing import List, Dict, Optional, Any

# Add current directory to path to import validator
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../..")
sys.path.insert(0, project_root)

try:
    from validator_4x4 import RushHour4x4Validator
except ImportError as e:
    print(f"Warning: Could not import validator_4x4: {e}")
    print(f"Please ensure validator_4x4.py is in project root: {project_root}")
    RushHour4x4Validator = None

def parse_solution_moves(response_text: str) -> List[str]:
    """Extract moves from model response text"""
    # Find <solution>...</solution> block
    solution_match = re.search(r'<solution>(.*?)</solution>', response_text, re.DOTALL | re.IGNORECASE)
    if not solution_match:
        return []
    
    solution_content = solution_match.group(1).strip()
    moves = []
    
    # Extract step lines
    for line in solution_content.split('\n'):
        line = line.strip()
        if line.startswith('Step '):
            moves.append(line)
    
    return moves

def validate_rushhour_solution(puzzle_state: dict, generated_moves: List[str]) -> dict:
    """Validate solution using existing validator"""
    if RushHour4x4Validator is None:
        return {
            'moves_are_legal': False,
            'reaches_target': False,
            'error': 'Validator not available'
        }
    
    try:
        validator = RushHour4x4Validator()
        target_pos = puzzle_state['exit_position']
        
        # Use existing validation method
        validation_result = validator.validate_solution(puzzle_state, generated_moves, target_pos)
        return validation_result
        
    except Exception as e:
        print(f"Validation error: {e}")
        return {
            'moves_are_legal': False,
            'reaches_target': False,
            'error': str(e)
        }

def compute_score(solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    Compute score for Rush Hour solution (veRL interface)
    
    Args:
        solution_str: Model's generated solution text
        ground_truth: Reference solution (not used in our implementation)
        extra_info: Dict containing puzzle_state, optimal_moves, etc.
    
    Returns:
        Float score value
    """
    
    # Handle missing extra_info
    if extra_info is None:
        return -10.0  # No puzzle information available
    
    # Parse moves from response
    moves = parse_solution_moves(solution_str)
    
    if not moves:
        return -10.0  # No valid solution format
    
    # Get puzzle information from extra_info
    puzzle_state = extra_info.get('puzzle_state', {})
    if isinstance(puzzle_state, str):
        try:
            puzzle_state = json.loads(puzzle_state)
        except:
            return -10.0  # Invalid puzzle state JSON
    
    optimal_moves = extra_info.get('optimal_moves', 999)
    
    # Validate solution using existing validator
    validation = validate_rushhour_solution(puzzle_state, moves)
    
    # Check for validation errors
    if 'error' in validation:
        return -5.0  # Validation error penalty
    
    # Penalty for illegal moves
    if not validation.get('moves_are_legal', False):
        return -10.0  # Illegal moves penalty
    
    # Penalty for not reaching target
    if not validation.get('reaches_target', False):
        return -2.0   # Legal but doesn't solve penalty
    
    # Success rewards based on optimality
    actual_length = len(moves)
    
    if actual_length == optimal_moves:
        return 10.0   # Perfect optimal solution
    elif actual_length <= optimal_moves * 1.2:  # Within 20% of optimal
        return 7.0    # Near-optimal solution
    elif actual_length <= optimal_moves * 1.5:  # Within 50% of optimal  
        return 4.0    # Reasonable solution
    else:
        return 1.0    # Correct but inefficient

# Test function
def test_compute_score():
    """Test the compute_score function with sample data"""
    
    # Sample puzzle state
    sample_puzzle_state = {
        "grid": [[".", ".", ".", "B1"], [".", ".", "B2", "."], [".", "H1", "H1", "."], [".", "C", ".", "B3"]],
        "car_position": [4, 2],
        "exit_position": [2, 4],
        "pieces": {
            "C": {"type": "car", "position": [4, 2]},
            "B1": {"type": "1x1_blocker", "position": [1, 4]},
            "B2": {"type": "1x1_blocker", "position": [2, 3]},
            "H1": {"type": "2x1_horizontal_blocker", "positions": [[3, 2], [3, 3]]},
            "B3": {"type": "1x1_blocker", "position": [4, 4]}
        },
        "puzzle_info": {
            "difficulty": "easy",
            "num_1x1_blockers": 3,
            "num_2x1_blockers": 1,
            "total_moves_in_solution": 5,
            "grid_size": "4x4"
        }
    }
    
    # Sample response with solution
    sample_response = """
    Looking at this puzzle, I need to move car C to position [2,4].
    
    <solution>
    Step 1: C [4,2] -> [4,3]
    Step 2: H1 [[3,2],[3,3]] -> [[3,1],[3,2]]
    Step 3: C [4,3] -> [3,3]
    Step 4: C [3,3] -> [3,4]
    Step 5: C [3,4] -> [2,4]
    </solution>
    """
    
    # Test the compute_score function
    extra_info = {
        'puzzle_state': json.dumps(sample_puzzle_state),
        'optimal_moves': 5,
        'difficulty': 'easy'
    }
    
    score = compute_score(sample_response, "", extra_info)
    print(f"Test score: {score}")
    
    return score

if __name__ == "__main__":
    print("Testing Rush Hour compute_score function...")
    test_score = test_compute_score()
    print(f"Test completed with score: {test_score}")