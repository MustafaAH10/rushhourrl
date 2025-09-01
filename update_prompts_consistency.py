#!/usr/bin/env python3
"""
Rush Hour Prompt Consistency Updater
Updates puzzle prompts to match the consistent system prompt format
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# System prompt template
SYSTEM_PROMPT = """You are an expert puzzle solver specializing in Rush Hour puzzles. Your task is to find the optimal sequence of moves to move the car 'C' to the TARGET position.

Key Instructions:
1. A 1-indexed coordinate system is being used
2. Each piece (car or blocker) can only move UP, DOWN, LEFT, or RIGHT by exactly one square
3. For 1x1 pieces (car C and blockers B1, B2, etc.): Use single coordinate format
4. For 2x1 pieces (blockers H1, H2, etc.): Use double coordinate format with both occupied positions
5. Pieces CANNOT move outside the 4x4 grid or into occupied squares at any instant
6. Provide your solution in the exact format requested

Be precise with coordinates and piece movements. Think logically about the sequence of moves needed."""

def format_1x1_blockers(pieces: Dict) -> List[str]:
    """Format 1x1 blockers for display"""
    blockers = []
    for piece_id, piece_data in pieces.items():
        if piece_data['type'] == '1x1_blocker':
            pos = piece_data['position']
            blockers.append(f"{piece_id} at [{pos[0]},{pos[1]}]")
    return sorted(blockers)

def format_2x1_blockers(pieces: Dict) -> List[str]:
    """Format 2x1 blockers for display"""
    blockers = []
    for piece_id, piece_data in pieces.items():
        if '2x1' in piece_data['type']:
            positions = piece_data['positions']
            orientation = piece_data['type'].split('_')[2]  # horizontal or vertical
            pos_str = f"({positions[0][0]},{positions[0][1]}), ({positions[1][0]},{positions[1][1]})"
            blockers.append(f"{piece_id} ({orientation}) at {pos_str}")
    return sorted(blockers)

def create_new_prompt(puzzle_state: Dict) -> str:
    """Create new consistent prompt format"""
    
    car_position = puzzle_state['car_position']
    exit_position = puzzle_state['exit_position']
    pieces = puzzle_state['pieces']
    grid = puzzle_state['grid']
    
    # Format grid as JSON string
    grid_json = json.dumps(grid, separators=(',', ':'))
    
    # Get blocker lists
    blockers_1x1 = format_1x1_blockers(pieces)
    blockers_2x1 = format_2x1_blockers(pieces)
    
    # Format blocker strings
    blockers_1x1_str = '\n'.join(f"  - {blocker}" for blocker in blockers_1x1) if blockers_1x1 else "  - None present"
    blockers_2x1_str = '\n'.join(f"  - {blocker}" for blocker in blockers_2x1) if blockers_2x1 else "  - None present"
    
    # Create the complete prompt (system + puzzle-specific)
    new_prompt = f"""{SYSTEM_PROMPT}

Task: Solve this 4x4 Rush Hour puzzle - move car "C" from position [{car_position[0]},{car_position[1]}] to the TARGET at position [{exit_position[0]},{exit_position[1]}] given the position of the blockers below.

Current Grid State (JSON format):
{grid_json}

Current Pieces:
- Car "C": Position [{car_position[0]},{car_position[1]}]
- 1x1 Blockers (B1, B2, etc.): Single-cell obstacles that can be moved to clear a path
{blockers_1x1_str}
- 2x1 Blockers (H1, H2, etc.): Two-cell obstacles that move as a single unit
{blockers_2x1_str}
- TARGET: Position [{exit_position[0]},{exit_position[1]}]

Movement Rules:
- Any piece (car "C", 1x1 blockers "B1, B2, etc.", or 2x1 blockers "H1, H2, etc.") can move UP, DOWN, LEFT, or RIGHT
- Each move is exactly ONE square in any direction for the entire piece
- For 2x1 blockers: The entire piece moves together as a unit (both cells move simultaneously)
- Pieces strictly CANNOT move outside the 4x4 grid
- Pieces strictly CANNOT move into occupied squares (i.e. squares that already have another piece)
- At ANY instant, there CANNOT be two pieces occupying the same square
- The same piece can move multiple times in a row if needed
- You win when car "C" reaches the TARGET cell

Coordinate System:
- Use [row,col] format where [1,1] is top-left, [4,4] is bottom-right
- Each cell shows its coordinates in black text: (row,col)
- For 2x1 blockers, both occupied cells are shown in the piece description

Expected Output Format:
Wrap your solution in <solution> tags and provide it as a numbered list of moves in this exact format:

<solution>
Step 1: [PIECE] [start_position] -> [end_position]
Step 2: [PIECE] [start_position] -> [end_position]
...
</solution>

For 1x1 pieces (car "C" and blockers "B1", "B2", etc.):
- Use single coordinate: C [2,1] -> [2,2]

For 2x1 pieces (blockers "H1", "H2", etc.):
- List both coordinates: H1 [[1,1],[1,2]] -> [[2,1],[2,2]]

Example response format:
<solution>
Step 1: B2 [3,2] -> [4,2]
Step 2: H1 [(2,3), (3,3)] -> [(1,3), (2,3)]
Step 3: B2 [2,4] -> [1,4]
Step 4: C [3,1] -> [3,2]
Step 5: C [3,2] -> [3,3]
Step 6: C [3,3] -> [3,4]
Step 7: C [3,4] -> [2,4]
</solution>"""
    
    return new_prompt

def update_puzzle_prompt(puzzle_folder: Path) -> bool:
    """Update a single puzzle's prompt to new format"""
    
    try:
        # Load puzzle state
        state_file = puzzle_folder / "puzzle_state.json"
        if not state_file.exists():
            print(f"❌ No puzzle_state.json found in {puzzle_folder}")
            return False
            
        with open(state_file, 'r') as f:
            puzzle_state = json.load(f)
        
        # Create new prompt
        new_prompt = create_new_prompt(puzzle_state)
        
        # Update puzzle_state.json with new prompt
        puzzle_state['prompt'] = new_prompt
        
        with open(state_file, 'w') as f:
            json.dump(puzzle_state, f, indent=2)
        
        # Update prompt.txt file
        prompt_file = puzzle_folder / "prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(new_prompt)
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating {puzzle_folder}: {e}")
        return False

def update_all_prompts(
    data_dir: str = "data/rl-data",
    puzzle_range: Tuple[int, int] = None,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Update all puzzle prompts to consistent format
    
    Args:
        data_dir: Directory containing puzzle folders
        puzzle_range: Tuple of (start, end) puzzle IDs to update. None for all
        dry_run: If True, only show what would be updated without making changes
    
    Returns:
        Dictionary with update statistics
    """
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return {"error": 1}
    
    # Get all puzzle folders
    puzzle_folders = sorted([d for d in data_path.iterdir() 
                           if d.is_dir() and d.name.startswith('puzzle')])
    
    # Filter by range if specified
    if puzzle_range:
        start_id, end_id = puzzle_range
        puzzle_folders = [d for d in puzzle_folders 
                         if start_id <= int(d.name.replace('puzzle', '')) <= end_id]
    
    print(f"{'🔍' if dry_run else '🔄'} Processing {len(puzzle_folders)} puzzles...")
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
    
    stats = {"success": 0, "error": 0, "total": len(puzzle_folders)}
    
    for i, puzzle_folder in enumerate(puzzle_folders, 1):
        puzzle_id = puzzle_folder.name
        
        if dry_run:
            print(f"[{i}/{len(puzzle_folders)}] Would update {puzzle_id}")
            stats["success"] += 1
        else:
            print(f"[{i}/{len(puzzle_folders)}] Updating {puzzle_id}...", end=" ")
            
            if update_puzzle_prompt(puzzle_folder):
                print("✅")
                stats["success"] += 1
            else:
                print("❌")
                stats["error"] += 1
    
    return stats

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Update Rush Hour puzzle prompts for consistency')
    parser.add_argument('--data_dir', default='data/rl-data',
                       help='Directory containing puzzle folders')
    parser.add_argument('--puzzle_start', type=int,
                       help='Start puzzle ID (optional)')
    parser.add_argument('--puzzle_end', type=int,
                       help='End puzzle ID (optional)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be updated without making changes')
    
    args = parser.parse_args()
    
    # Set puzzle range if specified
    puzzle_range = None
    if args.puzzle_start is not None and args.puzzle_end is not None:
        puzzle_range = (args.puzzle_start, args.puzzle_end)
    
    # Update prompts
    stats = update_all_prompts(
        data_dir=args.data_dir,
        puzzle_range=puzzle_range,
        dry_run=args.dry_run
    )
    
    # Print results
    print(f"\n📊 Update Results:")
    print(f"   Total puzzles: {stats['total']}")
    print(f"   ✅ Successfully updated: {stats['success']}")
    print(f"   ❌ Errors: {stats['error']}")
    
    if not args.dry_run and stats['error'] == 0:
        print(f"\n🎯 All prompts updated successfully!")
        print(f"   Ready to run data conversion: python rush_hour_data_converter.py")

if __name__ == "__main__":
    main()