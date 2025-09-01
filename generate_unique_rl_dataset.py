#!/usr/bin/env python3
"""
Unique RL Dataset Generator for 4x4 Rush Hour Puzzles
Generates 3000 additional unique puzzles (1000 easy, 1000 medium, 1000 hard)
First creates a merged JSON with all unique puzzles, then creates folders
Ensures true uniqueness by block positions across ALL 3150 puzzles
"""

import random
import json
import os
import hashlib
from collections import deque
import csv

def grid_to_block_positions(grid):
    """Extract block positions as a set of coordinates, ignoring labels."""
    positions = set()
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell != '.':
                positions.add((i, j))
    return frozenset(positions)

class RushHour4x4:
    def __init__(self):
        self.grid_size = 4
        self.exit_pos = (1, 3)  # Exit at row 1 (0-indexed), rightmost column

    def get_difficulty_config(self, difficulty):
        """Get difficulty configuration for generating puzzles"""
        difficulty_configs = {
            "easy": [(4, 0), (3, 1), (5, 0)],
            "medium": [(2, 2), (3, 2), (1, 2), (4, 1)],
            "hard": [(0, 3), (1, 3), (2, 3), (0, 4)]
        }
        return random.choice(difficulty_configs[difficulty])

    def create_empty_grid(self):
        """Create empty 4x4 grid"""
        return [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
    
    def find_piece_positions(self, grid, piece_id):
        """Find all positions occupied by a piece"""
        positions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] == piece_id:
                    positions.append((r, c))
        return positions
    
    def can_place_piece(self, grid, positions):
        """Check if a piece can be placed at given positions"""
        for r, c in positions:
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                return False
            if grid[r][c] != '.':
                return False
        return True
    
    def place_piece(self, grid, piece_id, positions):
        """Place a piece on the grid"""
        for r, c in positions:
            grid[r][c] = piece_id
    
    def remove_piece(self, grid, piece_id):
        """Remove a piece from the grid"""
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] == piece_id:
                    grid[r][c] = '.'
    
    def get_all_empty_positions(self, grid):
        """Get all empty positions in the grid"""
        empty = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] == '.':
                    empty.append((r, c))
        return empty
    
    def generate_2x1_positions(self, start_pos, orientation):
        """Generate positions for a 2x1 blocker"""
        r, c = start_pos
        if orientation == 'horizontal':
            return [(r, c), (r, c + 1)]
        else:  # vertical
            return [(r, c), (r + 1, c)]
    
    def generate_puzzle(self, num_1x1_blockers=3, num_2x1_blockers=2, max_attempts=1000):
        """Generate a random 4x4 Rush Hour puzzle"""
        for _ in range(max_attempts):
            grid = self.create_empty_grid()
            
            # Place car (1x1) - not at exit position
            available_positions = [(r, c) for r in range(self.grid_size) 
                                   for c in range(self.grid_size) if (r, c) != self.exit_pos]
            car_pos = random.choice(available_positions)
            self.place_piece(grid, 'C', [car_pos])
            
            # Place 2x1 blockers first
            blockers_placed = 0
            for i in range(num_2x1_blockers):
                piece_id = f'H{i+1}'  # H for 2x1 pieces
                placed = False
                attempts_for_this_piece = 0
                
                while not placed and attempts_for_this_piece < 50:
                    orientation = random.choice(['horizontal', 'vertical'])
                    if orientation == 'horizontal':
                        valid_starts = [(r, c) for r in range(self.grid_size) 
                                        for c in range(self.grid_size - 1)]
                    else:  # vertical
                        valid_starts = [(r, c) for r in range(self.grid_size - 1) 
                                        for c in range(self.grid_size)]
                    
                    if valid_starts:
                        start_pos = random.choice(valid_starts)
                        positions = self.generate_2x1_positions(start_pos, orientation)
                        if self.can_place_piece(grid, positions):
                            self.place_piece(grid, piece_id, positions)
                            placed = True
                            blockers_placed += 1
                    attempts_for_this_piece += 1
            
            # Place 1x1 blockers
            for i in range(num_1x1_blockers):
                piece_id = f'B{i+1}'
                empty_positions = self.get_all_empty_positions(grid)
                if empty_positions:
                    pos = random.choice(empty_positions)
                    self.place_piece(grid, piece_id, [pos])
                    blockers_placed += 1
            
            # Check if we have enough blockers and the puzzle is solvable
            expected_total = num_2x1_blockers + num_1x1_blockers
            if blockers_placed >= max(1, expected_total - 1):  # Allow some flexibility
                solution = self.bfs_solve(grid)
                if solution is not None and len(solution) > 0:  # At least 1 move
                    return grid
        
        raise Exception(f"Could not generate solvable puzzle after {max_attempts} attempts")
    
    def grid_to_tuple(self, grid):
        """Convert grid to hashable tuple"""
        return tuple(tuple(row) for row in grid)
    
    def tuple_to_grid(self, t):
        """Convert tuple back to grid"""
        return [list(row) for row in t]
    
    def is_solved(self, grid):
        """Check if car 'C' is at exit position"""
        return grid[self.exit_pos[0]][self.exit_pos[1]] == 'C'
    
    def get_piece_info(self, grid, piece_id):
        """Get information about a piece (positions and type)"""
        positions = self.find_piece_positions(grid, piece_id)
        if len(positions) == 1:
            return positions, '1x1'
        elif len(positions) == 2:
            r1, c1 = positions[0]
            r2, c2 = positions[1]
            if r1 == r2:
                return sorted(positions, key=lambda x: x[1]), 'horizontal'
            else:
                return sorted(positions, key=lambda x: x[0]), 'vertical'
        return positions, 'unknown'
    
    def get_neighbors(self, state):
        """Get all valid neighboring states"""
        grid = self.tuple_to_grid(state)
        neighbors = []
        
        # Find all unique pieces
        pieces = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] != '.':
                    pieces.add(grid[r][c])
        
        # Try moving each piece in all 4 directions
        for piece_id in pieces:
            positions, _ = self.get_piece_info(grid, piece_id)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # up, down, left, right
                new_positions = [(r + dr, c + dc) for r, c in positions]
                
                # Bounds check
                valid_move = True
                for nr, nc in new_positions:
                    if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                        valid_move = False
                        break
                if not valid_move:
                    continue
                
                # Collision check
                temp_grid = [row[:] for row in grid]
                self.remove_piece(temp_grid, piece_id)
                can_move = True
                for nr, nc in new_positions:
                    if temp_grid[nr][nc] != '.':
                        can_move = False
                        break
                if not can_move:
                    continue
                
                # Apply move
                new_grid = [row[:] for row in temp_grid]
                self.place_piece(new_grid, piece_id, new_positions)
                neighbors.append(self.grid_to_tuple(new_grid))
        
        return neighbors
    
    def bfs_solve(self, start_grid):
        """BFS solver that returns ONE optimal solution"""
        start_state = self.grid_to_tuple(start_grid)
        if self.is_solved(start_grid):
            return []  # Already solved
        
        queue = deque([(start_state, [])])
        visited = {start_state}
        
        while queue:
            current_state, path = queue.popleft()
            
            for next_state in self.get_neighbors(current_state):
                if next_state in visited:
                    continue
                    
                visited.add(next_state)
                new_path = path + [next_state]
                
                if self.is_solved(self.tuple_to_grid(next_state)):
                    return new_path
                
                queue.append((next_state, new_path))
        
        return None

    def get_grid_hash(self, grid):
        """Generate a hash for a grid to check uniqueness"""
        grid_str = str(self.grid_to_tuple(grid))
        exit_str = str(self.exit_pos)
        combined_str = grid_str + exit_str
        return hashlib.sha256(combined_str.encode()).hexdigest()
    
    def generate_solution_moves(self, start_grid, solution_path):
        """Generate text description of moves"""
        moves = []
        current_grid = start_grid
        
        for i, next_state in enumerate(solution_path):
            next_grid = self.tuple_to_grid(next_state)
            moved_piece = None
            old_positions = []
            new_positions = []
            
            # Unique pieces in current grid
            current_pieces = set()
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if current_grid[r][c] != '.':
                        current_pieces.add(current_grid[r][c])
            
            for piece in current_pieces:
                old_pos = self.find_piece_positions(current_grid, piece)
                new_pos = self.find_piece_positions(next_grid, piece)
                if old_pos != new_pos:
                    moved_piece = piece
                    old_positions = old_pos
                    new_positions = new_pos
                    break
            
            if moved_piece:
                # Convert to 1-indexed for display
                old_display = [(r+1, c+1) for r, c in old_positions]
                new_display = [(r+1, c+1) for r, c in new_positions]
                
                if len(old_positions) == 1:
                    moves.append(f"Step {i+1}: {moved_piece} [{old_display[0][0]},{old_display[0][1]}] -> [{new_display[0][0]},{new_display[0][1]}]")
                else:
                    moves.append(f"Step {i+1}: {moved_piece} {old_display} -> {new_display}")
            
            current_grid = next_grid
        
        return moves

    def generate_puzzle_specific_prompt(self, grid):
        """Generate a prompt for this specific puzzle"""
        car_pos = self.find_piece_positions(grid, 'C')[0]
        car_pos_1indexed = (car_pos[0] + 1, car_pos[1] + 1)
        exit_pos_1indexed = (self.exit_pos[0] + 1, self.exit_pos[1] + 1)
        
        blockers_1x1 = []
        blockers_2x1 = []
        
        pieces = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = grid[r][c]
                if cell != '.' and cell != 'C' and cell not in pieces:
                    pieces.add(cell)
                    positions = self.find_piece_positions(grid, cell)
                    if len(positions) == 1:
                        pos_1indexed = (positions[0][0] + 1, positions[0][1] + 1)
                        blockers_1x1.append(f"{cell} at [{pos_1indexed[0]},{pos_1indexed[1]}]")
                    else:
                        pos_1indexed = [(pos[0] + 1, pos[1] + 1) for pos in positions]
                        positions_str = ", ".join([f"[{p[0]},{p[1]}]" for p in pos_1indexed])
                        if positions[0][0] == positions[1][0]:
                            orientation = "horizontal"
                        else:
                            orientation = "vertical"
                        blockers_2x1.append(f"{cell} ({orientation}) at {positions_str}")

        prompt = f"""Task: You have been given a 4x4 Rush Hour puzzle above which you need to solve. Move car "C" from position [{car_pos_1indexed[0]},{car_pos_1indexed[1]}] to the TARGET at position [{exit_pos_1indexed[0]},{exit_pos_1indexed[1]}].

Current Pieces:
- Car "C": Position [{car_pos_1indexed[0]},{car_pos_1indexed[1]}]
- 1x1 Blockers: {', '.join(blockers_1x1) if blockers_1x1 else 'None'}
- 2x1 Blockers: {', '.join(blockers_2x1) if blockers_2x1 else 'None'}
- TARGET: Position [{exit_pos_1indexed[0]},{exit_pos_1indexed[1]}]

Rules:
- Any piece can move UP, DOWN, LEFT, or RIGHT by exactly one square
- For 2x1 blockers: The entire piece moves together as a unit
- Pieces cannot move outside the 4x4 grid
- Pieces cannot move into occupied squares
- No two pieces can occupy the same square
- Goal: Move car "C" to the TARGET position

Coordinate System: [row,col] format where [1,1] is top-left, [4,4] is bottom-right

Provide your solution as:
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
Step 1: C [2,1] -> [2,2]
Step 2: H1 [[1,3],[1,4]] -> [[2,3],[2,4]]  
Step 3: B1 [3,2] -> [3,1]
Step 4: C [2,2] -> [2,4]
</solution>"""
        return prompt

    def generate_puzzle_json(self, grid, difficulty="unknown", num_1x1_blockers=0, num_2x1_blockers=0,
                           solution_length=0):
        """Generate JSON representation of puzzle state"""
        # Find car position
        car_pos = self.find_piece_positions(grid, 'C')[0]
        
        # Convert positions to 1-indexed for JSON
        car_pos_1indexed = [car_pos[0] + 1, car_pos[1] + 1]
        exit_pos_1indexed = [self.exit_pos[0] + 1, self.exit_pos[1] + 1]
        
        # Build pieces dictionary
        pieces = {}
        pieces['C'] = {"type": "car", "position": car_pos_1indexed}
        
        # Find all blockers
        processed_pieces = {'C'}
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = grid[r][c]
                if cell != '.' and cell not in processed_pieces:
                    positions = self.find_piece_positions(grid, cell)
                    if len(positions) == 1:
                        pos_1indexed = [positions[0][0] + 1, positions[0][1] + 1]
                        pieces[cell] = {"type": "1x1_blocker", "position": pos_1indexed}
                    else:
                        pos_1indexed = [[pos[0] + 1, pos[1] + 1] for pos in positions]
                        if positions[0][0] == positions[1][0]:
                            piece_type = "2x1_horizontal_blocker"
                        else:
                            piece_type = "2x1_vertical_blocker"
                        pieces[cell] = {"type": piece_type, "positions": pos_1indexed}
                    processed_pieces.add(cell)
        
        # Generate the prompt for this specific puzzle
        prompt = self.generate_puzzle_specific_prompt(grid)
        
        # Create JSON structure
        puzzle_json = {
            "grid": grid,
            "car_position": car_pos_1indexed,
            "exit_position": exit_pos_1indexed,
            "pieces": pieces,
            "puzzle_info": {
                "difficulty": difficulty,
                "num_1x1_blockers": num_1x1_blockers,
                "num_2x1_blockers": num_2x1_blockers,
                "total_moves_in_solution": solution_length,
                "grid_size": f"{self.grid_size}x{self.grid_size}",
                "coordinate_system": "1-indexed, [row,col] format where [1,1] is top-left"
            },
            "prompt": prompt
        }
        
        return puzzle_json

def generate_unique_merged_dataset(random_seed=12345):
    """
    Phase 1: Generate merged master configuration with all 3150 unique puzzles
    """
    print("🚀 Phase 1: Generating Merged Master Configuration")
    print("=" * 70)
    print("Target: 3150 total puzzles (150 existing + 3000 new)")
    print("New distribution: 1000 easy + 1000 medium + 1000 hard")
    print(f"Random seed: {random_seed}")
    print()
    
    random.seed(random_seed)
    game = RushHour4x4()
    
    # Load existing 150 puzzles
    existing_file = "/home/mustafaah/rushhoureval/data/4x4/master_puzzle_configs.json"
    with open(existing_file, 'r') as f:
        existing_puzzles = json.load(f)
    
    print(f"Loaded {len(existing_puzzles)} existing puzzles")
    
    # Track all unique position sets
    all_position_sets = set()
    merged_puzzles = []
    
    # Add existing puzzles to merged list and position tracking
    for i, puzzle in enumerate(existing_puzzles):
        positions = grid_to_block_positions(puzzle['grid'])
        all_position_sets.add(positions)
        
        # Convert to new format
        merged_puzzle = {
            'puzzle_id': i + 1,
            'source': 'original_4x4',
            'difficulty': puzzle['difficulty'],
            'grid': puzzle['grid'],
            'num_1x1_blockers': puzzle['num_1x1_blockers'],
            'num_2x1_blockers': puzzle['num_2x1_blockers'],
            'grid_hash': puzzle['grid_hash'],
            'solution_length': puzzle['solution_length'],
            'solution': puzzle['solution']
        }
        merged_puzzles.append(merged_puzzle)
    
    print(f"Existing puzzles: {len(all_position_sets)} unique positions")
    
    # Generate new puzzles by difficulty
    difficulties = [("easy", 1000), ("medium", 1000), ("hard", 1000)]
    next_puzzle_id = len(existing_puzzles) + 1
    
    for difficulty, target_count in difficulties:
        print(f"\n🎯 Generating {target_count} new {difficulty} puzzles...")
        generated_count = 0
        attempts = 0
        max_attempts = target_count * 50  # Allow many attempts
        
        while generated_count < target_count and attempts < max_attempts:
            attempts += 1
            
            try:
                # Get difficulty configuration
                num_1x1, num_2x1 = game.get_difficulty_config(difficulty)
                
                # Generate puzzle
                grid = game.generate_puzzle(num_1x1_blockers=num_1x1, num_2x1_blockers=num_2x1)
                
                # Check uniqueness by positions
                positions = grid_to_block_positions(grid)
                if positions in all_position_sets:
                    continue
                
                # Solve puzzle
                solution = game.bfs_solve(grid)
                if solution is None or len(solution) == 0:
                    continue
                
                # Accept this puzzle
                all_position_sets.add(positions)
                grid_hash = game.get_grid_hash(grid)
                
                # Create puzzle data
                puzzle_data = {
                    'puzzle_id': next_puzzle_id,
                    'source': 'new_rl',
                    'difficulty': difficulty,
                    'grid': grid,
                    'num_1x1_blockers': num_1x1,
                    'num_2x1_blockers': num_2x1,
                    'grid_hash': grid_hash,
                    'solution_length': len(solution),
                    'solution': solution
                }
                
                merged_puzzles.append(puzzle_data)
                generated_count += 1
                next_puzzle_id += 1
                
                if generated_count % 100 == 0:
                    print(f"  ✓ Generated {generated_count}/{target_count} {difficulty} puzzles... (attempts: {attempts})")
                    
            except Exception as e:
                continue
        
        if generated_count < target_count:
            print(f"  ⚠️ Warning: Only generated {generated_count}/{target_count} {difficulty} puzzles after {attempts} attempts")
        else:
            print(f"  ✅ Successfully generated {generated_count} {difficulty} puzzles")
    
    # Verify total uniqueness
    print(f"\n🔍 Verifying uniqueness...")
    final_position_sets = set()
    for puzzle in merged_puzzles:
        positions = grid_to_block_positions(puzzle['grid'])
        final_position_sets.add(positions)
    
    print(f"Total puzzles in merged list: {len(merged_puzzles)}")
    print(f"Unique position sets: {len(final_position_sets)}")
    print(f"All puzzles unique: {len(merged_puzzles) == len(final_position_sets)}")
    
    if len(merged_puzzles) != len(final_position_sets):
        print("❌ ERROR: Duplicates found in final dataset!")
        return None
    
    # Save merged master configuration
    os.makedirs("data/rl-data", exist_ok=True)
    merged_file = "data/rl-data/merged_master_configs.json"
    
    json_puzzles = []
    for puzzle in merged_puzzles:
        json_puzzle = puzzle.copy()
        json_puzzle['solution'] = [list(sol) for sol in puzzle['solution']]
        json_puzzles.append(json_puzzle)
    
    with open(merged_file, 'w') as f:
        json.dump(json_puzzles, f, indent=2)
    
    # Print summary by difficulty and source
    summary = {}
    for puzzle in merged_puzzles:
        key = (puzzle['source'], puzzle['difficulty'])
        summary[key] = summary.get(key, 0) + 1
    
    print(f"\n📊 Final Summary:")
    print("Source -> Difficulty -> Count")
    for (source, difficulty), count in summary.items():
        print(f"  {source} -> {difficulty}: {count}")
    
    total_original = sum(count for (source, diff), count in summary.items() if source == 'original_4x4')
    total_new = sum(count for (source, diff), count in summary.items() if source == 'new_rl')
    print(f"\nTotals: {total_original} original + {total_new} new = {len(merged_puzzles)} total")
    
    print(f"\n✅ Phase 1 Complete! Merged configuration saved to: {merged_file}")
    return merged_puzzles

def create_puzzle_folders_from_merged(merged_file="data/rl-data/merged_master_configs.json"):
    """
    Phase 2: Create individual puzzle folders from merged configuration
    """
    print("\n🚀 Phase 2: Creating Individual Puzzle Folders")
    print("=" * 70)
    
    # Load merged configuration
    with open(merged_file, 'r') as f:
        merged_puzzles = json.load(f)
    
    print(f"Loaded {len(merged_puzzles)} puzzles from merged configuration")
    
    game = RushHour4x4()
    output_dir = "data/rl-data"
    
    # Only create folders for new RL puzzles (skip original 4x4)
    rl_puzzles = [p for p in merged_puzzles if p['source'] == 'new_rl']
    print(f"Creating folders for {len(rl_puzzles)} new RL puzzles (skipping {len(merged_puzzles) - len(rl_puzzles)} original puzzles)")
    
    created_count = 0
    
    for puzzle in rl_puzzles:
        try:
            puzzle_id = puzzle['puzzle_id']
            grid = puzzle['grid']
            difficulty = puzzle['difficulty']
            num_1x1_blockers = puzzle['num_1x1_blockers']
            num_2x1_blockers = puzzle['num_2x1_blockers']
            solution = puzzle['solution']
            
            # Create puzzle folder
            puzzle_folder = os.path.join(output_dir, f"puzzle{puzzle_id}")
            os.makedirs(puzzle_folder, exist_ok=True)
            
            # Generate puzzle JSON
            puzzle_json = game.generate_puzzle_json(
                grid=grid,
                difficulty=difficulty,
                num_1x1_blockers=num_1x1_blockers,
                num_2x1_blockers=num_2x1_blockers,
                solution_length=len(solution)
            )
            
            # Save puzzle files
            with open(os.path.join(puzzle_folder, "puzzle_state.json"), 'w') as f:
                json.dump(puzzle_json, f, indent=2)
            
            with open(os.path.join(puzzle_folder, "prompt.txt"), 'w') as f:
                f.write(puzzle_json["prompt"])
            
            # Generate solution text
            moves = game.generate_solution_moves(grid, solution)
            with open(os.path.join(puzzle_folder, "solution.txt"), 'w') as f:
                f.write(f"Puzzle: puzzle{puzzle_id}\n")
                f.write(f"Puzzle ID: {puzzle_id}\n")
                f.write(f"Source: {puzzle['source']}\n")
                f.write(f"Total moves: {len(solution)}\n")
                f.write(f"Exit position: [2,4]\n")
                f.write(f"Difficulty: {difficulty} ({num_1x1_blockers} 1x1 blockers, {num_2x1_blockers} 2x1 blockers)\n")
                f.write("Coordinate system: [row,col] where [1,1] is top-left\n\n")
                f.write("SOLUTION:\n")
                f.write("=" * 40 + "\n")
                for move in moves:
                    f.write(move + "\n")
            
            created_count += 1
            
            if created_count % 200 == 0:
                print(f"  ✓ Created {created_count}/{len(rl_puzzles)} puzzle folders...")
                
        except Exception as e:
            print(f"Error creating folder for puzzle {puzzle.get('puzzle_id', 'unknown')}: {e}")
            continue
    
    # Generate summary CSV for RL puzzles only
    csv_file = os.path.join(output_dir, "rl_puzzle_catalog.csv")
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['puzzle_id', 'folder_name', 'source', 'difficulty', 'num_1x1_blockers', 
                     'num_2x1_blockers', 'solution_length', 'exit_position']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for puzzle in rl_puzzles:
            writer.writerow({
                'puzzle_id': puzzle['puzzle_id'],
                'folder_name': f"puzzle{puzzle['puzzle_id']}",
                'source': puzzle['source'],
                'difficulty': puzzle['difficulty'],
                'num_1x1_blockers': puzzle['num_1x1_blockers'],
                'num_2x1_blockers': puzzle['num_2x1_blockers'],
                'solution_length': puzzle['solution_length'],
                'exit_position': '[2,4]'
            })
    
    # Print final summary
    difficulty_counts = {}
    for puzzle in rl_puzzles:
        difficulty = puzzle['difficulty']
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    print(f"\n🎉 Phase 2 Complete!")
    print(f"Created folders for {created_count} RL puzzles:")
    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty.capitalize()}: {count}")
    print(f"RL puzzle catalog: {csv_file}")
    print(f"Merged master config: {merged_file}")
    
    return created_count

if __name__ == "__main__":
    # Phase 1: Generate merged master configuration
    merged_puzzles = generate_unique_merged_dataset(random_seed=12345)
    
    if merged_puzzles:
        # Phase 2: Create puzzle folders
        created_count = create_puzzle_folders_from_merged()
        print(f"\n✅ COMPLETE: Generated {len(merged_puzzles)} total unique puzzles")
        print(f"✅ COMPLETE: Created {created_count} new RL puzzle folders")
    else:
        print("❌ Generation failed due to uniqueness issues")