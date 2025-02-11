import numpy as np
from collections import deque
import json
from itertools import permutations

def parse_files(file):
    with open(file, 'r') as f:
        data = json.load(f)
        # inputs = [pair['input'] for pair in data['train']]
        # outputs = [pair['output'] for pair in data['train']]
        train = data['train']
        test = data['test']
    return train, test

def bfs_all_combinations(grid, target):
    q = deque([grid]) # create queue with original grid to test each transformation
    visited = set()
    # print("Q: ", q)
    while q:
        frontier = q.popleft()
        print("Frontier: ", frontier)
        mappings = generate_mapping(frontier, target)
        print("Mappings: ", mappings)
        generated = generate_output(frontier, mappings)
        print("Generated: ", generated)
        if np.array_equal(generated, target):
            print("Match found: ", mappings)
            return mappings
        generated = tuple(map(tuple, generated))
        if generated not in visited:
            q.append(generated)
            print("Q: ", q)
            visited.add(generated)
            # print(visited)

    print("No match found.")

# node = [[3, 3, 3],
#   [0, 0, 0],
#   [0, 0, 0]]

# target = [[0, 0, 0],
#   [3, 3, 3],
#   [0, 0, 0]]

node = [[2, 2, 2],
  [0, 0, 0],
  [0, 0, 0]]

target = [[4, 4, 4],
  [0, 0, 0],
  [0, 0, 0]]

def generate_mapping(grid, new_grid):
    res = []
    visited = set()
    mapped = set()

    rows, cols = len(grid), len(grid[0])
    for row in range(rows):
        for col in range(cols):
            val = grid[row][col]
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in visited and (row, col) not in mapped and val == target[i][j]:
                        res.append((val, (row, col), val, (i, j))) # (original val, (coordinates), original val, (new coordinates))
                        visited.add((i, j))
                        mapped.add((row, col))
                        # print("res: ", res)
                        break # move to the next one
    return res

def generate_output(frontier, actions):
    """
    Given an input and our current mapping, what output would it generate?
    :param input_example:
    :param actions:
    :return:
    """
    rows, cols = len(frontier), len(frontier[0])
    generated = np.full((rows, cols), fill_value=-1, dtype=np.int8)
    # print(generated)

    for action in actions:
        val_in, (x_in, y_in), val_out, (x_out, y_out) = action
        # where we need to color
        generated[x_out][y_out] = val_out
    # print(generated)
    return generated

# def h_flip(grid):
#     new_grid = grid[::-1]
#     return new_grid

# def v_flip(grid):
#     new_grid = [row[::-1] for row in grid[::-1]]
#     return new_grid

# def d_flip(grid):
#     new_grid = [list(row) for row in zip(*grid)]
#     return new_grid

# def rotate(grid): 
#     new_grid = [list(row) for row in zip(*(grid[::-1]))]
#     return new_grid

# def v_shift(grid):
#     new_grid = [grid[i-1] for i in range(len(grid))]
#     return new_grid

# def h_shift(grid):
#     transpose = d_flip(grid)
#     shift = v_shift(transpose)
#     new_grid = d_flip(shift)
#     return new_grid

# def scale(grid):
#     new_grid = [list(row[:]) for row in grid]
#     rows = len(grid)
#     cols = len(grid[0])
#     for row in range(rows):
#         for col in range(cols):
#             if grid[row][col] >= 9:
#                 new_grid[row][col] = 0
#             else:
#                 new_grid[row][col] *= 2
#     return new_grid

if __name__ in "__main__":

    files = ['data/hw_2_data_0.json', 'data/hw_2_data_1.json', 'data/hw_2_data_2.json', 'data/hw_2_data_3.json']
    for file in files:
        train, test = parse_files(file)
        for pair in zip(train):
            grid = pair[0]['input']
            print("Grid: ", grid)
            target = pair[0]['output']
            print("Target: ", target)

            bfs_all_combinations(grid, target)