import numpy as np
from collections import deque
import json

def parse_files(file):
    with open(file, 'r') as f:
        data = json.load(f)
        train = data['train']
        test = data['test']
    return train, test

def bfs_all_combinations(grid, target):
    q = deque([grid]) # create queue with original grid to test each transformation
    visited = set()
    scale = True
    while q:
        frontier = q.popleft()
        mappings = generate_mapping(frontier, target)
        generated = generate_output(frontier, mappings)
        print("Generated: ", generated)
        if np.array_equal(generated, target):
            print("Match found: ", mappings)
            return mappings
        generated = tuple(map(tuple, generated))
        if generated not in visited:
            q.append(generated)
            visited.add(generated)
        scale = False

    print("No match found.")

def generate_mapping(grid, scale=False):
    res = []
    visited = set()
    mapped = set()

    # compare each element in one grid with each element in the other until a match is found
    rows, cols = len(grid), len(grid[0])
    for row in range(rows):
        for col in range(cols):
            val = grid[row][col]
            if scale == True: # for bonus
                new_val = val*2 if val <= 9 else 0
            else:
                new_val = val
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in visited and (row, col) not in mapped and new_val == target[i][j]:
                        res.append((val, (row, col), new_val, (i, j))) # (original val, (coordinates), original val, (new coordinates))
                        visited.add((i, j)) # don't map same destination twice
                        mapped.add((row, col)) # don't map same source twice
                        break # move to next col when mapped
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

    for action in actions:
        val_in, (x_in, y_in), val_out, (x_out, y_out) = action
        # where we need to color
        generated[x_out][y_out] = val_out
    return generated

# translations

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

    files = ['data/data_0.json', 'data/data_1.json', 'data/data_2.json', 'data/data_3.json']
    for file in files:
        train, test = parse_files(file)
        for pair in zip(train):
            grid = pair[0]['input']
            # print("Grid: ", grid)
            target = pair[0]['output']
            # print("Target: ", target)

            print(f"Original grid: {grid}")
            print(f"Target grid: {target}")
            mappings = bfs_all_combinations(grid, target)