import numpy as np
from collections import deque
import json
import copy

def parse_files(file):
    with open(file, 'r') as f:
        data = json.load(f)
        train = data['train']
        test = data['test']
    return train, test

def bfs_all_combinations(grid, target):
    iterations = 0
    q = deque([(grid, [])]) # create queue with original grid to test each transformation
    generated = np.full((len(grid), len(grid[0])), fill_value=-1, dtype=np.int8)
    generated = generated.tolist()
    visited = set()
    
    while q:
        print("Q: ", q)
        frontier, cur_mappings = q.popleft()
        next_mapping = generate_mapping(frontier, target)
        mappings = cur_mappings + [next_mapping]
        print(mappings)
        cur_generation = generate_output(frontier, next_mapping, generated)
        print(cur_generation)
        if np.array_equal(cur_generation, target):
            print(f"Match found after {iterations} iterations!")
            print("Length: ", len(mappings))
            return mappings
        generated_tuple = tuple(map(tuple, cur_generation))
        if generated_tuple not in visited: # needs to be tuple to add to set
            q.append((cur_generation, mappings))
            visited.add(generated_tuple)
        iterations += 1

    print("No match found.")

def generate_mapping(frontier, target):
    res = []
    visited = set()
    mapped = set()

    # compare each element in one grid with each element in the other until a match is found
    rows, cols = len(frontier), len(frontier[0])
    for row in range(rows):
        for col in range(cols):
            if (row, col) in mapped:
                continue
            val = frontier[row][col]
            new_val = val
            for i in range(rows):
                for j in range(cols):
                    if (i, j) not in visited and (row, col) not in mapped and new_val == target[i][j]:
                        visited.add((i, j)) # don't map same destination twice
                        mapped.add((row, col)) # don't map same source twice
                        # break # move to next col when mapped
                        return (val, (row, col), new_val, (i, j)) # (original val, (coordinates), original val, (new coordinates))

def generate_output(frontier, mapping, generated):
    """
    Given an input and our current mapping, what output would it generate?
    :param input_example:
    :param actions:
    :return:
    """
    generated = copy.deepcopy(generated)
    _, (_, _), new_val, (new_x, new_y) = mapping
    generated[new_x][new_y] = new_val
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
            print("Mappings: ", mappings)