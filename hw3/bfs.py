import numpy as np
import json


def goal_test(actions, input_example, output_example):
    """
    Check if the generated output matches the expected output.
    """
    generated = generate_output(input_example, actions)

    return np.array_equal(generated, output_example)


def goal_distance(actions, input_example, output_example):
    """
    Compute heuristic distance: count mismatched elements.
    """
    generated = generate_output(input_example, actions)
    distance = np.sum(generated != output_example)

    return distance

def generate_output(input_example, actions):
    """
    Apply the set of actions to generate an output.
    """
    generated = np.full(input_example.shape, fill_value=-1, dtype=np.int8)  # Initialize empty grid

    for action in actions:
        val, (old_x, old_y), new_val, (new_x, new_y) = action
        generated[new_x, new_y] = input_example[old_x, old_y]
    return generated


def get_actions(grid, new_grid):
    """
    Find all valid input-output mappings.
    """
    actions = set()
    rows, cols = grid.shape
    visited_inputs = set()
    visited_outputs = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited_inputs:  # Skip if already mapped
                continue
            val = grid[r, c]
            found = False
            for r2 in range(rows):
                for c2 in range(cols):
                    if new_grid[r2, c2] == val and (r2, c2) not in visited_outputs:
                        actions.add((val, (r, c), val, (r2, c2)))  
                        visited_inputs.add((r, c))
                        visited_outputs.add((r2, c2))
                        found = True
                        break  

                if found:
                    break  
    return actions


def greedy_search(input_example, output_example):
    """
    Implements Greedy Best-First Search (GBFS) using a sorted list instead of heapq.
    """
    frontier = [(goal_distance(set(), input_example, output_example), set())]  # Priority queue as a sorted list
    explored = set()
    iterations = 0

    while frontier:
        frontier.sort(key=lambda x: x[0])  # Sort by lowest goal distance
        distance, actions = frontier.pop(0)  

        if goal_test(actions, input_example, output_example):
            return actions, iterations  # Return the correct mapping function

        explored.add(tuple(actions))

        # Generate new possible actions
        new_actions = get_actions(input_example, output_example)
        for action in new_actions:
            child = set(actions)
            child.add(action)
            if tuple(child) not in explored:
                new_distance = goal_distance(child, input_example, output_example)
                frontier.append((new_distance, child))  # Add new state
        iterations += 1
    return None, None 


def get_example_elements(json_example):
    """
    Load training examples from JSON.
    """
    train = json_example["train"]
    pairs = []

    for pair in train:
        example = {
            "input": np.array(pair["input"]),
            "output": np.array(pair["output"])
        }
        pairs.append(example)

    return pairs


files = ['data/data_0.json', 'data/data_1.json', 'data/data_2.json', 'data/data_3.json']
for file in files:
    with open(file, 'r') as f:
        train_data = json.load(f)
        examples = get_example_elements(train_data)

        for example in examples:
            input_grid = example["input"]
            output_grid = example["output"]

            print("\nInput grid: ")
            print(input_grid)
            print("\nOutput grid: ")
            print(output_grid)

            found_function, iterations = greedy_search(input_grid, output_grid)

            if found_function:
                print(f"\n--=== Trained Model Found in {iterations} iterations ===--")
                for mapping in found_function:
                    print(mapping)
            else:
                print("\n--=== No Model Found ===--")