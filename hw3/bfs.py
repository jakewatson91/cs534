import numpy as np
import json
import matplotlib.pyplot as plt

def goal_test(actions, input_example, output_example):
    """
    Are we at the goal?
    :param all_actions:
    :param input_example:
    :param output_example:
    :return:
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
    Given an input and our current mapping, what output would it generate?
    :param input_example:
    :param actions:
    :return:
    """
    generated = np.full(input_example.shape, fill_value=-1, dtype=np.int8)  # Initialize empty grid

    for action in actions:
        val, (old_x, old_y), new_val, (new_x, new_y) = action
        generated[new_x, new_y] = input_example[old_x, old_y]
    return generated


def get_actions(grid, new_grid):
    """
    Get the next batch of possible mappings
    :param input_example:
    :param output_example:
    :param current_actions:
    :return:
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
                        actions.add((val, (r, c), val,(r2, c2)))  
                        visited_inputs.add((r, c))
                        visited_outputs.add((r2, c2))
                        found = True
                        break  # Stop after finding one valid match

                if found:
                    break  # Ensure only one match per input position

    return actions


def greedy_search(input_example, output_example):
    """
    Implements Greedy Best-First Search (GBFS) using a sorted list instead of heapq.
    """
    frontier = [(goal_distance(set(), input_example, output_example), set())]  # Priority queue as a sorted list
    explored = set()
    iterations = 0
    distances = []

    while frontier:
        frontier.sort(key=lambda x: x[0])  # Sort by lowest goal distance
        distance, actions = frontier.pop(0)  
        
        goal_reached = goal_test(actions, input_example, output_example)
        if goal_reached:
            return actions, iterations, len(distances), distances  # Return the correct mapping function

        explored.add(tuple(actions))

        # Generate new possible actions
        new_actions = get_actions(input_example, output_example)
        for action in new_actions:
            child = set(actions)
            child.add(action)
            if tuple(child) not in explored:
                new_distance = goal_distance(child, input_example, output_example)
                distances.append(new_distance)
                frontier.append((new_distance, child))  # Add new state
        iterations += 1

def get_example_elements(json_example):
    """
    Load training examples from JSON.
    """
    pairs = []

    for pair in json_example:
        example = {
            "input": np.array(pair["input"]),
            "output": np.array(pair["output"])
        }
        pairs.append(example)

    return pairs


files = ['data/data_0.json', 'data/data_1.json', 'data/data_2.json', 'data/data_3.json']
for i, file in enumerate(files):
    with open(file, 'r') as f:
        data = json.load(f)
        train = get_example_elements(data["train"])
        test = get_example_elements(data["test"])

        for example in test:
            input_grid = example["input"]
            output_grid = example["output"]

            print("/nInitial: ", input_grid)
            print("Final: ", output_grid)

            found_function, iterations, total_children, distances = greedy_search(input_grid, output_grid)

            if found_function:
                print(f"\n--=== Trained Model Found in {iterations} iterations with {total_children} total children ===--")
                for mapping in found_function:
                    print(mapping)
            else:
                print("\n--=== No Model Found ===--")
        
        plt.figure()
        x = range(len(distances))
        y = distances
        plt.plot(x, y)
    plt.legend()
    plt.title(f"{file}")
    plt.xlabel("total_children")
    plt.ylabel("Distance from goal")
    plt.savefig(f"plots/bfs_plot_file_{i}")


    


