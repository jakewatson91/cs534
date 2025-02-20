import numpy as np
import json

loss_values = []


def goal_test(all_actions, input_example: np.array, output_example: np.array):
    """
    Are we at the goal?
    :param all_actions:
    :param input_example:
    :param output_example:
    :return:
    """
    generated = generate_output(input_example, all_actions)
    pass

def goal_distance(all_actions, input_example: np.array, output_example: np.array):
    #TODO add heuristic for how close to goal you are
    pass

def generate_output(input_example, actions):
    """
    Given an input and our current mapping, what output would it generate?
    :param input_example:
    :param actions:
    :return:
    """
    generated = np.full(input_example.shape, fill_value=-1, dtype=np.int8)
    for action in actions:
        # for action in all_actions:
        # where we need to color
        output_x = action[1][0]  # get x cor of output
        output_y = action[1][1]
        input_x = action[0][0]
        input_y = action[0][1]
        color = input_example[input_x][input_y]
        generated[output_x][output_y] = color
    return generated


def get_actions(input_example: np.array, output_example: np.array, current_actions):
    """
    Get the next batch of possible mappings
    :param input_example:
    :param output_example:
    :param current_actions:
    :return:
    """
    actions = simple_correspondence(input_example, output_example)
    pass


def simple_correspondence(input_example: np.array, output_example: np.array):
    pass


def generate_child(action, current_actions):
    # used in greedy_search
        pass


def greedy_search(input_example: np.array, output_example: np.array, goal_test, frontier, explored):
    # TODO implement BFS but now using a frontier that picks child based on distance to goal using goal distance func

    """
        Once the function is found return the function itself as defined below

    :return: an array of tuples of coordinate mappings like:
                [( (1, (0,0)), (1, (0,1)) ), ..., ( (0, (1,0)), (0, (1,1)) ), ...]
                where the first element of each tuple contains the value and its coordinate (also a tuple) from the input
                and the second element of each tuple contains the value and its coordinate to the output.


                For example, if we have input:

                [[3, 3, 3],
                  [0, 0, 0],
                  [0, 0, 0]]

                  Then the first tuple of a function like [(3, (0, 0)), (3, (1, 1)),...]

                  means take the 3 and coordinates 0, 0 in the input and map it to value 3 at 1, 1 in the output

                  The output  after this mapping would be:
                  [[-, -, -],
                  [-, 3, -],
                  [-, -, -]] where "-" just means we have not set a value for that cell yet in this example

    """
    # .....
    # We can search for a function first ignoring colors
    # then we can just add the colors back at the end
    function_with_colors = add_colors(found_function)
    pass


def get_example_elements(json_example):
    """
    Load the data
    :param json_example:
    :return:
    """
    train = json_example["train"]

    pairs = []
    for pair in train:
        input = pair["input"]
        output = pair["output"]

        example = {}
        example["input"] = np.array(input)
        example["output"] = np.array(output)
        pairs.append(example)
    return pairs


def add_colors(found_function):
    pass


# ..........................................................................
# .............. Entry code to run Greedy Best First Search  ...............
# ..........................................................................

#TODO: Note we want to point this to any json file
with open('data/data_0.json', 'r') as file:
    train_data = json.load(file)

    examples = get_example_elements(train_data)
    passed = 0
    for example in examples:
        input = example["input"]
        output = example["output"]

        start_state_actions = set()
        frontier = None # TODO what data structure should we use?
        #TODO add the start actions to the frontier
        explored = []
        found_function = greedy_search(input, output, goal_test, frontier, explored)

        if found_function is not None:
            print("--=== Trained Model found ===--")
            # Add colors
            function_with_colors = add_colors(found_function)
            for mapping in function_with_colors:
                print(mapping)
        else:
            print("--=== Trained Model NOT found ===--")
        break
