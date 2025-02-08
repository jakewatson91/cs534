from collections import deque

visited = []  # Stores visited states
shortest_path = []  # Keeps the shortest path found

def bfs(node, output):
    """
    TODO
    Use each train example in the data dir (i.e., json files) to search for a function that
    can take each input and produce the corresponding output. A complete function itself
    is defined as F(value V at a coordinate X) -> value Z at coordinate Y for all values in input and output

    ***Bonus***: hw_2_data_3.json is bonus and if done will triple (4x) the score of your assignment!!
    ***Bonus***: Write another function that can output all functions that work - worth 2x the score on your assignment!

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

                  You want to return the function found by BFS.
                  We then can apply that function in the graders private test set.

    """
    
    queue = deque([(node, [])])  

    while queue:
        current_node, current_mappings = queue.popleft()

        if current_node == output:
            return current_mappings

        transformations = [rotate_clock, v_reflect, h_reflect]
        for trans in transformations:
            _, transformed_node = trans(current_node)

            new_mappings = current_mappings + generate_mappings(current_node, transformed_node)

            if transformed_node not in visited:
                visited.append(transformed_node)
                queue.append((transformed_node, new_mappings))

    return [new_mappings]  

def generate_mappings(node, transformed_node):
    mappings = []
    for i in range(len(node)):
        for j in range(len(node[i])):
            mappings.append(((node[i][j], (i, j)), (transformed_node[i][j], (i, j))))
    return mappings

def rotate_clock(node):
    new_node = [list(row) for row in zip(*node[::-1])]
    return (node, new_node)

def v_reflect(node):
    new_node = [row[::-1] for row in node]
    return (node, new_node)

def h_reflect(node):
    new_node = node[::-1]
    return (node, new_node)

node = [[3, 2],
        [0, 1]]

target = [[0, 1], 
          [3, 2]]

mappings = bfs(node, target)
print(mappings)