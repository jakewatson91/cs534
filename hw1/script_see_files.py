import time
from data_loader import walk_files, get_example_elements
from explore_data import visualize_explore

# Replace 'your_directory_path' with the path of the directory you want to walk
# training_json = walk_files('./data/ARC-AGI/data/training')
# evaluation_json = walk_files('./data/ARC-AGI/data/evaluation')
count = 0
# get the data in a usable format
json_data = walk_files('./data/ARC-AGI/data/')
for example in json_data:
    print(f"example_{count}")
    if example['train']:
        examples = get_example_elements(example['train'])
        visualize_explore(examples, f"example_train_{count}.png")
        time.sleep(2)

    # TODO get this to work
    if example['test']:
        examples = get_example_elements(example['test'])
        visualize_explore(examples, f"example_test_{count}")
        time.sleep(2)
    count += 1
