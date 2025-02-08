import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# formating
matplotlib.rc('xtick', labelsize=5)
matplotlib.rc('ytick', labelsize=5)

def visualize_explore(examples, title=""):
    """
    :param examples: list of all pairs of arc examples. That is a list of dictionaries
        where each element is a dictionary with keys 'input' and 'output' and double array values representing
        an input and output pair int eh arc dataset
        [{'input'[[], ..., []], 'output:[[], ...[]]}, ..., {'input'[[], ..., []], 'output:[[], ...[]]}]
    :param title:
    :return: nothing is returned

    This method should save 400 plots in a plots directory. The plots should look
    like the example plot in the README file
    """
    fig, ax = plt.subplots(len(examples), ncols=2, figsize=(10, 6))
    fig.tight_layout(pad=3.0)
    index = 0
    # print("Examples length: ", len(examples))
    # print("Examples: ", examples)

    if len(examples) == 1:
        ax = [ax]

    for idx, example in enumerate(examples):
        inp = example["input"]
        # print(inp)
        # print(inp.shape)
        output = example["output"]

        # Color each of the cells in the input matrix using matshow
        ax[idx][0].matshow(inp, cmap='tab20c', vmin=0, vmax=8)

        shape = inp.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                c = inp[i, j]
                ax[idx][0].text(j, i, str(c), ha='center', va='center')
                ax[idx][0].axis('off')
                # Add number in each cell using text method

        # Color each of the cells in the output matrix using matshow
        ax[idx][1].matshow(output, cmap='tab20c', vmin=0, vmax=8)

        shape = output.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                c = output[i, j]
                ax[idx][1].text(j, i, str(c), ha='center', va='center')
                ax[idx][0].axis('off')
                # Add number in each cell using text method

        index += 1

    fig.suptitle(title)
    fig.savefig(f'./plots/{title}', bbox_inches='tight')
    plt.close()
    # Save plot to ./plots/<title> , where <title> is title passed to method