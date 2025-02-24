# Homework 3

## Grade
3 Points
1 points for code
1 point for README being accurate (see bottom)
1 point for being functionally correct

## Data
Datasets are in the data directory and same format


## Instructions
For this homework, please implement the genetic algorithm
and greedy best first search approach to find mapping functions (as we did in hw2).

To get started, please see the python files (genetic_algo.py and greed_best_first_search.py) and the TODOs in each.

For each algorithm approach, create a graph with "goal score" or "fitness function" on y-axis
and "iteration" or "generation" on x-axis. Which algorithm was faster?

Your graphs should be similar to this:
![example_plot.png](example_plot.png)
Save your graphs to the graphs directory

Commit your graphs and code to GitHub by the due date.

## Bonus
### 1
(+1 point to total )
For genetic algorithm, writeup an analysis of the impacts of what different
cross-overs have on how fast a child is found. Commit your writeup as Analysis.md
in your repo.

### 2
(+2 point to total )
Get both algorithms to work on data_bonus.json

## ADD your directions to run code here!!

Genetic Algorithm & Greedy Search for Grid Transformations

📌 Overview

This repository contains implementations of two algorithms for learning mappings between input and output grids:
	1.	Genetic Algorithm (GA) – Evolves a population of candidate solutions to learn transformation rules. The code for the genetic algorithm is in `genetic_algo.py`
	2.	Greedy Best-First Search (GBFS) – Uses a heuristic-driven approach to find an optimal mapping. The code for GBFS is in `bfs.py`

Both algorithms use JSON-formatted training and testing data, where each example consists of an input and output grid.

🛠 Installation

Ensure you have Python 3.7+ installed, then install dependencies:

pip install numpy matplotlib

📂 Data Format

Training and testing data are stored in JSON files (data/data_0.json, etc.). Each file follows this structure:

{
    "train": [
        {"input": [[...]], "output": [[...]]}
    ],
    "test": [
        {"input": [[...]], "output": [[...]]}
    ]
}

🚀 Running the Genetic Algorithm

To run the Genetic Algorithm on all dataset files:

python genetic_algo.py

🔹 How It Works:

✅ Loads training and test examples from JSON files.
✅ Evolves a population using selection, crossover, and mutation.
✅ Tests various crossover indices to analyze their impact on convergence.
✅ Tracks fitness loss over generations.
✅ Saves fitness loss vs. generations plots and crossover index vs. generations plots in the plots/ directory.

📊 Example Output:

Processing data/data_0.json...
No match found.
Processing data/data_1.json...
Match found after 350 generations!

Plots will be saved as:
📌 plots/genetic_algo_plot_file_0.png – Fitness loss vs. Generations
📌 plots/genetic_algo_plot_file_1.png – Fitness loss vs. Generations
📌 plots/crossover_tests.png – Crossover Index vs. Convergence Speed

🔍 Crossover Index Impact on Convergence
	•	Crossover points near the center generally result in faster convergence.
	•	Crossover points near the edges lead to slower convergence, likely due to less genetic diversity being introduced per generation.
	•	The optimal crossover point varies based on input complexity.

🔍 Running the Greedy Search Algorithm

To run the Greedy Best-First Search:

python greedy_search.py

🔹 How It Works:

✅ Uses heuristic search to map input grid elements to the output grid.
✅ Tracks the distance from the goal over iterations.
✅ Saves distance vs. total children explored plots in plots/.

📊 Example Output:

--=== Trained Model Found in 12 iterations with 45 total children ===--
(3, (0,0), 3, (1,1))
(2, (1,2), 2, (0,2))

Plots will be saved as:
📌 plots/bfs_plot_file_0.png – Distance vs. Total Children Explored
📌 plots/bfs_plot_file_1.png – Distance vs. Total Children Explored

⚙ Modifying Parameters

To adjust Genetic Algorithm settings, modify genetic_algo.py:

population_size = 100
mutation_rate = 0.1
generations = 10000

To modify the search heuristic, edit goal_distance() in greedy_search.py.

❓ Troubleshooting

⚠ Common Issues & Fixes

1️⃣ JSON File Not Found

📌 Ensure data/ contains valid JSON files.
📌 Run: ls data/ to check available files.

2️⃣ Plots Not Showing or Saving

📌 Ensure matplotlib is installed (pip install matplotlib).
📌 Try: plt.show() before plt.savefig().

3️⃣ Algorithm Doesn’t Converge

📌 Increase generations in genetic_algo.py.
📌 Tune mutation_rate (higher = more diversity).

Feel free to modify and experiment with different configurations! 🚀
