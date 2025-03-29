# HW 5

For this homework, create a sigmoid classifier for the data sets in the data dir by filling in the TODOs in the classify_block.py file.
The data classifies some ARC shapes as an "L block" or not. Later we will plug this classifier (or better versions of it) into the planner we created in HW 4.

Create it so that you get at least the following accuracy results for the datasets:

- data_0.json: 100% train, 100% test
- data_1.json: 70% train, 50% test
- data_2.json: 100% train, 100% test

# Grading
The points are as usual and out of 3pts total.

- 3 Points
  - 1 points for code
  - 1 point for README being accurate (see bottom) with your answers
  - 1 point for being functionally correct

# Bonus
+6 points for creating a data_3.json in the data dir with your own shape category and getting at least 60% on train and test using sigmoid classifier.
The grid size can be larger.

# <--- Your readme here --->
## Instructions for running your code on the data sets here
## Usage

1. Set the JSON filename in `classify_block.py`:
   ```python
   filename = 'data_0.json'
   ```

2. Run the script:
   ```bash
   python classify_block.py
   ```

Outputs predictions, actuals, and accuracy.
```

### For Train data
(1) What percent accuracy on train data do you get for datasets 0, 1, 2, and your bonus (if you do it)?

data_0.json: 
| Learning Rate | Iterations | Train Accuracy | Test Accuracy |
|---------------|------------|----------------|---------------|
| 0.5           | 1          | 0.67           | 1.00          |
| 0.5           | 10         | 1.00           | 1.00          |
| 0.5           | 100        | 1.00           | 1.00          |
| 0.1           | 1          | 0.50           | 1.00          |
| 0.1           | 10         | 1.00           | 1.00          |
| 0.1           | 100        | 1.00           | 1.00          |
| 0.01          | 1          | 0.83           | 0.50          |
| 0.01          | 10         | 1.00           | 1.00          |
| 0.01          | 100        | 1.00           | 1.00          |

data_1.json:
| Learning Rate | Iterations | Train Accuracy | Test Accuracy |
|---------------|------------|----------------|---------------|
| 0.5           | 1          | 0.64           | 0.50          |
| 0.5           | 10         | 0.71           | 0.50          |
| 0.5           | 100        | 1.00           | 1.00          |
| 0.1           | 1          | 0.43           | 0.50          |
| 0.1           | 10         | 0.79           | 0.50          |
| 0.1           | 100        | 1.00           | 1.00          |
| 0.01          | 1          | 0.64           | 0.50          |
| 0.01          | 10         | 0.57           | 0.00          |
| 0.01          | 100        | 1.00           | 1.00          |

data_2.json:
| Learning Rate | Iterations | Train Accuracy | Test Accuracy |
|---------------|------------|----------------|---------------|
| 0.5           | 1          | 0.57           | 0.50          |
| 0.5           | 10         | 0.74           | 0.50          |
| 0.5           | 100        | 1.00           | 1.00          |
| 0.1           | 1          | 0.43           | 0.00          |
| 0.1           | 10         | 0.52           | 0.50          |
| 0.1           | 100        | 1.00           | 1.00          |
| 0.01          | 1          | 0.74           | 0.50          |
| 0.01          | 10         | 0.70           | 0.50          |
| 0.01          | 100        | 1.00           | 1.00          |

data_3.json:
| Learning Rate | Iterations | Train Accuracy | Test Accuracy |
|---------------|------------|----------------|---------------|
| 0.5           | 1          | 0.40           | 0.50          |
| 0.5           | 10         | 1.00           | 1.00          |
| 0.5           | 100        | 1.00           | 1.00          |
| 0.1           | 1          | 0.80           | 0.50          |
| 0.1           | 10         | 1.00           | 1.00          |
| 0.1           | 100        | 1.00           | 1.00          |
| 0.01          | 1          | 0.60           | 0.00          |
| 0.01          | 10         | 1.00           | 1.00          |
| 0.01          | 100        | 1.00           | 1.00          |


(2) What kinds of training examples does the classifier get right/wrong for each dataset?
It seems that the classifier performs better when output = 1, ie. when the grid has an L-shaped pattern. Possibly due to the sparseness of the grid when output = 0 making it difficult for the model to learn any pattern. The model may also learn to associate any non-zero value with a positive output on shorter training runs. 

(3) Plot the sigmoid curve learned for each data set.
Plots in plots/

(4) What iterations and learning rate worked best?
In general - more iterations and lower learning rate performed better.

(5) What observations can you make about the data and the difference in accuracy scores?
The model performs well on structured patterns like bars or "N" shapes but struggles with sparse, isolated non-zero values in negative examples. Higher learning rates and more iterations help it learn the correct patterns.

### For Test data
(1) What percent accuracy on test data do you get for datasets 0, 1, 2, and your bonus (if you do it)?

Tables above.

(2) What kinds of test examples does the classifier get right/wrong for each dataset?
As with the training set, it seems that the classifier performs better when output = 1, ie. when the grid has an L-shaped pattern. Possibly due to the sparseness of the grid when output = 0 making it difficult for the model to learn any pattern. The model may also learn to associate any non-zero value with a positive output on shorter training runs. 
