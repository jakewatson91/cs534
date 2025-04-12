# HW 6 Description
Create a neural net that can classify the simple example problems in the data dir by searching for it using a genetic algorithm.

The genetic algorithm needs to *at least* search over a space of parameters that includes nodes, 
batch size, learning rate, and regularization lambda hyperparameters.

These parameters can be encoded as a string and then interpreted to construct a neural net in Keras.

Use the neural net constructed and its accuracy in the fitness function to determine its score.
The fitness should be found by training the constructed neural net and the outputting its score on the test data.

After running your code, print out:
- The summary of one of the most optimal neural net "children" using model.summary() in Keras
- Accuracy of model

Please see the genetic_nerual_search.py file to get started.

# Grading
The points are as usual and out of 3pts total.

- 3 Points
  - 1 points for code
  - 1 point for README being accurate (see bottom) with your answers
  - 1 point for being functionally correct

# Bonus
+1 points for applying your approach on MNIST_train.csv (in data dir) to classify the images of a three as "3" or not
with accuracy above 75%.

# Your README HERE

Answer these questions:

- How to run the code?
- Accuracy of model found?
- Model summary?
- Hyperparameters of genetic algorithm used?
- Did you try to reduce the running time?
- What other kinds of neural net parameters did you try for your genetic algorithm to search over?

# Running the code
run the code with `python genetic_neural_search.py`
You can run specific files by commenting out the data or BONUS section, or updating the `files` data list

# Results
Hyperparams are [learning rate, regularization lambda, nodes, batch size]

data_0.json
	•	Accuracy: 100%
	•	Hyperparams: [0.001128, 0.001058, 64, 128]
	•	Params: 2,117
	•	Screenshot: data0.png

data_1.json
	•	Accuracy: 100%
	•	Hyperparams: [0.001259, 5.546e-5, 8, 64]
	•	Params: 269
	•	Screenshot: data1.png

data_2.json
	•	Accuracy: 50% (low generalization due to small test size)
	•	Hyperparams: [0.001558, 3.509e-5, 64, 32]
	•	Params: 2,117
	•	Screenshot: data2.png

MNIST
	•	Accuracy: 100% on test set (binary: “is 3 or not”)
	•	Hyperparams: [0.001768..., 2.8165e-5, 16, 64]
	•	Model Summary:
	•	Conv2D with 16 filters, (3x3)
	•	GlobalAveragePooling2D
	•	Dropout
	•	Dense(10) with softmax
	•	Params: 992 total (330 trainable)

# Reducing runtime:
Implemented early stopping as soon as the model acheived 75% accuracy on the validation or test set.

# Params
The neural network searched for the optimal learning rate, regularization lambda, nodes, and batch size

# Model Architecture
Conv2D: Extracts feature maps (edges/textures) from the image.
GlobalAveragePooling2D: Compresses each feature map to a single value (reduces size, keeps key info).
Dropout: Randomly drops neurons (prevents overfitting).
Dense (Softmax for MNIST, Sigmoid for data): Outputs class probabilities for digits 0–9 or binary 0-1.

# Issues
The model classifies data examples at an accuracy of 0.5. This is likely due to the low number of test examples.



