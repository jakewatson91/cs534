import keras
from keras import Sequential, optimizers, utils, regularizers

from keras import utils
import numpy as np
import pandas as pd
import random

from itertools import product
import json

from sklearn.model_selection import train_test_split

def run(X, y, max_generations=5, population_size=10, mutation_rate=0.1, mnist=False):
    cur_best = 0
    fitness_scores_list = []
    population = [create_individual() for _ in range(population_size)]
    generations = 0

    while generations < max_generations:
        generations += 1

        if mnist == True:
            fitness_scores = [(calculate_fitness(X, y, *individual, mnist=mnist), individual) for individual in population]
        else:
            fitness_scores = [(calculate_test_fitness(X, y, *individual, mnist=mnist), individual) for individual in population]
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
            # print(fitness_scores)

        best_fitness = fitness_scores[0][0]
        fitness_scores_list.append(best_fitness)

        if best_fitness > cur_best:
            cur_best = best_fitness
            best_ind = fitness_scores[0][1]              
            print(f"New best: {cur_best}, {best_ind}")
            print(f"Generations: {generations}")
            if mnist and cur_best >= 0.75:
                break # exit and return
            # if not mnist and cur_best >= 1.0:
            #     break

        # Select top 50% as parents
        parents = [individual for _, individual in fitness_scores[:population_size // 2]]

        # Reproduce
        new_population = []
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(c2, mutation_rate))
        population = new_population

    best_model, _ = train(X, y, *best_ind, mnist=mnist)
    model_found(best_model)
    print(f"Fitness scores mean: {np.mean(fitness_scores_list)}")
    print(f"New best: {cur_best}, {best_ind}")
    return best_model

# TODO your function defs here ....

def mnist_nnet(input_shape=(28,28,1), lr=1e-4, reg=1e-4, nodes=128, num_classes=10):

    model = Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(nodes, kernel_size=(3, 3), activation="relu", kernel_regularizer=regularizers.l2(reg)),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )

    return model

def data_nnet(input_shape=(4,4,1), lr=1e-4, reg=1e-4, nodes=128, num_classes=1):

    model = Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(nodes, kernel_size=(3, 3), activation="relu", kernel_regularizer=regularizers.l2(reg)),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="sigmoid"),
    ])

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )

    return model

def create_individual(): # flattened array # should just be input
    # individual = [random.randint(0,9) for _ in range(length)]
    # individual = np.reshape(individual_flat, shape)
    lr = 10 ** np.random.uniform(-4, -2)
    reg = 10 ** np.random.uniform(-5, -2)
    nodes = 2 ** random.randint(3, 6)
    batch_size = np.random.choice([32,64,128])

    return [lr, reg, nodes, batch_size] 

def mutate(individual, mutation_rate):
    new_individual = individual[:]
    # if random.random() < mutation_rate:
    #     new_individual[1] = 2 ** random.randint(4, 7)       # batch_size
    if random.random() < mutation_rate:
        new_individual[0] = 10 ** random.uniform(-4, -2)    # lr
    if random.random() < mutation_rate:
        new_individual[1] = 10 ** random.uniform(-5, -2)    # reg
    if random.random() < mutation_rate:
        new_individual[2] = 2 ** random.randint(4, 8)       # nodes
    if random.random() < mutation_rate:
        new_individual[3] = np.random.choice([32,64,128])
    return new_individual

def crossover(parent1, parent2):
    crossover_idx = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_idx] + parent2[crossover_idx:]
    child2 = parent2[:crossover_idx] + parent1[crossover_idx:]
    return child1, child2

def train(X, y, *args, mnist):
    input_shape = (image_dims, image_dims, 1)
    lr, reg, nodes, batch_size = args
    if mnist is True:
        model = mnist_nnet(input_shape, lr, reg, nodes)
    else:
        model = data_nnet(input_shape, lr, reg, nodes)

    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15 if mnist else None
    )
    return model, history

def calculate_fitness(X, y, *args, mnist):
    _, history = train(X, y, *args, mnist=mnist)
    val_accuracy = history.history["val_acc"][-1]
    if val_accuracy < 0.15:
        return 0 # stop exploring run
    return val_accuracy

def calculate_test_fitness(X, y, *args, mnist):
    model, _ = train(X, y, *args, mnist=mnist)
    loss, test_accuracy = model.evaluate(X, y)
    # print("Test: ", test_accuracy)
    return test_accuracy

# function to print model found at the very end
def model_found(model):
    print("\n-------Model Found Summary------")
    print(model.summary())

# exectue code

def parse_files(file):
        with open(file, 'r') as f:
            data = json.load(f)
            train = data['train']
            test = data['test']
        return train, test

files = ['data/data_1.json','data/data_1.json', 'data/data_2.json']

image_dims = 4
num_classes = 10
# batch_size = 1
epochs = 5

for i, file in enumerate(files):
    train_data, test_data = parse_files(file)
    # print(train)

    accuracy_scores = []
    X = np.array([d['input'] for d in train_data], dtype=np.float32).reshape(-1, 4, 4, 1)
    y = np.array([[d['output']] for d in train_data], dtype=np.float32)
    best_model = run(X, y)
    for dict in test_data:
        # print(dict)
        individual = np.array(dict['input'], dtype=np.float32).reshape(-1, image_dims, image_dims, 1)
        target = np.array([[dict['output']]], dtype=np.float32)
        loss, acc = best_model.evaluate(individual, target)
        accuracy_scores.append(acc)
    print(f"{file} Accuracy: {np.mean(accuracy_scores)}")    

# BONUS
df = pd.read_csv("data/mnist_train.csv")
data = df.to_numpy()
data = data[data[:, 0] == 3]
# print(data.shape)
X = data[:, 1:]
y = data[:, 0]
# print(X.shape)
# print(y.shape)
# print(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

image_dims = 28
num_classes = 10
# batch_size = 128
epochs = 8

# best_model = run(X_train_reshaped, y_train, mnist=True)
# print("\nEvaluating...")
# best_model.evaluate(X_test_reshaped, y_test)





