import random 
import numpy as np
import json
import matplotlib.pyplot as plt

def parse_files(file):
        with open(file, 'r') as f:
            data = json.load(f)
            train = data['train']
            test = data['test']
        return train, test

def calculate_fitness(target, individual):
    fitness = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            fitness += 1
    return fitness

def create_individual(individual): # flattened array # should just be input
    # individual = [random.randint(0,9) for _ in range(length)]
    # individual = np.reshape(individual_flat, shape)
    individual = [individual[row][col] for row in range(len(individual)) for col in range(len(individual[0]))] # flatten
    return individual

def crossover(parent1, parent2, crossover_idx=None):
    if not crossover_idx:
       crossover_idx = random.randint(1, len(parent1) - 1) # can't be first or last cell
    child1 = parent1[:crossover_idx] + parent2[crossover_idx:]
    child2 = parent2[:crossover_idx] + parent1[crossover_idx:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0,9)
    return individual

def generate_mappings(initial, target):
    res = []
    visited = set()
    mapped = set()

    rows, cols = len(initial), len(initial[0])
    for row in range(rows):
        # print(row)
        for col in range(cols):
            # print(col)
            val = initial[row][col]
            new_val = val
            for i in range(rows):
                for j in range(cols):
                    if (row, col) not in mapped and (i, j) not in visited and new_val == target[i][j]:
                        res.append((val, (row, col), new_val, (i, j))) # (original val, (coordinates), original val, (new coordinates))
                        visited.add((i, j))
                        # print("Visited: ", visited)
                        mapped.add((row, col))
                        # print("res: ", res)
                        break
    return res

def genetic_algorithm(target, individual, population_size, mutation_rate, generations, crossover_index=None):
    # target_array = np.array(target)
    mappings = []
    fitness_scores_list = []
    target_flat = [target[row][col] for row in range(len(target)) for col in range(len(target[0]))]
    individual_flat = [individual[row][col] for row in range(len(individual)) for col in range(len(individual[0]))] # flatten

    population = [individual_flat for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [(calculate_fitness(target_flat, individual_flat), individual_flat) for individual_flat in population]
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        # print(fitness_scores)

        best_fitness = fitness_scores[0][0]
        fitness_scores_list.append(best_fitness)

        if best_fitness == len(target_flat):
            final = np.reshape(fitness_scores[0][1], np.array(target).shape)
            mappings = generate_mappings(individual, final)
            print(f"Match found after {generation} generations!")
            return final, generation, mappings, fitness_scores_list

        parents = [individual_flat for _, individual_flat in fitness_scores[:population_size // 2]]
        
        new_population = []

        # create a new population from the parents and children
        while len(new_population) < population_size:

            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            # print("Parent1: ", parent1)
            # print("Parent2: ", parent2)
            child1, child2 = crossover(parent1, parent2, crossover_index)
            # print("Child1: ", child1)
            # print("Child2: ", child2  )
            new_population.append(mutate(child1, mutation_rate))
            # print("new_pop: ", new_population)
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))
        population = new_population
    
    print("No match found.")
    return final, generation, mappings, fitness_scores_list

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

#TODO find parameters that work for you and use target in your goal test or fitness function

if __name__ == '__main__':
    population_size = 100
    mutation_rate = 0.1
    generations = 10000

    files = ['data/data_0.json', 'data/data_1.json', 'data/data_2.json', 'data/data_3.json', 'data/data_bonus.json']

    for i, file in enumerate(files):
        train, test = parse_files(file)
        # print(train)

        plt.figure()

        for dict in test:
            print(dict)
            individual = dict['input']
            target = dict['output']

            final, generation, mappings, fitness_scores_list = genetic_algorithm(target, individual, population_size, mutation_rate, generations)
            print("Initial: ", individual)
            print("Final: ", final)
            # print(final == target)
            print("Mappings: ", mappings)

            x = list(range(len(fitness_scores_list)))
            y = fitness_scores_list[::-1]
            plt.plot(x, y)
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Fitness Loss")
        plt.title(f"{file}")
        # plt.show()
        plt.savefig(f"plots/genetic_algo_plot_file_{i}")
