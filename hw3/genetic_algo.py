import random 
import numpy as np

def calculate_fitness(target, individual):
    fitness = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            fitness += 1
    return fitness

def create_individual(length): # flattened array # should just be input
    individual = [random.randint(0,9) for _ in range(length)]
    # individual = np.reshape(individual_flat, shape)
    return individual

def crossover(parent1, parent2):
    crossover_idx = random.randint(1, len(parent1) - 1) # can't be first or last cell
    child1 = parent1[:crossover_idx] + parent2[crossover_idx:]
    child2 = parent2[:crossover_idx] + parent1[crossover_idx:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0,9)
    return individual

def genetic_algorithm(target, population_size, mutation_rate, generations):
    # target_array = np.array(target)
    target_flat = [target[row][col] for row in range(len(target)) for col in range(len(target[0]))]
    population = [create_individual(len(target_flat)) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [(calculate_fitness(target_flat, individual), individual) for individual in population]
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        if fitness_scores[0][0] == len(target_flat):
            final = np.reshape(fitness_scores[0][1], np.array(target).shape)
            return final, generation
        
        parents = [individual for _, individual in fitness_scores[:population_size // 2]]
        
        new_population = []

        # create a new population from the parents and children
        while len(new_population) < population_size:

            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))
        population = new_population
    
    return final, generation

    # population = [create_individual()]
    #TODO (1) create initial population (generation 1)
    #TODO (2) for each generation
    #TODO (3) if generation contains an individual (i.e., mapping function) stop and return
    #TODO (4) choose parents using fitness function (i.e., goal score) and produce next generation using crossover
    #TODO (5) update generation and repeat starting at (2) above
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
target_functions = [[3, 3],[0, 0]]
population_size = 100
mutation_rate = 0.1
generations = 10000

final, generation = genetic_algorithm(target_functions, population_size, mutation_rate, generations)
print(final, generation)