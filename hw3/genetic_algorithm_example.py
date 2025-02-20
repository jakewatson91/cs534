import random


def calculate_fitness(target, individual):
    fitness = 0
    # fitness = number of correctly generated letters
    for i in range(len(target)):
        if individual[i] == target[i]:
            fitness += 1
    return fitness


def create_individual(length):
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz1234567890 ') for _ in range(length))


def crossover(parent1, parent2):
    # take a mix of the parents to generate a new child
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, mutation_rate):
    # randomnly mutate every letter in child if random > mutation rate
    individual_list = list(individual)
    for i in range(len(individual_list)):
        if random.random() < mutation_rate:
            individual_list[i] = random.choice('abcdefghijklmnopqrstuvwxyz1234567890 ')
    return "".join(individual_list)


def genetic_algorithm(target, population_size, mutation_rate, generations):
    # randomnly generate the first population
    population = [create_individual(len(target)) for _ in range(population_size)]

    for generation in range(generations):
        print("Generation " + str(generation))
        # print(population)

        # calculate how similar the generation is and order by score
        fitness_scores = [(calculate_fitness(target, individual), individual) for individual in population]
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        # if fitness score == len(target), the generation is the same. Return the generated word gen index
        if fitness_scores[0][0] == len(target):
            return fitness_scores[0][1], generation

        # select parents that we're working from to calculate next generation. Reduce pop size by 2 each gen
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
    return fitness_scores[0][1], generations


target_string = "hello cs 534"
population_size = 100
mutation_rate = 0.01
generations = 10000

best_string, num_generations = genetic_algorithm(target_string, population_size, mutation_rate, generations)

print(f"Target string: {target_string}")
print(f"Best string found: {best_string}")
print(f"Generations: {num_generations}")