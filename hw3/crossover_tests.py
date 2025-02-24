import genetic_algo as ga
import matplotlib.pyplot as plt

population_size = 100
mutation_rate = 0.1
generations = 10000

file = 'data/data_2.json'

train, test = ga.parse_files(file)
print(test[0])

individual = test[0]['input']
target = test[0]['output']

individual_flat = [individual[row][col] for row in range(len(individual)) for col in range(len(individual[0]))]
crossover_indices = list(range(1, len(individual_flat) - 1))
print(crossover_indices)

generations_list = []

fig, ax = plt.subplots(1,2, figsize=(10,5))
for idx in crossover_indices:
    final, generation, mappings, fitness_scores_list = ga.genetic_algorithm(target, individual, population_size, mutation_rate, generations, idx)
    generations_list.append(generation)

    x1 = list(range(len(fitness_scores_list)))
    y1 = fitness_scores_list[::-1]
    ax[0].plot(x1, y1, label=f"Crossover index {idx}")
    ax[0].set_xlabel("Generation")
    ax[0].set_ylabel("Fitness Loss")
    ax[0].legend()

x2 = crossover_indices
y2 = generations_list
ax[1].plot(x2, y2)
ax[1].set_xlabel("Crossover index")
ax[1].set_ylabel("Generation")

fig.suptitle(f"{file}")
plt.savefig(f"plots/crossover_tests")
plt.show()

