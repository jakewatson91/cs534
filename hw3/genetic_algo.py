
def calculate_fitness(target, individual):
   #TODO how close is my current mapping function to the target?
    pass


def create_individual(length):
    #TODO create a random individual mapping function to seed the initial population
    pass

def crossover(parent1, parent2):
    #TODO create a cross over point (random or otherwise) and reutrn to children
    pass


def mutate(individual, mutation_rate):
    #TODO given some probabilty, mutation part of the individual and return
    pass

def genetic_algorithm(target, population_size, mutation_rate, generations):
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
    pass


#TODO find parameters that work for you and use target in your goal test or fitness function
target_functions = None # TODO load from json
population_size = 100
mutation_rate = 0.01
generations = 10000

best_function = genetic_algorithm(target_functions, population_size, mutation_rate, generations)
