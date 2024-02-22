import math
import random
import time

# Define the genetic algorithm parameters
POPULATION_SIZE = 100
MAX_GENERATIONS = 1000
MUTATION_RATE = 0.1

# Define the range of values for x, y, and z
X_RANGE = (-10, 10)
Y_RANGE = (-10, 10)
Z_RANGE = (-10, 10)


def equation1(alpha, x, y, z):
    result = (alpha * x) + (y * math.pow(x, 2)) + math.pow(y, 3) + math.pow(z, 3)
    return math.pow(result, 2)


def equation2(beta, x, y, z):
    result = (beta * y) + math.sin(y) + math.pow(2, y) - z + math.log10(abs(x) + 1)
    return math.pow(result, 2)


def equation3(teta, x, y, z):
    denominator = math.sin((z * y) - (y * y) + z) + 2
    deduction_result = math.cos(x + y) / denominator
    return math.pow((teta * z) + y - deduction_result, 2)


def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        z = random.uniform(-10, 10)
        population.append((x, y, z))
    return population


def evaluate_fitness(alpha, beta, teta, population):
    fitness_scores = []
    for individual in population:
        x, y, z = individual
        fitness = equation1(alpha, x, y, z) + equation2(beta, x, y, z) + equation3(teta, x, y, z)
        fitness_scores.append({'x': x, 'y': y, 'z': z, 'fitness': fitness})

    return sorted(fitness_scores, key=lambda d: d['fitness'])


def select_parents(fitness_scores):
    # Select two parents using ranking selection
    parent1, parent2 = random.sample(fitness_scores[0:50], 2)
    parent1 = tuple(parent1[coord] for coord in ['x', 'y', 'z'])
    parent2 = tuple(parent2[coord] for coord in ['x', 'y', 'z'])

    return parent1, parent2


def crossover(parent1, parent2):
    # Perform crossover by taking a random weighted average of the parents' genes
    x = (parent1[0] + parent2[0]) / 2
    y = (parent1[1] + parent2[1]) / 2
    z = (parent1[2] + parent2[2]) / 2
    return x, y, z


def mutate(individual):
    # Mutate an individual by randomly perturbing its genes
    x, y, z = individual
    return x + random.uniform(-0.005, 0.005), y + random.uniform(-0.005, 0.005), z + random.uniform(-0.005, 0.005)


def merge_population(population1, population2):
    population = []
    length = len(population1)
    i = 0
    j = 0
    # rank should be smaller ...
    for _ in range(POPULATION_SIZE):
        if population1[i]['rank'] < population2[j]['rank']:
            population.append(tuple([population1[i]['x'], population1[i]['y'], population1[i]['z']]))
            i += 1
        else:
            population.append(tuple([population2[j]['x'], population2[j]['y'], population2[j]['z']]))
            j += 1

    return population


def solver(alpha, beta, teta):
    population = initialize_population()
    fitness_scores = []
    while True:
        # first population
        fitness_scores = evaluate_fitness(alpha, beta, teta, population)

        # Check if we have found a solution
        print(fitness_scores[0]['fitness'])
        if fitness_scores[0]['fitness'] <= 0.01:
            return fitness_scores[0]['x'], fitness_scores[0]['y'], fitness_scores[0]['z']

        # Select parents, perform crossover and mutate !
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent1, parent2 = select_parents(fitness_scores)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child)
            new_population.append(mutated_child)
            
        # select the best childes and parents and merge them for next population
        population = new_population

    # # Return the best individual from the last generation (shouldn't execute !)
    # selected = min(evaluate_fitness(alpha, beta, teta, population), key=lambda elem: elem['fitness'])
    # return selected['x'], selected['y'], selected['z']


if __name__ == '__main__':
    a = -1
    b = -3
    t = 5
    x1 = time.time()
    x, y, z = solver(a, b, t)
    x2 = time.time()
    print(x, ' ,', y, ',', z)
    print(equation1(a, x, y, z) + equation2(b, x, y, z) + equation3(t, x, y, z))
    print('time : ', x2 - x1)
