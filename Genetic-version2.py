import itertools
import math
import random
import time

# Define the genetic algorithm parameters
POPULATION_SIZE = 1000
MAX_GENERATIONS = 500

# Define the range of values for x, y, and z
X_RANGE = (-10, 10)
Y_RANGE = (-10, 10)
Z_RANGE = (-10, 10)


def equation1(alpha, x, y, z):
    result = (alpha * x) + (y * math.pow(x, 2)) + math.pow(y, 3) + math.pow(z, 3)
    return math.pow(result, 2)


def equation2(beta, x, y, z):
    result = (beta * y) + math.sin(math.radians(y)) + math.pow(2, y) - z + math.log10(abs(x) + 1)
    return math.pow(result, 2)


def equation3(teta, x, y, z):
    denominator = (math.sin(math.radians((z * y) - (y * y) + z)) + 2)
    deduction_result = math.cos(math.radians(x + y)) / denominator
    return math.pow((teta * z) + y - deduction_result, 2)


def initialize_population(size=POPULATION_SIZE):
    population = []
    for _ in range(size):
        x = round(random.uniform(*X_RANGE), 6)
        y = round(random.uniform(*Y_RANGE), 6)
        z = round(random.uniform(*Z_RANGE), 6)
        population.append((x, y, z))
    return population


def initialize_population2(alpha, beta, teta, size=POPULATION_SIZE):
    best_populations = []
    for _ in range(16):
        population = evaluate_fitness(alpha, beta, teta, initialize_population())
        new_population = []
        for counter in range(100):
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            mutated_child = mutate1(child, counter)
            new_population.append(mutated_child)

        new_population = evaluate_fitness(alpha, beta, teta, new_population)
        best_populations.append(sorted(new_population, key=lambda d: d['fitness'])[0:125])

    best_populations = list(itertools.chain(*best_populations))
    random.shuffle(best_populations)
    return best_populations


def merge_population2(population1, population2):
    population = []
    length = min(len(population1), len(population2))
    i = 0
    j = 0
    # fitness should be smaller ...
    for _ in range(length):
        if population1[i]['fitness'] < population2[j]['fitness']:
            population.append(population1[i])
            i += 1
        else:
            population.append(population2[j])
            j += 1

    return population


def evaluate_fitness(alpha, beta, teta, population):
    fitness_scores = []
    for individual in population:
        x, y, z = individual
        fitness = equation1(alpha, x, y, z) + equation2(beta, x, y, z) + equation3(teta, x, y, z)
        fitness_scores.append({'x': x, 'y': y, 'z': z, 'fitness': fitness})

    return fitness_scores


def select_parents(fitness_scores):
    # population1 = sorted(fitness_scores, key=lambda d: d['fitness'])
    # ranks = [entry['fitness'] for entry in fitness_scores]
    # chosen = random.choices(population1, weights=ranks[::-1], k=2)
    # first = chosen[0]
    # second = chosen[1]
    # Q-tournament
    first = random.sample(fitness_scores, random.randint(10, 20))
    second = random.sample(fitness_scores, random.randint(10, 20))

    first = min(first, key=lambda elem: elem['fitness'])
    second = min(second, key=lambda elem: elem['fitness'])

    first = tuple(first[coord] for coord in ['x', 'y', 'z'])
    second = tuple(second[coord] for coord in ['x', 'y', 'z'])

    return first, second


def crossover(parent1, parent2):
    # Perform crossover by taking a random weighted average of the parents' genes
    x = round((parent1[0] + parent2[0]) / 2, 6)
    y = round((parent1[1] + parent2[1]) / 2, 6)
    z = round((parent1[2] + parent2[2]) / 2, 6)
    return tuple([x, y, z])


def mutate1(individual, num):
    # Mutate an individual by randomly perturbing its genes
    # mutated_individual = []
    # if num == 0:
    #     num = 0.01
    # flag = random.randint(0, 1)
    # for gene in individual:
    #     if flag:
    #         gene -= round(gene * (math.pow(2, -1 * math.log10(num))), 6)
    #     else:
    #         gene += round(gene * (math.pow(2, -1 * math.log10(num))), 6)
    #     mutated_individual.append(round(gene, 6))
    # return tuple(mutated_individual)
    x, y, z = individual
    return x + random.uniform(-0.01, 0.01), y + random.uniform(-0.01, 0.01), z + random.uniform(-0.01, 0.01)


def mutate2(individual, mutation_rate=0.5):
    # mutated_individual = []
    # for gene in individual:
    #     if random.random() < mutation_rate:
    #         gene += round(random.uniform(-10, 10) - gene, 6)
    #     mutated_individual.append(gene)
    # return tuple(mutated_individual)
    x, y, z = individual
    return x + random.uniform(-0.01, 0.01), y + random.uniform(-0.01, 0.01), z + random.uniform(-0.01, 0.01)


def mutate3(chromosome):
    arr = []
    if random.randint(0, 1):
        flag = random.randint(0, 1)
        if flag:
            arr.append(chromosome[0] + round(random.uniform(-1, 1), 6))
        else:
            arr.append(chromosome[0] - round(random.uniform(-1, 1), 6))
    else:
        arr.append(chromosome[0])
    if random.randint(0, 1):
        flag = random.randint(0, 1)
        if flag:
            arr.append(chromosome[1] + round(random.uniform(-1, 1), 6))
        else:
            arr.append(chromosome[1] - round(random.uniform(-1, 1), 6))
    else:
        arr.append(chromosome[1])
    if random.randint(0, 1):
        flag = random.randint(0, 1)
        if flag:
            arr.append(chromosome[2] + round(random.uniform(-1, 1), 6))
        else:
            arr.append(chromosome[2] - round(random.uniform(-1, 1), 6))
    else:
        arr.append(chromosome[2])

    return tuple(arr)


def merge_population(population1, population2):
    population = []
    population1 = sorted(population1, key=lambda d: d['fitness'])
    population2 = sorted(population2, key=lambda d: d['fitness'])
    length = min(len(population1), len(population2))
    i = 0
    j = 0
    # fitness should be smaller ...
    for _ in range(length):
        if population1[i]['fitness'] < population2[j]['fitness']:
            population.append(tuple([population1[i]['x'], population1[i]['y'], population1[i]['z']]))
            i += 1
        else:
            population.append(tuple([population2[j]['x'], population2[j]['y'], population2[j]['z']]))
            j += 1

    random.shuffle(population)
    return population


def ordinary_merge_population(population1, population2):
    length = min(len(population1), len(population2))
    population1 = population1[0:length]
    population2 = population2[0:length]
    population = []

    population1 = sorted(population1, key=lambda d: d['fitness'])
    population2 = sorted(population2, key=lambda d: d['fitness'])
    population = population1[0:50] + population2[0:950]

    random.shuffle(population)
    return population


def solver(alpha, beta, teta):
    population = initialize_population()
    fitness_scores = []
    old_fitness = 0
    opportunity = 0
    for generation in range(MAX_GENERATIONS):
        # first population
        fitness_scores = evaluate_fitness(alpha, beta, teta, population)

        # Check if we have found a solution
        min_fitness = min(fitness_scores, key=lambda elem: elem['fitness'])
        print('for generation ', generation, 'fitness is : ', min_fitness['fitness'])
        if min_fitness['fitness'] < 0.001:
            return min_fitness['x'], min_fitness['y'], min_fitness['z']
            
            
        elif generation != 0 and generation % 4 == 0:
            fitness_scores = evaluate_fitness(alpha, beta, teta, initialize_population2(alpha, beta, teta))
        
            if round(min_fitness['fitness'], 2) == round(old_fitness, 2) and opportunity >= 1:
                # ! -> local optimum
                # scenario 1
                fitness_scores2 = fitness_scores + initialize_population2(alpha, beta, teta)
                random.shuffle(fitness_scores2)
                fitness_scores = random.sample(fitness_scores2, 1000)

                 # scenario2
                 new_pop = []
                 for chromosome in fitness_scores:
                     mu = mutate2(tuple([chromosome['x'], chromosome['y'], chromosome['z']]))
                     new_pop.append(mu)
                 fitness_scores = evaluate_fitness(alpha, beta, teta, new_pop)
                 opportunity = 0
             else:
                 reproduction = initialize_population2(alpha, beta, teta)
                 merge_reproduction = ordinary_merge_population(fitness_scores, reproduction)
                 fitness_scores = merge_reproduction
                 opportunity += 1
                 old_fitness = min_fitness['fitness']
        

        # Select parents, perform crossover and mutate !
        new_population = []
        for counter in range(1000):
            parent1, parent2 = select_parents(fitness_scores)
            child = crossover(parent1, parent2)
            mutated_child = mutate1(child, counter)
            new_population.append(mutated_child)

        # select the best childes and parents and merge them for next population
        population = merge_population(fitness_scores, evaluate_fitness(alpha, beta, teta, new_population))
        # population = new_population

    # Return the best individual from the last generation
    population = evaluate_fitness(alpha, beta, teta, population)
    selected = min(population, key=lambda elem: elem['fitness'])
    return selected['x'], selected['y'], selected['z']


if __name__ == '__main__':
    a = 1
    b = 2
    t = 3
    x1 = time.time()
    x, y, z = solver(a, b, t)
    x2 = time.time()
    print(equation1(a, x, y, z) + equation2(b, x, y, z) + equation3(t, x, y, z))
    print('time : ', x2 - x1)
