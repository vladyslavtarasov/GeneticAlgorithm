import random
import matplotlib.pyplot as plt

from classes import *

bin_capacity = 1000
bins = []
items = []
population_size = 100
population = []
elite = []
selected_individuals = []
temporary_population = []
num_generations = 100

crossover_chance = 0.8
mutation_chance = 0.2

fitness_value = 0
generation_with_same_fitness = 0
sum_fitness = 0
average_fitness = 0

file_paths = [
    '../problems/bpp_1.txt',
    '../problems/bpp_2.txt',
    '../problems/bpp_3.txt',
    '../problems/bpp_4.txt',
    '../problems/bpp_5.txt'
]

average_fitness_over_generations = []
best_fitness_over_generations = []


def calculate_fitness(pop):
    global bins, items

    for individual in pop:
        bins = [Bin(bin_capacity)]
        total_empty_space = 0

        for item_index in individual.code:
            item_weight = items[item_index - 1]

            if item_weight <= bins[-1].remaining_capacity:
                bins[-1].add(item_weight)
            else:
                total_empty_space += bins[-1].remaining_capacity
                new_bin = Bin(bin_capacity)
                new_bin.add(item_weight)
                bins.append(new_bin)

        total_empty_space += bins[-1].remaining_capacity

        individual.amount_of_bins = len(bins)
        penalty_factor = 0.01
        individual.fitness = 1000 / len(bins) - penalty_factor * total_empty_space


def get_elites():
    global elite
    elite.clear()

    max_fitness = max(individual.fitness for individual in population)

    max_fitness_individuals = [Individual(ind.code) for ind in population if ind.fitness == max_fitness]

    if len(max_fitness_individuals) == 2:
        elite.extend(max_fitness_individuals)
    elif len(max_fitness_individuals) > 2:
        elite.extend(random.sample(max_fitness_individuals, 2))
    else:
        elite.append(max_fitness_individuals[0])

        second_best_fitness = max(individual.fitness for individual in population if individual.fitness != max_fitness)
        second_best_individuals = [Individual(ind.code) for ind in population if ind.fitness == second_best_fitness]

        elite.append(random.choice(second_best_individuals))


def tournament_selection(tournament_size):
    global selected_individuals
    selected_individuals.clear()

    while population:
        current_tournament_size = min(tournament_size, len(population))
        tournament_individuals = population[:current_tournament_size]
        winner = max(tournament_individuals, key=lambda ind: ind.fitness)
        selected_individuals.append(winner)
        population[:current_tournament_size] = []


def duplicate_by_rank():
    global temporary_population
    temporary_population = [Individual(ind.code) for ind in selected_individuals]

    total_needed = population_size - len(elite) - len(temporary_population)

    if total_needed > 0:
        sorted_individuals = sorted(selected_individuals, key=lambda ind: ind.fitness, reverse=True)

        weights = list(range(1, len(sorted_individuals) + 1))[::-1]

        selected_indices = random.choices(range(len(sorted_individuals)), weights=weights, k=total_needed)

        for index in selected_indices:
            individual = sorted_individuals[index]
            new_ind = Individual(individual.code)
            temporary_population.append(new_ind)


def pmx_crossover():
    global temporary_population
    next_temporary_population = []

    random.shuffle(temporary_population)

    while len(temporary_population) > 1:
        parent1 = temporary_population.pop()
        parent2 = temporary_population.pop()

        if random.random() < crossover_chance:
            child1_code, child2_code = perform_pmx(parent1.code, parent2.code)

            child1 = Individual(child1_code)
            child2 = Individual(child2_code)

            next_temporary_population.extend([child1, child2])
        else:
            next_temporary_population.extend([parent1, parent2])

    if temporary_population:
        next_temporary_population.append(temporary_population.pop())

    temporary_population = next_temporary_population


def perform_pmx(parent1, parent2):
    length = len(parent1)
    child1, child2 = parent1.copy(), parent2.copy()

    cross_points = sorted(random.sample(range(length), 2))

    mapping1, mapping2 = {}, {}
    for i in range(cross_points[0], cross_points[1] + 1):
        item1, item2 = child1[i], child2[i]
        child1[i], child2[i] = item2, item1
        mapping1[item1] = item2
        mapping2[item2] = item1

    for i in range(length):
        if i not in range(cross_points[0], cross_points[1] + 1):
            while child1[i] in mapping2:
                child1[i] = mapping2[child1[i]]

    for i in range(length):
        if i not in range(cross_points[0], cross_points[1] + 1):
            while child2[i] in mapping1:
                child2[i] = mapping1[child2[i]]

    return child1, child2


def ox_crossover():
    global temporary_population
    next_temporary_population = []

    random.shuffle(temporary_population)

    while len(temporary_population) > 1:
        parent1 = temporary_population.pop()
        parent2 = temporary_population.pop()

        if random.random() < crossover_chance:
            child1_code, child2_code = perform_ox(parent1.code, parent2.code)

            child1 = Individual(child1_code)
            child2 = Individual(child2_code)

            next_temporary_population.extend([child1, child2])
        else:
            next_temporary_population.extend([parent1, parent2])

    if temporary_population:
        next_temporary_population.append(temporary_population.pop())

    temporary_population = next_temporary_population


def perform_ox(parent1, parent2):
    length = len(parent1)
    child1, child2 = [-1] * length, [-1] * length

    cross_points = sorted(random.sample(range(length), 2))

    child1[cross_points[0]:cross_points[1]+1] = parent1[cross_points[0]:cross_points[1]+1]
    child2[cross_points[0]:cross_points[1]+1] = parent2[cross_points[0]:cross_points[1]+1]

    def fill_child(child, parent):
        current_pos = (cross_points[1] + 1) % length
        for gene in parent:
            if gene not in child:
                child[current_pos] = gene
                current_pos = (current_pos + 1) % length
                if current_pos == cross_points[0]:
                    current_pos = (cross_points[1] + 1) % length

    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return child1, child2


def cx_crossover():
    global temporary_population
    next_temporary_population = []

    random.shuffle(temporary_population)

    while len(temporary_population) > 1:
        parent1 = temporary_population.pop()
        parent2 = temporary_population.pop()

        if random.random() < crossover_chance:
            child1_code, child2_code = perform_cx(parent1.code, parent2.code)

            child1 = Individual(child1_code)
            child2 = Individual(child2_code)

            next_temporary_population.extend([child1, child2])
        else:
            next_temporary_population.extend([parent1, parent2])

    if temporary_population:
        next_temporary_population.append(temporary_population.pop())

    temporary_population = next_temporary_population


def perform_cx(parent1, parent2):
    length = len(parent1)
    child1, child2 = parent1.copy(), parent2.copy()
    visited = [False] * length

    cycle_num = 0
    while not all(visited):
        if not visited[cycle_num]:
            start_index = cycle_num
            while not visited[start_index]:
                visited[start_index] = True
                start_index = parent1.index(parent2[start_index])
            cycle_num += 1
        else:
            cycle_num = visited.index(False)

    for i in range(length):
        if visited[i]:
            child1[i], child2[i] = child2[i], child1[i]

    return child1, child2


def mutation():
    global temporary_population
    for individual in temporary_population:
        if random.random() < mutation_chance:
            index1, index2 = random.sample(range(len(individual.code)), 2)
            individual.code[index1], individual.code[index2] = individual.code[index2], individual.code[index1]


def read_items_from_file(file_path):
    global items

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:
                weight, quantity = parts
                weight = int(weight)
                quantity = int(quantity)

                items.extend([weight] * quantity)


def display_best_solution(best_solution):
    bins = [Bin(bin_capacity)]

    for item_index in best_solution.code:
        item_weight = items[item_index - 1]
        placed = False
        for b in bins:
            if item_weight <= b.remaining_capacity:
                b.add(item_weight)
                placed = True
                break

        if not placed:
            new_bin = Bin(bin_capacity)
            new_bin.add(item_weight)
            bins.append(new_bin)

    print("Best Solution Bins:")
    for i, bin in enumerate(bins, start=1):
        print(f"Bin {i}: {bin}")


read_items_from_file(file_paths[0])
print(items)


for i in range(population_size):
    new_code = random.sample(range(len(items)), len(items))
    population.append(Individual(new_code))

calculate_fitness(population)

best_fitness = max(individual.fitness for individual in population)
best_fitness_over_generations.append(best_fitness)
print(f"Best Fitness in Generation 0: {best_fitness}")

sum_fitness = 0
for individual in population:
    sum_fitness += individual.fitness
average_fitness = sum_fitness / len(population)
average_fitness_over_generations.append(average_fitness)
print(f"Average Fitness in Generation 0: {average_fitness}")

for generation in range(num_generations):
    print(f"Generation {generation + 1}")

    get_elites()

    tournament_selection(2)
    duplicate_by_rank()

    pmx_crossover()
    # ox_crossover()
    # cx_crossover()
    calculate_fitness(temporary_population)

    mutation()
    calculate_fitness(temporary_population)

    temporary_population.extend(elite)

    population = temporary_population.copy()
    temporary_population.clear()
    calculate_fitness(population)

    best_fitness = max(individual.fitness for individual in population)
    best_fitness_over_generations.append(best_fitness)
    print(f"Best Fitness in Generation {generation + 1}: {best_fitness}")

    sum_fitness = 0
    for individual in population:
        sum_fitness += individual.fitness

    average_fitness = sum_fitness / len(population)
    average_fitness_over_generations.append(average_fitness)
    print(f"Average Fitness in Generation {generation + 1}: {average_fitness}")

    if best_fitness == fitness_value:
        generation_with_same_fitness += 1
    else:
        fitness_value = best_fitness
        generation_with_same_fitness = 0

    if generation_with_same_fitness >= 10:
        break


population.sort(key=lambda ind: ind.fitness, reverse=True)
print("\nBest individual:")
print(population[0])
print()
display_best_solution(population[0])

plt.plot(average_fitness_over_generations)
plt.title('Average Fitness Over Generations (Order, Problem 1)')
plt.xlabel('Generation Number')
plt.ylabel('Average Fitness')
plt.grid(True)
plt.show()

plt.plot(best_fitness_over_generations)
plt.title('Best Fitness Over Generations (Order, Problem 1)')
plt.xlabel('Generation Number')
plt.ylabel('Average Fitness')
plt.grid(True)
plt.show()
