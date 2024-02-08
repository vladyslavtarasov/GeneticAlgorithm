import random
import matplotlib.pyplot as plt

from classes import *

bin_capacity = 1000
bits_per_bin_assignment = 5
items = []
start_bin_amount = 30
population_size = 1000
population = []
elite = []
num_elites = 50
selected_individuals = []
temporary_population = []
num_generations = 1000

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
    global items, bin_capacity
    penalty_overfill = 10000
    penalty_empty_space = 0.01

    for individual in pop:
        bins = decode_to_bins(individual.code, items, bin_capacity)
        bins = [bin for bin in bins if bin.items]
        overfilled = any(b.remaining_capacity < 0 for b in bins)
        total_empty_space = sum(b.remaining_capacity for b in bins if b.remaining_capacity >= 0)

        if overfilled:
            individual.fitness = -penalty_overfill
        else:
            individual.fitness = 1000 / len(bins) - penalty_empty_space * total_empty_space

        individual.amount_of_bins = len(bins)


def decode_to_bins(code, items, bin_capacity):
    bins = []
    bin_assignments = decode_bin_assignments(code, start_bin_amount)

    for item_index, bin_index in enumerate(bin_assignments):
        while bin_index >= len(bins):
            bins.append(Bin(bin_capacity))

        bins[bin_index].add(items[item_index])

    return bins

def decode_bin_assignments(code, bin_count):
    bin_assignments = []
    for i in range(0, len(code), bits_per_bin_assignment):
        bin_assignment = int(code[i:i+bits_per_bin_assignment], 2)
        if bin_assignment >= bin_count:
            bin_assignment = bin_count - 1
        bin_assignments.append(bin_assignment)
    return bin_assignments


def get_elites():
    global elite
    elite.clear()

    max_fitness = max(individual.fitness for individual in population)

    max_fitness_individuals = [Individual(ind.code) for ind in population if ind.fitness == max_fitness]

    if len(max_fitness_individuals) == num_elites:
        elite.extend(max_fitness_individuals)
    elif len(max_fitness_individuals) > num_elites:
        elite.extend(random.sample(max_fitness_individuals, num_elites))
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


def single_point_crossover():
    global temporary_population
    next_temporary_population = []

    random.shuffle(temporary_population)

    while len(temporary_population) > 1:
        parent1 = temporary_population.pop()
        parent2 = temporary_population.pop()

        if random.random() < crossover_chance:
            cross_point = random.randint(1, ((bits_per_bin_assignment * len(items)) // bits_per_bin_assignment) - 1) * bits_per_bin_assignment
            child1_code = parent1.code[:cross_point] + parent2.code[cross_point:]
            child2_code = parent2.code[:cross_point] + parent1.code[cross_point:]
            child1 = Individual(child1_code)
            child2 = Individual(child2_code)
            next_temporary_population.extend([child1, child2])
        else:
            next_temporary_population.extend([parent1, parent2])

    if temporary_population:
        next_temporary_population.append(temporary_population.pop())

    temporary_population = next_temporary_population


def mutation():
    for individual in temporary_population:
        if random.random() < mutation_chance:
            item_to_mutate = random.randint(0, ((bits_per_bin_assignment * len(items)) // bits_per_bin_assignment) - 1)
            mutate_position = item_to_mutate * bits_per_bin_assignment
            code_list = list(individual.code)

            current_bin = int(''.join(code_list[mutate_position:mutate_position + bits_per_bin_assignment]), 2)
            new_bin = current_bin
            while new_bin == current_bin:
                new_bin = random.randint(0, start_bin_amount - 1)

            new_bin_assignment = format(new_bin, f'0{bits_per_bin_assignment}b')
            code_list[mutate_position:mutate_position + bits_per_bin_assignment] = list(new_bin_assignment)
            individual.code = ''.join(code_list)


def adjust_bins(pop):
    global items, bin_capacity, start_bin_amount

    for individual in pop:
        bins = [Bin(bin_capacity) for _ in range(start_bin_amount)]
        bin_assignments = decode_bin_assignments(individual.code, start_bin_amount)

        item_placed = [False] * len(items)

        for item_index, bin_index in enumerate(bin_assignments):
            if bins[bin_index].remaining_capacity >= items[item_index]:
                bins[bin_index].add(items[item_index])
                item_placed[item_index] = True

        for item_index, placed in enumerate(item_placed):
            if not placed:
                placed_in_existing_bin = False
                for bin in range(len(bins)):
                    if bins[bin].remaining_capacity >= items[item_index]:
                        bins[bin].add(items[item_index])
                        bin_assignments[item_index] = bins.index(bins[bin])
                        placed_in_existing_bin = True
                        break

                if not placed_in_existing_bin:
                    new_bin = Bin(bin_capacity)
                    new_bin.add(items[item_index])
                    bins.append(new_bin)
                    bin_assignments[item_index] = bins.index(new_bin)
                    start_bin_amount += 1


        new_code = ''.join([format(bin_assignment, f'0{bits_per_bin_assignment}b') for bin_assignment in bin_assignments])
        individual.code = new_code


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
    global items, bin_capacity

    bins = [Bin(bin_capacity) for _ in range(100)]
    bin_assignments = decode_bin_assignments(best_solution.code, start_bin_amount)

    for item_index, bin_index in enumerate(bin_assignments):
        item_weight = items[item_index]
        bins[bin_index].add(item_weight)

    non_empty_bins = [bin for bin in bins if bin.items]

    print("Best Solution Bins:")
    for i, bin in enumerate(non_empty_bins, start=1):
        print(f"Bin {i}: {bin}")


read_items_from_file(file_paths[4])
print(items)

for _ in range(population_size):
    code = ''
    for _ in range(len(items)):
        bin_index = random.randint(1, start_bin_amount)
        bin_index_binary = format(bin_index, f'0{bits_per_bin_assignment}b')
        code += bin_index_binary
    population.append(Individual(code))

adjust_bins(population)
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

    single_point_crossover()
    adjust_bins(temporary_population)
    calculate_fitness(temporary_population)

    mutation()
    adjust_bins(temporary_population)
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

    if generation_with_same_fitness >= 40:
        break


population.sort(key=lambda ind: ind.fitness, reverse=True)
print("\nBest individual:")
print(population[0])
print()
display_best_solution(population[0])

plt.plot(average_fitness_over_generations)
plt.title('Average Fitness Over Generations (Bits)')
plt.xlabel('Generation Number')
plt.ylabel('Average Fitness')
plt.grid(True)
plt.show()

plt.plot(best_fitness_over_generations)
plt.title('Best Fitness Over Generations (Bits)')
plt.xlabel('Generation Number')
plt.ylabel('Average Fitness')
plt.grid(True)
plt.show()
