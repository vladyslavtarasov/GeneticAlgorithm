import random
import matplotlib.pyplot as plt

from classes import *

string_length = 30
population_size = 100
population = []
elite = []
selected_individuals = []
temporary_population = []
num_generations = 300
fitness_value = 0
generation_with_same_fitness = 0
sum_fitness = 0
average_fitness = 0

crossover_chance = 0.7
mutation_chance = 0.1

target_string = ''.join(random.choice(['0', '1']) for _ in range(string_length))

average_fitness_over_generations = []
best_fitness_over_generations = []


def calculate_fitness(pop):
    one_max_fitness(pop)
    # target_string_fitness(pop)
    # deceptive_landscape_fitness(pop)


def one_max_fitness(pop):
    for individual in pop:
        individual.fitness = individual.code.count('1')


def target_string_fitness(pop):
    for individual in pop:
        individual.fitness = sum(ind_char == targ_char for ind_char, targ_char in zip(individual.code, target_string))


def deceptive_landscape_fitness(pop):
    for individual in pop:
        count_of_ones = individual.code.count('1')
        if count_of_ones == 0:
            individual.fitness = 2 * string_length
        else:
            individual.fitness = count_of_ones


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


def single_point_crossover():
    global temporary_population
    next_temporary_population = []

    random.shuffle(temporary_population)

    while len(temporary_population) > 1:
        parent1 = temporary_population.pop()
        parent2 = temporary_population.pop()

        if random.random() < crossover_chance:
            cross_point = random.randint(1, string_length - 1)
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
            mutate_position = random.randint(0, string_length - 1)
            code_list = list(individual.code)
            code_list[mutate_position] = '1' if code_list[mutate_position] == '0' else '0'
            individual.code = ''.join(code_list)


for i in range(population_size):
    new_code = ''.join(random.choice(['0', '1']) for _ in range(string_length))
    new_individual = Individual(new_code)
    population.append(new_individual)

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

for o in population:
    print(o)

for generation in range(num_generations):
    print(f"Generation {generation + 1}")

    get_elites()

    tournament_selection(2)
    duplicate_by_rank()

    single_point_crossover()
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

    for individual in population:
        print(individual)

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

plt.plot(average_fitness_over_generations)
plt.title('Average Fitness Over Generations (1.2)')
plt.xlabel('Generation Number')
plt.ylabel('Average Fitness')
plt.grid(True)
plt.show()

plt.plot(best_fitness_over_generations)
plt.title('Best Fitness Over Generations (1.2)')
plt.xlabel('Generation Number')
plt.ylabel('Average Fitness')
plt.grid(True)
plt.show()
