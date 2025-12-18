import random
import matplotlib.pyplot as plt


# ***** Genetic Algorithm for the Knapshack Problem *****

# 1. Genome of the problem

# Initializing a dictionary of boxes
boxes = {
    "box_1": [20, 6],
    "box_2": [30, 5],
    "box_3": [60, 8],
    "box_4": [90, 7],
    "box_5": [50, 6],
    "box_6": [70, 9],
    "box_7": [30, 4],
    "box_8": [30, 5],
    "box_9": [70, 4],
    "box_10": [20, 9],
    "box_11": [20, 2],
    "box_12": [60, 1],
}


# Initializing variables
box_labels = list(boxes.keys())
max_weight = 250    # max weight threshold
mutation_rate = 0.1  # mutation rate
print(' Please enter Parent population size and Number of generations')
try:
        Pop_size = int(input("Parent population size: "))
        Generation = int(input("Number of generation"))
except ValueError:
        print("Invalid input: Please enter integer values")


# Generate genomes
#  The genome function allow us to generate a chromosome (in binary format)
# and evaulate the total fitness value of it and return the corresponding weight and value

def genome():
    while True:
        indiv = [random.randint(0, 1) for _ in range(len(boxes))]
        weight, value = evaluate_fitness(indiv)        
        if weight <= max_weight:
            # print(f'Genome:{indiv} Total weight:{weight} Total value:{value}')
            selected_boxes = [label for gene, label in zip(indiv, box_labels)]
            print(f'Genome:{selected_boxes} Total weight: {weight} Total value: {value}')
            return indiv, weight, value
        

# Evaluate Function
# This funciton evaluates fitness of each genome and gives total weight and total value
def evaluate_fitness(indiv):
    total_weight = total_value = 0
    for gene, label in zip(indiv, box_labels):
        if gene:                         
            w, v = boxes[label]
            total_weight += w
            total_value += v
    return total_weight, total_value

# Generating a population of size n 
def generate_population(n = Pop_size):
    population = []
    for _ in range(n):
        indiv, w, v = genome()
        population.append((indiv, w, v))
    return population


#  Fringe operators

# CROSSOVER
# This function allow us to perform single point crossover of two parents and produce a child
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    child = parent1[:point]+parent2[point:]
    return child

# Mutation operator
#  Ths function randomly mutates the genome at mutation rate
def mutate(indiv, rate=mutation_rate):
    for i in range(len(indiv)):
        if random.random() < rate:
            indiv[i] = 1 - indiv[i]
    return indiv


# The evolve function evaluates the fitness of the generation and creates the average and best fitness 
def evolve():
    print('******* Parent Generation *******')
    p = generate_population(Pop_size)
    best_fitness = []
    avg_fitness = []
    best_genome = []
    best_weight = []
    for g in range(1, Generation+1):
        
        #  Sort the population in descending order of values
        p.sort(key=lambda x:x[2], reverse=True)
        
        # Cull only top 50% of population
        top50 = p[:len(p)//2]

        # Crating new generation based on top 50% population
        new_gen = []
        while len(new_gen) < Pop_size:
            p1, p2 = random.sample(top50, 2)
            child_genome = crossover(p1[0], p2[0])
            child_genome = mutate(child_genome)
            weight, value = evaluate_fitness(child_genome)
            if weight <= max_weight:
                new_gen.append((child_genome, weight, value))
        
        # Extarcting best children generation
        population_new = new_gen
        if population_new:
            best = max(population_new, key=lambda x:x[2])
            avg = sum(indiv[2] for indiv in population_new) / Pop_size
            # print(f"Gen {g}: Best Value={best[2]} | Weight={best[1]} | Genome={best[0]}")

        else:
            print(f"Warning: No valid individuals in generation {g}. ")
            continue

        best_genome.append(best[0])
        best_weight.append(best[1])
        best_fitness.append(best[2])
        avg_fitness.append(avg)

    max_value = max(best_fitness)
    best_index = best_fitness.index(max_value)
    best_genome_overall = best_genome[best_index]
    selected_boxes = [label for _, label in zip(best_genome_overall, box_labels)]
    best_weight_overall = best_weight[best_index]


    print('******* Best Children Generation *******')            
    print(f"Gen {best_index+1}: Best Value={max_value} | Weight={best_weight_overall} | Genome={selected_boxes}")
    plt.plot(range(1, Generation+1), best_fitness, label='Best Fitness', marker='o')
    plt.plot(range(1, Generation + 1), avg_fitness, label='Avg Fitness', marker='x')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Total Value)')                
    plt.legend()
    plt.grid(True)
    plt.title('Genetic Algorithm: Analysis of Knapsack')
    plt.show()

if __name__ == '__main__':

    evolve()




