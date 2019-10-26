from Genetic_Algorithm import *
from Snake_Game import *

# n_x -> no. of input units
# n_h -> no. of units in hidden layer 1
# n_h2 -> no. of units in hidden layer 2
# n_y -> no. of output units

# The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
def main():
	sol_per_pop = 50
	num_weights = n_x * n_h + n_h * n_h2 + n_h2 * n_y

	# Defining the population size.
	pop_size = (sol_per_pop, num_weights)
	# Creating the initial population.
	new_population = np.random.choice(np.arange(-1, 1, step=0.01), size=pop_size, replace=True)
	num_generations = 100
	fitest_chromosome = np.zeros((num_generations,num_weights))
	fitest_score=np.zeros((num_generations))
	num_parents_mating = 12
	myclock = pygame.time.Clock()
	myclock.tick(1000)
	for generation in range(num_generations):
		print('##############        GENERATION ' + str(generation) + '  ###############')
		# Measuring the fitness of each chromosome in the population.
		fitness = cal_pop_fitness(new_population,myclock)
		print('#######  fittest chromosome in generation ' + str(generation) + ' , is having fitness value:  ',
		      np.max(fitness))
		# fitness2 = cal_pop_fitness(new_population)
		# print('#######  variation in fitness for the fittest chromosomes in generation ' + str(generation) + ' ,  is :  ',
		#       2*(np.max(fitness)-np.max(fitness2))/(np.max(fitness)+np.max(fitness2)))
		# import time
		# time.sleep(3)
		fitest_score[generation]=np.max(fitness)
		# store the fittest chromosome
		fitest_idx = np.argmax(fitness)
		fitest_chromosome[generation,:] = new_population[fitest_idx,]
		#print(fitest_chromosome)
		# Selecting the best parents in the population for mating.
		parents = select_mating_pool(new_population, fitness, num_parents_mating)

		# Generating next generation using crossover.
		offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))

		# Adding some variations to the offsrping using mutation.
		offspring_mutation = mutation(offspring_crossover)

		# Creating the new population based on the parents and offspring.
		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation

	np.save("totoneurone",fitest_chromosome)
	loaded_chromosomes=np.load("totoneurone.npy")

	for i in range(num_generations):
		# display and clock are defined in Snake_game in the original files
		# This creates shadowing of those when they are used as parameters
		# This should be cleaned litlle by little

		#fit = run_game_with_ML(display, myclock, loaded_chromosomes[i],True)
		fit = watch(loaded_chromosomes[i],display,myclock)
		print(f"fittest chromosome of generation {i} has a fitness of: {fit}")

		#print(f"fittest of generation {i} was {fitest_score[i]}")

def watch(chromosome,display,pclock):
	fit=run_game_with_ML(display, pclock, chromosome,True)
	return fit

def watch_chromosome_from_file(filename:str,chromosome_idx:int,display, pclock):
	try:
		loaded_chromosomes=np.load(filename)
		fit=watch(loaded_chromosomes[chromosome_idx],display,pclock)
	except Exception as e:
		print(e)


def only_watch():
	myclock=pygame.time.Clock()
	myclock.tick(10)
	watch_chromosome_from_file("totoneurone.npy",99,display,myclock)

#main()
only_watch()