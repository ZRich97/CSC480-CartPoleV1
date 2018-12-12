# Zachary Richardson
# CSC 480 Final Project
# OpenAI CartPole - Genetic Algorithm / Neural Network Solver
import math, random, bisect
import gym
import numpy as np
from matplotlib import pyplot

class NeuralNetwork:

    def __init__(self, space):
        # Calculated fitness of this child
        self.fitness = 0
        # Values for in/out-put
        self.space = space
        # Weights for NN
        self.weights = []
        # Biases for NN
        self.biases = []
        # Calculate weights and biases 
        # TODO: improve setup of weights/biases
        for i in range(len(self.space) - 1):
            self.weights.append( np.random.uniform(low=-1, high=1, size=(self.space[i], self.space[i+1])).tolist() )
            self.biases.append( np.random.uniform(low=-1, high=1, size=(self.space[i+1])).tolist())

    # Get decision from NN
    # TODO: Better understanding of how/why bias and weights interact
    def getOutput(self, input):
        output = np.reshape(np.matmul(input, self.weights[0]) + self.biases[0], (self.space[1]))
        return np.argmax(self.sigmoid(output))

    # Take input and squash it between 0 and 1
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

class Population:

    def __init__(self, populationCount, mutationRate, space):
        self.space = space
        self.pop_count = populationCount
        self.mutate_rate = mutationRate
        self.population = [NeuralNetwork(space) for i in range(self.pop_count)]

    # Creates a child by mutating two parents
    def createChild(self, node1, node2):        
        child = NeuralNetwork(self.space)
        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    # Mutate weight? Or just slice?
                    if random.random() < self.mutate_rate:
                        child.weights[i][j][k] = random.uniform(-1, 1)
                    else:
                        child.weights[i][j][k] = (node1.weights[i][j][k] + node2.weights[i][j][k]) / 2.0
        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                # Mutate bias? Or just slice?
                if random.random() < self.mutate_rate:
                    child.biases[i][j] = random.uniform(-1, 1)
                else:
                    child.biases[i][j] = (node1.biases[i][j] + node2.biases[i][j]) / 2.0
        return child

    # Creates a generation via evolution
    def createNewGeneration(self):       
        nextGen = []
        fitnessSum = [0]
        fit = 0
        for i in range(len(self.population)):
            fit += self.population[i].fitness
            fitnessSum.append(fitnessSum[i]+self.population[i].fitness)
        while(len(nextGen) < self.pop_count):
            # Determine random bisect point
            r1 = random.uniform(0, fitnessSum[len(fitnessSum) - 1] )
            r2 = random.uniform(0, fitnessSum[len(fitnessSum) - 1] )
            # Perform bisect
            node1 = self.population[bisect.bisect_right(fitnessSum, r1) - 1]
            node2 = self.population[bisect.bisect_right(fitnessSum, r2) - 1]
            # Make child (with possible mutation)
            nextGen.append(self.createChild(node1, node2))
        self.population = nextGen


def main():
    # Variables for algorithm tweaking
    GENERATIONS = 10
    STEPS = 500
    POPULATION = 200
    MUTATION_RATE = 0.01
    # Initialize OpenAI gym environment
    env = gym.make('CartPole-v1')
    cur_env = env.reset()
    obs_space = env.observation_space.shape[0] # 4
    act_space = env.action_space.n # 2
    generations = []
    # Initialize population for genetic algorithm 
    pop = Population(POPULATION, MUTATION_RATE, [obs_space,  act_space])
    # Iterate through generations
    for gen in range(GENERATIONS):
        sumFit = 0.0
        # For child in population
        for node in pop.population:
            # Make child's moves until environment ends
            for _ in range(STEPS):
                # Render CartPole
                env.render()
                action = node.getOutput(cur_env)
                # Perform action
                cur_env, reward, done, _ = env.step(action)
                # Increment reward for child
                node.fitness  += reward
                # If environment done, end actions
                if done:
                    cur_env = env.reset()
                    break
            # Accumulate reward for later stats        
            sumFit += node.fitness
        # Calculate average and print stats
        genAvgFit = sumFit / pop.pop_count
        print("Generation: %4d |  Average Fitness: %2.0f" % (gen + 1, genAvgFit))
        pop.createNewGeneration()
        generations.append(genAvgFit)
    env.close()
    # Generate graph
    pyplot.plot(generations)
    pyplot.xlabel('Generation')
    pyplot.ylabel('Avg. Score (Timesteps)')
    pyplot.grid()
    pyplot.show()
    print("...DONE!")

if __name__ == "__main__":
    main()
