

"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""
import math
import random
import time
from inspect import k

from Individual import *
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

myStudentNum = 182864 # Replace 12345 with your student number
random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, config = 1):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}
        self.config         = config
        self.fitnessLog = []

        self.readInstance()
        if self.config  >= 1 and self.config <= 6:
            self.initPopulation()
        else:
            self.initPopulation_with_heuristic()


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())

    def initPopulation_with_heuristic(self):
        self.population = []
        for i in range(0, self.popSize):
            instance = self.data
            cities = list(instance.keys())
            cIndex = random.randint(0, len(instance) - 1)

            tCost = 0

            solution = [cities[cIndex]]

            del cities[cIndex]

            current_city = solution[0]
            while len(cities) > 0:
                bCity = cities[0]
                bCost = self.euclideanDistance(instance[current_city], instance[bCity])
                bIndex = 0
                #        print(bCity,bCost)

                for city_index in range(1, len(cities)):
                    city = cities[city_index]
                    cost = self.euclideanDistance(instance[current_city], instance[city])
                    #            print(cities[city_index], "Cost: ",cost)
                    if bCost > cost:
                        bCost = cost
                        bCity = city
                        bIndex = city_index
                tCost += bCost
                current_city = bCity
                solution.append(current_city)
                del cities[bIndex]
            tCost += self.euclideanDistance(instance[current_city], instance[solution[0]])
            individual = Individual(self.genSize, self.data)
            individual.setGene(solution)
            individual.computeFitness()
            self.updateBest(individual)
            self.population.append(individual)

        self.best = self.population[0].copy()

        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print("Best initial sol: ", self.best.getFitness())

    # Used in initPopulation_with_heuristic method
    def euclideanDistance(self, d1, d2):
        return math.sqrt( (d1[0]-d2[0])**2 + (d1[1]-d2[1])**2 )

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            #print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def stochasticUniversalSampling(self):
        '''
        1. Let F be the sum of the fitness values of all chromosomes in the population.
        2. Let N be the number of parents to select.
        3. Compute the distance P between successive points: P = F/N.
        4. Generate a random number between 0 and P as the starting point for the ruler. The
            ruler has N equally spaced points, each P distance apart.
        5. Assign each chromosome a range equal in length to its fitness and a starting point that
            is after the end point of the previous chromosome (e.g. first chromosome 0-1.53, 2nd
            chromosome 1.54-2.26, 3rd chromosome 2.27-3.42, etc).
        6. Select the chromosomes whose range contains a marker (note that a chromosome
            may have 2 markers in which case it is chosen twice).
        '''

        # 1. Let F be the sum of the fitness values of all chromosomes in the population.

        F = 0
        weight = 100000000000
        tmp = 0
        ruler = []
        for i in self.matingPool:
            i.computeFitness()
            # As best fitness is the lower the number in TSP, we are performing inversion and multiplying it with a large
            # number to avoid handling incredibly small calculation
            fitness = (1/i.getFitness()) * weight
            F = F + fitness
            ruler.append([tmp, F, i])
            tmp = tmp + fitness

        # 2. Let N be the number of parents to select.
        N = 2

        # 3. Compute the distance P between successive points: P = F/N.
        P = F/N

        # 4. Generate a random number between 0 and P as the starting point for the ruler. The ruler has N equally spaced points, each P distance apart.
        selection_pointerA = random.uniform(0, P)
        selection_pointerB = selection_pointerA+P
        '''
        5. Assign each chromosome a range equal in length to its fitness and a starting point that is after the end point of the previous chromosome 
            (e.g. first chromosome 0-1.53, 2nd chromosome 1.54-2.26, 3rd chromosome 2.27-3.42, etc).
        '''
        mating_partners = []
        for element in ruler:
            [start, end, ind] = element
            if selection_pointerA >= start and selection_pointerA < end:
                mating_partners.append(ind)
            if selection_pointerB >= start and selection_pointerB < end:
                mating_partners.append(ind)
        return mating_partners


    def uniformCrossover(self, indA, indB):
        # Contains randomly selected elements from the gene
        tmp = random.sample(indA.genes, random.randint(0, self.genSize-1))

        # Generate blank gene for child
        childA = np.zeros(self.genSize, dtype=int)
        # Randomly selecting positions that will no longer change for childA.
        for rand_i in tmp:
            childA[indA.genes.index(rand_i)] = rand_i
        i = 0
        while i < len(childA):
            if childA[i] == 0:
                for element_in_indB in indB.genes:
                    if (element_in_indB in childA):
                        pass
                    else:
                        childA[i] = element_in_indB
            i += 1
        return childA.tolist()

    def pmxCrossover(self, indA, indB):


        start = np.random.randint(0,self.genSize - 2)
        end = np.random.randint(start+1, self.genSize -1)
        #print(start)
        #print(end)

        indA = np.array(indA.genes)
        indB = np.array(indB.genes)

        slice1 = indA[start:end]
        slice2 = indB[start:end]

        # Swapping the random selection in blank individual gene
        newIndA = np.zeros(self.genSize, dtype=int)
        newIndB = np.zeros(self.genSize, dtype=int)
        i = start
        while (i < end):
            newIndA[i] = indB[i]
            newIndB[i] = indA[i]
            i += 1


        # Adding no conflict element in the new individual for both individuals
        i = 0
        while i < len(indA):
            if not len(np.where(newIndA == indA[i])[0] == 0) and (i < start or i >= end):
                newIndA[i] = indA[i]
            if not len(np.where(newIndB == indB[i])[0] == 0) and (i < start or i >= end):
                newIndB[i] = indB[i]
            i += 1
        # Adding mapping sequence
        i = 0
        mapping = []

        while i < len(slice1):
            if (len(np.where(slice1 == slice2[i])[0]) == 1) and (len(np.where(slice2 == slice1[i])[0]) == 1):

                    index = np.where(slice1 == slice2[i])[0][0]
                    set_mapping = set()
                    set_mapping.add(slice1[i])
                    set_mapping.add(slice2[i])
                    set_mapping.add(slice2[index])
                    set_mapping.add(slice1[index])
                    if len(set_mapping) == 3:
                        (a, b, c) = set_mapping
                        mapping.append((a, b, c))
                        i += 1
                    else:
                        i += 1

            else:
                if (len(np.where(slice1 == slice2[i])[0]) == 0) and (len(np.where(slice2 == slice1[i])[0]) == 0):
                    mapping.append((slice1[i], slice2[i]))
            i += 1

        mapping.reverse()

        # Filling the missing conflict elements in the gene
        #*****************************************
        i = 0
        while i < len(newIndA):
            if newIndA[i] == 0:
                for m in mapping:
                    if(len(m) == 2):
                        (B, A) = m
                        if(A == indA[i]):
                            newIndA[i] = B
                    else:
                        (A, B, C) = m
                        index = np.where(indA == A)[0][0]
                        newIndA[index] = C
            i+=1

        #Resuing the variable and updaing the gene for newIndB
        i = 0
        while i < len(newIndB):
            if newIndB[i] == 0:
                for m in mapping:
                    if(len(m) == 2):
                        (A, B) = m
                        if(A == indB[i]):
                            newIndB[i] = B
                    else:
                        (A, B, C) = m
                        index = np.where(indB == C)[0][0]
                        newIndB[index] = A
            i += 1



        return  newIndA.tolist()


    def pmxCrossover1(self, indA, indB):

        #start = np.random.randint(0,self.genSize - 2)
        #end = np.random.randint(start+1, self.genSize -1)
        #print(start)
        #print(end)
        start = 3
        end = 6
        indA = np.array(indA.genes)
        indB = np.array(indB.genes)

        slice1 = indA[start:end]
        slice2 = indB[start:end]

        # Swapping the random selection in blank individual gene
        newIndA = np.zeros(self.genSize, dtype=int)
        newIndB = np.zeros(self.genSize, dtype=int)
        i = start
        while (i < end):
            newIndA[i] = indB[i]
            newIndB[i] = indA[i]
            i += 1


        # Adding no conflict element in the new individual for both individuals
        i = 0
        while i < len(indA):
            if not len(np.where(newIndA == indA[i])[0] == 0) and (i < start or i >= end):
                newIndA[i] = indA[i]
            if not len(np.where(newIndB == indB[i])[0] == 0) and (i < start or i >= end):
                newIndB[i] = indB[i]
            i += 1
        # Adding mapping sequence
        i = 0
        mapping = []

        for s in range(0, len(slice1)):
            if len(np.where(slice2 == slice1[s])[0]) == 0 and len(np.where(slice1 == slice2[s])[0]) == 0:
                mapping.append((slice2[s], slice1[s]))
            else:
                index = np.where(slice2 == slice1[s])[0][0]
                tmp = []
                tmp.append(slice1[s])
                tmp.append(slice2[s])
                tmp.append(slice2[index])
                tmp.append(slice1)[index]
                for element in tmp:
                    if tmp.count(element) != 1:
                        while element in tmp: tmp.remove(element)
                (a,b) = tmp
                mapping.append((a,b))


        print(mapping)



        return  newIndA.tolist()


    def reciprocalExchangeMutation(self, ind):

        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize - 2)
        indexB = random.randint(indexA, self.genSize - 1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)


    def inversionMutation(self, ind):

        if random.random() > self.mutationRate:
            return

        indexA = random.randint(0, self.genSize - 2)
        indexB = random.randint(indexA+1, self.genSize - 1)
        ind_gene = np.array(ind.genes)
        slice1 = ind_gene[indexA:indexB]
        slice1 = slice1.tolist()
        slice1.reverse()
        for i in slice1:
            ind_gene[indexA] = i
            indexA += 1
        ind.computeFitness()
        self.updateBest(ind)

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy() )

    def filter(self, gene):
        data = np.array(gene)
        j = np.where(data == 0)[0]
        t1 = []
        for x in range(1, 129):
            if not x in data:
                t1.append((x))

        for xx in j:
            data[xx] = t1.pop()

        double = []
        for xxx in data:
            if len(np.where(data == xxx)[0]) > 1:
                index = np.where(data == xxx)[0]
                data[index[0]] = t1[0]
                double.append(xxx)

        return data.tolist()


    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(0, self.popSize):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            if self.config == 1:
                [indA, indB] = self.randomSelection()
                gene = self.uniformCrossover(indA,indB)
                individual = Individual(self.genSize, self.data)
                individual.setGene(gene)
                self.inversionMutation(individual)
                individual.computeFitness()
                self.updateBest(individual)
                self.population.append(individual)

            elif self.config == 2:
                [indA, indB] = self.randomSelection()
                gene = self.pmxCrossover1(indA, indB)
                #gene = self.filter(gene)
                individual = Individual(self.genSize, self.data)
                individual.setGene(gene)
                self.inversionMutation(individual)
                individual.computeFitness()
                self.updateBest(individual)
                self.population.append(individual)

            elif self.config == 3:
                [indA, indB] = self.stochasticUniversalSampling()
                gene = self.uniformCrossover(indA, indB)
                individual = Individual(self.genSize, self.data)
                individual.setGene(gene)
                individual.computeFitness()
                self.reciprocalExchangeMutation(individual)
                self.updateBest(individual)
                self.population.append(individual)

            elif self.config == 4:
                [indA, indB] = self.stochasticUniversalSampling()
                gene = self.pmxCrossover(indA, indB)
                gene = self.filter(gene)
                individual = Individual(self.genSize, self.data)
                individual.setGene(gene)
                self.reciprocalExchangeMutation(individual)
                individual.computeFitness()
                self.updateBest(individual)
                self.population.append(individual)
            elif self.config == 5:
                self.population = []
                self.initPopulation()
                [indA, indB] = self.stochasticUniversalSampling()
                gene = self.pmxCrossover(indA, indB)
                gene = self.filter(gene)
                individual = Individual(self.genSize, self.data)
                individual.setGene(gene)
                self.inversionMutation(individual)
                individual.computeFitness()
                self.updateBest(individual)
                self.population.append(individual)
            elif self.config == 6:
                [indA, indB] = self.stochasticUniversalSampling()
                gene = self.uniformCrossover(indA, indB)
                individual = Individual(self.genSize, self.data)
                individual.setGene(gene)
                self.inversionMutation(individual)
                individual.computeFitness()
                self.updateBest(individual)
                self.population.append(individual)
            elif self.config == 7:

                [indA, indB] = self.stochasticUniversalSampling()
                gene = self.pmxCrossover(indA, indB)
                gene = self.filter(gene)
                individual = Individual(self.genSize, self.data)
                individual.setGene(gene)
                self.reciprocalExchangeMutation(individual)
                individual.computeFitness()
                self.updateBest(individual)
                self.population.append(individual)
            elif self.config == 8:

                [indA, indB] = self.stochasticUniversalSampling()
                gene = self.uniformCrossover(indA, indB)
                individual = Individual(self.genSize, self.data)
                individual.setGene(gene)
                self.inversionMutation(individual)
                individual.computeFitness()
                self.updateBest(individual)
                self.population.append(individual)

            else:
                print("Config error")

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """

        self.iteration = 0
        start_time = time.time()
        print("Started on: ", time.asctime(time.localtime(start_time)))
        while self.iteration < self.maxIterations:
            print("Iteration -->",self.iteration)
            self.GAStep()
            print("Current best ==>",self.best.getFitness())
            self.fitnessLog.append(self.best.getFitness())
            self.iteration += 1

        plt.xlabel('No of Itrerations')
        plt.ylabel('Fitness')
        plt.plot(self.fitnessLog)
        plt.show()
        print("Total iterations: ", self.iteration)
        print("Best Solution: ", self.best.getFitness())
        end_time = time.time()
        print("Ended on: ", time.asctime(time.localtime(end_time)))
        print("Time required: ", end_time-start_time)
        tmp = self.fitnessLog
        self.fitnessLog = []
        return tmp

    def tester(self):
        g1 = [3,4,8,2,7,1,6,5]
        g2 = [4,2,5,1,6,8,3,7]

        individualA = Individual(8, self.data)
        individualA.setGene(g1)
        individualA.computeFitness()

        individualB = Individual(8, self.data)
        individualB.setGene(g2)
        individualB.computeFitness()

        child = self.pmxCrossover1(individualA,individualB)
        print(child)





'''if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)
'''

#problem_file = sys.argv[1]
# def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, config = 1):
problem_file = "inst-19.tsp"
#ga = BasicTSP(problem_file, 150, 0.1, 500, 8)
#ga.search()
allRestults = []
results = []
print("Config 3 ****************************************")
for i in range(0, 5):
    ga = BasicTSP(problem_file, 150, 0.1, 500, 3)
    results.append(ga.search())
    print("*************************")
print("Results")
print(results)
allRestults.append(results)

print("Config 6 ****************************************")
for i in range(0, 5):
    ga = BasicTSP(problem_file, 150, 0.1, 500, 6)
    results.append(ga.search())
    print("*************************")
print("Results")
print(results)
allRestults.append(results)


print("Config 8 ****************************************")
for i in range(0, 5):
    ga = BasicTSP(problem_file, 150, 0.1, 500, 8)
    results.append(ga.search())
    print("*************************")
print("Results")
print(results)
allRestults.append(results)
#ga.tester()
#ga.pmxCrossover1()
#ga.tester()




