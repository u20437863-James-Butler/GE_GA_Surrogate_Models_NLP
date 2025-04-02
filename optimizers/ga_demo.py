import random
from TSP_Individual import *
import sys
import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd 
from time import perf_counter
import time
import os
import csv
myStudentNum = 259728
random.seed(myStudentNum)

def readInstance(fName):
    file = open(fName, 'r')
    size = int(file.readline())
    inst = {}
    for i in range(size):
        line=file.readline()
        (myid, x, y) = line.split()
        inst[int(myid)] = (int(x), int(y))
    file.close()
    return inst

def genDists(fName):
    file = open(fName, 'r')
    size = int(file.readline())
    instance = {}
    for i in range(size):
        line=file.readline()
        (myid, x, y) = line.split()
        instance[int(myid)] = (int(x), int(y))
    file.close()
    dfcity= pd.DataFrame.from_dict(instance, orient="index")
    dfcity.rename(columns ={0:"x",1:"y"}, inplace = True)
    flt_dists = distance_matrix(dfcity.values,dfcity.values)
    return (np.rint(flt_dists)).astype(int)

def verify_sol(sol, size):
    # Check that there are no duplicates
    if len(sol) != size:
        return False
    # Check that every city is visited
    for i in range(1,size + 1):
        if i not in sol:
            return False
    return True

def write_result_to_file(file, instance, seed, iterations, pop, heuristic, xprob, mutprob, elites, trunk, bestd, bestinit, runtime, goli):
    file_exists = os.path.isfile(file)
    with open(file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                'Instance', 'Seed', 'Iterations', 'Population', 'Heuristic', 'Crossover Probability', 
                'Mutation Probability', 'Elites', 'Trunk', 'Best Distance', 
                'Best Initialization', 'Runtime', 'Generation of Last Improvement'
            ])
        writer.writerow([instance, seed, iterations, pop, heuristic, xprob, mutprob, elites, trunk, bestd, bestinit, runtime, goli])


class BasicTSP:
    def __init__(self, _fName, _maxIterations, _popSize, _initH, _xoverProb, _mutationRate, _elites, _trunk, _dists):
        """
        Parameters and general variables
        Note not all parameters are currently used, it is up to you to implement how you wish to use them and where
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = int(_popSize)
        self.genSize        = None
        self.initH        = int(_initH)
        self.crossoverProb  = float(_xoverProb)
        self.mutationRate   = float(_mutationRate)
        self.maxIterations  = int(_maxIterations)
        self.fName          = _fName
        self.iteration      = 0
        self.data           = {}
        self.elites        = round(self.popSize * float(_elites))
        self.trunkSize = round(self.popSize * float(_trunk))
        self.dists           = _dists

        "Metrics"
        self.gen_of_last_improvement = 0

        self.readInstance()
        self.bestInitSol = self.initPopulation()


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (cid, x, y) = line.split()
            self.data[int(cid)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating individuals in the initial population
        Either pure random tours (initH=0), or with insertion heuristic (initH=1)
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data,self.initH, self.dists, [])
            if not(self.initH):
                individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        return self.best.getFitness()

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            self.gen_of_last_improvement = self.iteration
    
    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.trunkSize-1) ]
        indB = self.matingPool[ random.randint(0, self.trunkSize-1) ]
        return [indA, indB]

    def crossover(self, indA, indB):
        """
        Executes an order1 crossover and returns the genes for a new individual
        """
        if random.random() > self.crossoverProb:
            child = Individual(self.genSize, self.data, 0, self.dists, random.choice([indA,indB]))
            return child
        
        midP=random.randint(1, self.genSize-2)
        p1 =  indA[0:midP]
        genes = p1 + [i for i in indB if i not in p1]
        child = Individual(self.genSize, self.data, 0,self.dists, genes)
        return child
    
    def crossoverTwoChild(self, indA, indB):
        """
        Executes order1 crossover between two parents and returns the genes for two new individuals
        """
        # If crossover does not take place, reproduce the parents
        if random.random() > self.crossoverProb:
            child1 = Individual(self.genSize, self.data, 0, self.dists, indA)
            child2 = Individual(self.genSize, self.data, 0, self.dists, indB)
            return child1, child2
        # Select a point at which crossover takes place
        midP=random.randint(1, self.genSize-2)
        # Retain the first half of each parent
        p1 =  indA[0:midP]
        p2 = indB[0:midP]
        # Add the ramaining genes from the other respective parent 
        genes1 = p1 + [i for i in indB if i not in p1]
        genes2 = p2 + [i for i in indA if i not in p2]
        # Initialize the children with the resulting genes and return them
        child1 = Individual(self.genSize, self.data, 0, self.dists, genes1)
        child2 = Individual(self.genSize, self.data, 0, self.dists, genes2)
        return child1, child2
    
    def mutation(self, ind):
        """
        Mutate an individual by swapping two cities with certain probability (i.e., mutation rate)
        This mutator performs recipricol exchange
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation.
        Uses truncation selection
        Note we are only storing the gene values and fitness of every 
        chromosome in prev pop
        """
        mybest = self.population[0:self.trunkSize]
        best_fits = [i.getFitness() for i in mybest]
        worst_fit=max(best_fits)
        worst_idx = best_fits.index(worst_fit)
        for i in range(self.trunkSize,self.popSize):
            if self.population[i].getFitness() < worst_fit:
                mybest[worst_idx] = self.population[i]
                best_fits[worst_idx] = self.population[i].getFitness()
                worst_fit = max(best_fits)
                worst_idx = best_fits.index(worst_fit)
        self.matingPool = [[]+ind_i.genes for ind_i in mybest]
                
        ## Add truncation to mating pool, separately store elite best
        if self.elites < self.trunkSize:
            x = self.elites
        else:
            x = self.trunkSize
        elite_sols = mybest[0:x]
        if x:
            elite_fits = [i.getFitness() for i in elite_sols]
            worst_fit = max(elite_fits)
            worst_idx = elite_fits.index(worst_fit)
            for i in range(x,len(mybest)):
                if mybest[i].getFitness() < worst_fit:
                    elite_sols[worst_idx] = mybest[i]
                    elite_fits[worst_idx] = mybest[i].getFitness()
                    worst_fit = max(elite_fits)
                    worst_idx = elite_fits.index(worst_fit)
        return elite_sols


    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        if Use_grimes_crossover:
            for i in range(self.elites, self.popSize):
                [ind1, ind2] = self.randomSelection() # select from mating pool
                child = self.crossover(ind1, ind2)
                self.mutation(child)
                child.computeFitness()
                self.updateBest(child)
                self.population[i] = child
        else:
            for i in range(self.elites, self.popSize, 2):
                [ind1, ind2] = self.randomSelection() # select from mating pool
                child1, child2 = self.crossoverTwoChild(ind1, ind2)
                self.mutation(child1)
                self.mutation(child2)
                child1.computeFitness()
                child2.computeFitness()
                self.updateBest(child1)
                self.updateBest(child2)
                self.population[i] = child1
                self.population[i+1] = child2


    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        elite_sols = self.updateMatingPool()
        # print()
        self.population[:self.elites] = elite_sols
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1
        return self.best.getFitness(), self.bestInitSol, self.best.genes
    
    # def report_best_dist(self):
    #     """
    #     Report best distance so far and gen count. use when improvement
    #     """
    #     # Plot on double step line chart for sample run with given seed
    #     pass


Use_grimes_crossover = False
Gen_result_files = False
resfile_name = 'result_sample1.csv'
# store_gen_report = False

def main():
    if len(sys.argv) < 9:
        print ("Error - Incorrect input")
        print ("Expecting python TSP.py [instance] [number of runs] [number of iterations] [population size]", 
                "[initialisation method] [xover prob] [mutate prob] [elitism] [truncation] [student number]")
        sys.exit(0)
    '''
    Reading in parameters, but it is up to you to implement what needs implementing
    TO DO:
    1/ Adapt to produce 2 children from each crossover
    2/ Add solution checker of final GA solution in each run to verify it is correct
    3/ Add code for metrics
        * distance (best and average)
        * runtime ()
        * iteration num of last improvement
        * iteration num of largest improvement
        * initial solution quality (average of best in each)
        * average fitness per generation (for sample run)
        * solution improvements vs iterations (for sample runs)
    '''
    _, inst, nRuns, nIters, pop, initH, pC, pM, el, tr = sys.argv
    d = genDists(inst)
    nRuns = int(nRuns)
    bestRuntime = bestInitDist = bestDist = avgDist = avgInitDist = avgRuntime = avgGenOLI = 0
    for i in range(0,nRuns):
        curr_seed = myStudentNum+i*100
        random.seed(curr_seed)
        print('Iteration {iter}, for seed {s}'.format(iter = i, s = curr_seed))
        start_time = time.time()
        ga = BasicTSP(inst, nIters, pop, initH, pC, pM, el, tr, d)
        dist, distInit, sol = ga.search()
        runtime = time.time() - start_time
        # Check solution validity
        print('Solution is valid' if verify_sol(sol, len(d)) else 'Solution is invalid')
        gen_of_last_improvement = ga.gen_of_last_improvement
        # Report run measures: Best distance, best initial distance, runtime, and generation of last improvement
        print('Best distance found:{d1}\tBest initial distance:{d2}\tRuntime:{r}\tGeneration of last improvement:{g}\n'.format(d1 = dist, d2 = distInit, r = runtime, g = gen_of_last_improvement))
        if Gen_result_files:
            write_result_to_file(resfile_name, inst, curr_seed, nIters, pop, initH, pC, pM, el, tr, dist, distInit, runtime, gen_of_last_improvement)
        # Update averages
        avgDist += dist
        avgInitDist += distInit
        avgRuntime += runtime
        avgGenOLI += gen_of_last_improvement
        # Update best results
        if dist < bestDist or bestDist == 0:
            bestDist = dist
            bestSol = sol
        if runtime < bestRuntime or bestRuntime == 0:
            bestRuntime = runtime
        if distInit < bestInitDist or bestInitDist == 0:
            bestInitDist = distInit
    print("Results")
    # Print out the results collected over all the runs, and display they best and average for each measure
    print(
    'Best Distance found:{bd}\tAverage Best Distance Found:{abd}\tBest Runtime:{br}\tAverage Runtimes:{ar}\nBest Initial Distance:{bid}\tAverage Best Initial Distance:{abid}\tAverage Generation of Last Improvement:{agli}'.
    format(bd=bestDist, abd=avgDist/nRuns, br=bestRuntime, ar=avgRuntime/nRuns, bid=bestInitDist, abid=avgInitDist/nRuns, agli=avgGenOLI/nRuns)
    )
           
main()