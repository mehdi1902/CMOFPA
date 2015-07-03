# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:40:21 2015

@author: 8
"""
import numpy as np
from numpy.random import gamma, normal, rand, randint
from math import pi, exp, sqrt, sin
from copy import deepcopy

class Individual():
    def __init__(self, chrom=np.array([]), fitness=0):
        self.chrom = chrom
        self.fitness = fitness
		
		
class Population():
    def __init__(self, lchrom):
        self.lchrom = lchrom 

        self.gen = 0
        self.sumFitness = 0
        self.max = 0
        self.min = 0
        self.avg = 0
        
        self.np = []    #new population
        self.op = []    #old population
        
class Problem():
    def __init__(self, dimension, data_range, mu=10, landa=10,
                    pSwitch=.8, max_gen=10000, prob_type='ackley',
                    beta=1.5, step=0.075, search_type = 'levy'):
        
        '''
        sigma_type : we have 3 types of ES
        1 : 1 sigma for all dimension
        2 : n sigma for n dimension
        3 : n sigma + n alpha for n dimension
        '''        
        
        self.pop = Population(lchrom=dimension)
        
#        np.random.seed(self.random_seed)
        self.lchrom = self.pop.lchrom
        self.g_star = Individual()
        self.range = data_range
        self.max_gen = max_gen
        
        self.mu = mu
        self.landa = landa
        
        self.pSwitch = pSwitch
        self.search_type = 'levy'
        
        self.beta = beta
        self.step = step
        
        self.prob_type = prob_type        
        
        self.best_fitness = 10e10
        self.best_gen = 0
        self.best_chrom = Individual()
        
    
    def initialize(self):
        for i in range(self.mu):
            chrom = rand(self.lchrom)*self.range*2 - self.range
            
            fitness = self.evaluation(chrom)
            individual = Individual(chrom, fitness)
            self.pop.np.append(individual)
#            print self.pop.np[len(self.pop.np)-1].chrom
#            print self.pop.np[len(self.pop.np)-1].fitness
                
                
    def evaluation(self, chrom):
        if self.prob_type=='ackley':
            sum1 = np.sum(chrom**2)
            sum2 = np.sum(np.cos(2*pi*chrom))
            n = len(chrom)
            return -20 * exp(-.2 * sqrt(sum1/n)) - exp(sum2/n) + 20 + exp(1)
#        elif self.prob_type=='ZDT1':
#            g = 


    def find_best_current(self):
        pop = [ind for ind in self.pop.np]
        pop.sort(key=lambda x:x.fitness, reverse=True)
        self.g_star = deepcopy(pop[0])
#        print 'here'
#        print self.g_star.fitness
        

    def generate(self):
        for ind in self.pop.np:
#            print prob.pop.np[0].chrom
            if self.flip(self.pSwitch):
                self.global_search(ind)
            else:
                self.local_search(ind)
            
      
    def global_search(self, ind):
        chrom = deepcopy(ind.chrom)
        L = self.levy()

        chrom += self.step * L * (self.g_star.chrom - chrom)
        
        fitness = self.evaluation(chrom)
        if fitness<ind.fitness:
            ind.fitness = fitness
            ind.chrom = chrom
        
    
    def local_search(self, ind):
        eps = rand()
        
        chrom1 = self.pop.np[int(rand()*self.mu)].chrom
        chrom2 = self.pop.np[int(rand()*self.mu)].chrom
        chrom = deepcopy(ind.chrom)
        
        chrom += eps * (chrom1-chrom2)
        fitness = self.evaluation(chrom)
        if fitness<ind.fitness:
            ind.fitness = fitness
            ind.chrom = chrom        

        
#    def search_dist(self):
#        if self.search_type=='levy':
#            return levy()
        

    def levy(self):
#        landa = float(landa)

        s = []
        for i in range(self.lchrom):
            up = gamma(1+self.beta) * (pi*self.beta/2.)
            down = gamma((1+self.beta)/2.) * 2**((self.beta-1)/2.)
            sigma = (up / down) ** (1/self.beta)
            
            U = normal(0, sigma)
            V = normal(0, 1)
        
            s.append(U / abs(V)**(1/self.beta))
        return np.array(s)
            
          
          
    def gen_statistics(self):
        F = [ind.fitness for ind in self.pop.np]
        self.pop.max = max(F)
        self.pop.min = min(F)
#        self.pop.avg = np.average(np.array(F))
        
        for i in self.pop.np:
            fitness = i.fitness
            if fitness<self.best_fitness:
                self.best_fitness = fitness
                self.best_gen = self.pop.gen
                self.best_chrom = i.chrom
                
    def report(self):
        f = open('out', 'a')
        print ('%i\t%.10f\t%i\t%.10f'% (self.pop.gen, self.pop.min, self.best_gen, self.best_fitness))
        f.write('%i\t%.10f\t%i\t%.10f\n'% (self.pop.gen, self.pop.min, self.best_gen, self.best_fitness))
        for ind in self.pop.np:
            for i in ind.chrom:
                f.write('%.2f\t'%i)
            f.write('\t')
            f.write('%.2f\n'%ind.fitness)
##            print ind.chrom
##            print ind.fitness
        f.writelines('--------------------------------\n')
           
           
    def final_report(self):
        f = open('report', 'a+')
        f.write('dimension\t%i\n'%prob.lchrom)
        f.write('data range\t(-%.3f,%.3f)\n'%(prob.range, prob.range))
        f.write('max gen\t\t%i\n'%prob.max_gen)
        f.write('mu\t\t\t%i\n'%prob.mu)
        f.write('landa\t\t%i\n'%prob.landa)
        f.write('problem\t\t%s\n'%prob.prob_type)
        f.write('seed\t\t%i\n'%seed)
        f.write('Result:\n')
        f.write('\tbest fitness\t%.10f\n'%prob.best_fitness)
        f.write('\tbest generation\t%i\n'%prob.best_gen)
        f.write('\tbest chromosome\t')
        for c in prob.best_chrom:
            f.write('%.10f  '%c)
        f.writelines('-----------------------------------------------------\n')
        f.close()
        
        
    def flip(self, p):
        if p>np.random.rand():
            return True
        else:
            return False
                
                
if __name__=='__main__':
    
    f = open('out', 'w+')
    import random

    best_fitness = 10
    cnt = 0
    while best_fitness>1e-15:
        seed = int(random.random()*999)

#        np.random.seed(seed)
        prob = Problem(dimension=30, data_range=30, max_gen=100000,
                       mu=25, landa=35)
        prob.initialize()
#        print prob.pop.np[0].chrom
#        prob.pop.np = prob.pop.op
        prob.find_best_current()
        
        prob.gen_statistics()    
        prob.report()

#        print prob.pop.np[0].chrom
#        print prob.pop.np[0].fitness
        
        while prob.pop.gen<prob.max_gen:
            prob.pop.gen += 1
            
            prob.generate()
            prob.gen_statistics()
            prob.find_best_current()
            
            if prob.pop.gen%100==0:
                prob.report()
            prob.pop.op = prob.pop.np
    #        print '-----------------------------'
        f.close()
        
        print 'solution :'
        for c in prob.best_chrom:
            print '%.15f  '%c,
        print 'fitness :  ', prob.best_fitness
        print 'seed :   ', seed
    #    print '-----------------------------\n'
        
        prob.final_report()
                    
        best_fitness = prob.best_fitness
    		
        ### 0 improvement at beginning of each epoch
        ### toole gam 0 nashe