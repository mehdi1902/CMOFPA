# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:40:21 2015

@author: 8
"""
import numpy as np
from numpy.random import gamma, normal, rand, randint
from math import pi, exp, sqrt, sin
from copy import deepcopy
import matplotlib.pyplot as plt
from chaos import *
plt.ioff()


class Individual():
    def __init__(self, chrom=np.array([]), fitness=0, w=.5):
        self.chrom = chrom
        self.fitness = fitness
        self.w = w
		
		
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
    def __init__(self, dimension, lb, ub, mu=10, landa=10,
                    pSwitch=.8, max_gen=10000, prob_type='ackley',
                    beta=1.5, step=0.075, search_type = 'levy',
                    map_type='logistic', 
                    chaos_init=False, chaos_local=False):
        
        '''
        sigma_type : we have 3 types of ES
        1 : 1 sigma for all dimension
        2 : n sigma for n dimension
        3 : n sigma + n alpha for n dimension
        '''        
        
        self.pop = Population(lchrom=dimension)
        
        self.lchrom = self.pop.lchrom
        self.g_star = Individual()
        self.lb = np.array(lb)
        self.ub = np.array(ub)

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
        
        self.zeros = np.zeros(self.lchrom)
        self.ones = np.ones(self.lchrom)
        
        self.map_type = map_type
        self.map = Chaos_map(map_type)
        self.chaos_init = chaos_init
        self.chaos_local = chaos_local
#        self.w = w
        self.max = np.array([])
#        self.chaotic_seq = self.map.logistic_map(self.max_gen*self.mu, .01, 3.6)
        self.chaotic_seq = self.map.logistic_map(self.max_gen*self.mu, .01, 3.6)
        
        self.L = []
    
    def initialize(self):
        if self.chaos_init:
            W = np.array(self.map.logistic_map(self.mu, .01, 4))
        else:
            W = rand(self.mu)
        
#        W = W/np.max(W)
#        W = (np.array(range(self.mu))+1)/float(self.mu+2)
#        W = np.ones(self.mu)*self.w
        
#        X = self.map.logistic_map(self.lchrom*self.mu, .01, 4)
        
        for i in range(self.mu):
            if self.chaos_init==True:    
                chrom = np.array(self.map.logistic_map(self.lchrom, rand()/10., rand()*.5+3.5))
#                chrom = X[i*self.lchrom:(i+1)*self.lchrom]
                chrom = chrom*(self.ub-self.lb) + self.lb
            else:
            
                chrom = rand(self.lchrom)*(self.ub-self.lb) + self.lb
    
            ind = Individual()
            ind.chrom = chrom
            ind.w = W[i]
            ind.fitness = self.evaluation(ind)
            self.pop.np.append(ind)
#            print self.pop.np[len(self.pop.np)-1].chrom
#            print self.pop.np[len(self.pop.np)-1].fitness
                
                
    def evaluation(self, ind):
        chrom = ind.chrom
        w = ind.w
        if self.prob_type=='ackley':
            sum1 = np.sum(chrom**2)
            sum2 = np.sum(np.cos(2*pi*chrom))
            n = len(chrom)
            return -20 * exp(-.2 * sqrt(sum1/n)) - exp(sum2/n) + 20 + exp(1)
            
        elif self.prob_type[:-1]=='ZDT':
            g = 1 + (9*np.sum(chrom[1:]) / float(len(chrom[1:])))
#            g = 1
            f1 = chrom[0]
            
            if self.prob_type[-1]=='1':
                f2 = g * (1 - sqrt(f1/g))
            elif self.prob_type[-1]=='2':
                f2 = g * (1 - (f1/g))**2
            elif self.prob_type[-1]=='3':
                f2 = g * (1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1))
            
            return w*f1 + (1-w)*f2
                
        elif self.prob_type=='LZ':
            x1 = chrom[0]
            J1 = chrom[1::2]
            J2 = chrom[2::2]
            
            t = np.array([sin(6*pi*x1 + j*pi/self.lchrom) for j in range(self.lchrom)])
            t1 = t[1::2]
            t2 = t[3::2]
            
            f1 = x1 + 2/float(len(J1)) * np.sum((J1-t1)**2)
            f2 = 1 - sqrt(x1) + 2/float(len(J2)) * np.sum((J2-t2)**2)
            
            return w*f1 + (1-w)*f2

    
    def pareto(self):
        X = []
        Y = []
        
        if self.prob_type[:-1]=='ZDT':
            for ind in self.pop.np:
                chrom = ind.chrom
                g = 1 + (9*np.sum(chrom[1:]) / float(len(chrom[1:])))
    #            g = 1
                f1 = chrom[0]
                
                if self.prob_type[-1]=='1':
                    f2 = g * (1 - sqrt(f1/g))
                elif self.prob_type[-1]=='2':
                    f2 = g * (1 - (f1/g))**2
                elif self.prob_type[-1]=='3':
                    f2 = g * (1 - sqrt(f1/g) - (f1/g)*sin(10*pi*f1))        

                X.append(f1)
                Y.append(f2)
                    
        
            x = [i/31. for i in range(30)]
            if self.prob_type[-1]=='1':
                y = [1-sqrt(i) for i in x]
            elif self.prob_type[-1]=='2':
                y = [(1-i)**2 for i in x]
            elif self.prob_type[-1]=='3':
                y = [(1-sqrt(i)-i*sin(10*pi*i))**2 for i in x]

        elif self.prob_type=='LZ':
            for ind in self.pop.np:
                chrom = ind.chrom
                
                x1 = chrom[0]
                J1 = chrom[1::2]
                J2 = chrom[2::2]
                
                t = np.array([sin(6*pi*x1 + j*pi/self.lchrom) for j in range(self.lchrom)])
                t1 = t[1::2]
                t2 = t[3::2]
                
                f1 = x1 + 2/float(len(J1)) * np.sum((J1-t1)**2)
                f2 = 1 - sqrt(x1) + 2/float(len(J2)) * np.sum((J2-t2)**2)
                
                X.append(f1)
                Y.append(f2)
        
#        plt.scatter(y, x, marker='*')
        plt.scatter(Y, X, marker='x', cmap='r')
        
        plt.show()
#        plt.scatter(y, x, marker='x')

        

    def find_best_current(self):
        pop = [ind for ind in self.pop.np]
        pop.sort(key=lambda x:x.fitness, reverse=False)
        self.g_star = deepcopy(pop[0])
#        print 'here'
#        print self.g_star.fitness
        

    def generate(self):
        self.max = np.average(np.array([ind.chrom for ind in self.pop.np]), axis=0)
        for ind in self.pop.np:
#            print prob.pop.np[0].chrom
            if self.flip(self.pSwitch):
                self.global_search(ind)
            else:
                self.local_search(ind)
            
      
    def global_search(self, ind):
        ind2 = deepcopy(ind)
        
        L = self.levy()
        self.L.append(L)

        ind2.chrom += self.step * L * (self.g_star.chrom - ind2.chrom)
        
        if self.prob_type in ['ZDT1', 'ZDT2', 'ZDT3', 'LZ']:
            ind2.chrom = np.max((ind2.chrom,self.zeros), axis=0)
            ind2.chrom = np.min((ind2.chrom,self.ones), axis=0)
        if self.prob_type=='ZDT3':
            ones = deepcopy(self.ones)
            ones[0] = .852
            ind2.chrom = np.min((ind2.chrom,ones), axis=0)

            
#            ind2.chrom = np.max((ind2.chrom,zeros), axis=0)
#            ind2.chrom = np.min((ind2.chrom,ones), axis=0)
        
        fitness = self.evaluation(ind2)
        if fitness<ind.fitness:
            ind.fitness = fitness
            ind.chrom = ind2.chrom
        
    
    def local_search(self, ind):

#        eps = c.logistic_map(self.max_gen, .01, 1)[self.pop.gen]
        ind2 = deepcopy(ind)
        
        if self.chaos_local==True:
#            dif = self.chaotic_seq[self.pop.gen-1] * self.max
#            ind2.chrom += eps * dif
            eps = self.chaotic_seq[(self.pop.gen-1)*self.mu + randint(0, self.mu)]
        else:
            eps = rand()
        chrom1 = self.pop.np[int(rand()*self.mu)].chrom
        chrom2 = self.pop.np[int(rand()*self.mu)].chrom
        
        ind2.chrom += eps * (chrom1-chrom2)
        
        if self.prob_type in ['ZDT1', 'ZDT2', 'ZDT3', 'LZ']:
            ind2.chrom = np.max((ind2.chrom,self.zeros), axis=0)
            ind2.chrom = np.min((ind2.chrom,self.ones), axis=0)
        if self.prob_type=='ZDT3':
#            zeros = deepcopy(self.zeros-.733)
#            zeros[0] = 0
            ones = deepcopy(self.ones)
            ones[0] = .852
            ind2.chrom = np.min((ind2.chrom,ones), axis=0)
            
#            ind2.chrom = np.max((ind2.chrom,zeros), axis=0)
#            ind2.chrom = np.min((ind2.chrom,ones), axis=0)
            
        fitness = self.evaluation(ind2)
        if fitness<ind.fitness:
            ind.fitness = fitness
            ind.chrom = ind2.chrom

        
#    def search_dist(self):
#        if self.search_type=='levy':
#            return levy()
        

    def levy(self):
        s = []
        for i in range(self.lchrom):
            up = gamma(1+self.beta) * (pi*self.beta/2.)
            down = gamma((1+self.beta)/2.) * 2**((self.beta-1)/2.)
            sigma = (up / down) ** (1/self.beta)
            
            U = normal(0, sigma)
            V = normal(0, 1)
        
            s.append(U / abs(V)**(1/self.beta))
            
#        s = self.map.logistic_map(self.lchrom, rand()/10., rand()*4)
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
        print ('%i\t%f\t%i\t%f'% (self.pop.gen, self.pop.min, self.best_gen, self.best_fitness))
        f.write('%i\t%.10f\t%i\t%.10f\n'% (self.pop.gen, self.pop.min, self.best_gen, self.best_fitness))
        for ind in self.pop.np:
            for i in ind.chrom:
                f.write('%.2f\t'%i)
            f.write('\t')
            f.write('%.2f\n'%ind.fitness)
##            print ind.chrom
##            print ind.fitness
        f.writelines('--------------------------------\n')
           
           
    def final_report(self, seed):
        f = open('report', 'a+')
        f.write('dimension\t%i\n'%prob.lchrom)
#        f.write('data range\t(-%.3f,%.3f)\n'%(prob.range, prob.range))
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
    '''
    955
    626
    '''
    f = open('out', 'w+')
    import random

    best_fitness = 10
    cnt = 0

    pop = []
    F = []
    
    for i in [True, False]:
#    for i in range(1):
#        print i
        seed = int(random.random()*999)
        seed = 200
        np.random.seed(seed)
        dim = 30
        prob = Problem(dimension=dim, lb=[-30]*dim, ub=[30]*dim, max_gen=3000,
                       mu=50, landa=35, beta=1.5, step=.1, prob_type='ackley',
                       pSwitch=.8, chaos_init=i, chaos_local=i)
        print i
        prob.initialize()
#        print prob.pop.np[0].chrom
#        prob.pop.np = prob.pop.op
        prob.find_best_current()
        
        prob.gen_statistics()    
        prob.report()
        
        F.append(prob.best_fitness)

#        print prob.pop.np[0].chrom
#        print prob.pop.np[0].fitness
        
        while prob.pop.gen<prob.max_gen:
            prob.pop.gen += 1
            
            prob.generate()
            prob.gen_statistics()
            prob.find_best_current()
            F.append(prob.best_fitness)
            
            if prob.pop.gen%100==0:
                prob.report()
            prob.pop.op = prob.pop.np
    #        print '-----------------------------'
        f.close()
        
#        print 'solution :'
#        for c in prob.best_chrom:
#            print '%.15f  '%c,
#        print 'fitness :  ', prob.best_fitness
#        print 'seed :   ', seed
    #    print '-----------------------------\n'
        
        prob.final_report(seed)
                    
        best_fitness = prob.best_fitness
    	
        pop.extend([ind for ind in prob.pop.np])
#        prob.pareto()
#        break

    x = range(len(F)/2)        
    plt.plot(x, np.log(F[:len(F)/2]), 'g', x, np.log(F[len(F)/2:]), 'r')
    plt.show()

    
    prob.pop.np = [ind for ind in pop]
#    prob.pareto()