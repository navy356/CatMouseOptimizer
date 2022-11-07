import numpy
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class CMBO:
    def __init__(self,costfunc,parameters,population_size,iterations=1000,domain=(-100,100),w=0.8,c=2):
        self.costfunc = costfunc
        self.parameters = parameters
        self.population_size = population_size
        self.iterations = iterations
        self.domain = domain
        self.shape = (population_size,parameters)
        self.population = numpy.zeros(self.shape)
        self.costs = numpy.zeros((population_size,1))
        self.mice_shape = (population_size//2,parameters)
        self.mice = numpy.zeros(self.mice_shape)
        self.cat_shape = (population_size-population_size//2,parameters)
        self.cats = numpy.zeros(self.cat_shape)
        self.update_size = (1,parameters)
        self.gbest = numpy.zeros(self.update_size)
        self.gbest_cost = 0
        self.pbest_costs = numpy.zeros((population_size,1))
        self.pbest = numpy.zeros(self.shape)
        self.cat_velocities = numpy.zeros(self.cat_shape)
        self.mice_velocities = numpy.zeros(self.mice_shape)
        self.w = w
        self.c = c

    def init(self):
        self.population = numpy.random.rand(*self.shape) * ((self.domain[1] - self.domain[0]) + self.domain[0])
        self.cat_velocities = numpy.random.rand(*self.cat_shape) * ((self.domain[1] - self.domain[0]) + self.domain[0])
        self.mice_velocities = numpy.random.rand(*self.cat_shape) * ((self.domain[1] - self.domain[0]) + self.domain[0])
        self.calculate_cost()
        self.pbest = self.population
        self.pbest_costs = self.costs
        self.sort()
        if self.population_size > 0:
            self.gbest = self.population[0]
            self.gbest_cost = numpy.take(self.costs,0)

    def calculate_cost(self,arr=None):
        if arr is not None:
            return numpy.apply_along_axis(lambda row : self.costfunc(*tuple(row)),1,arr)
        self.costs = numpy.apply_along_axis(lambda row : self.costfunc(*tuple(row)),1,self.population)

    def sort(self):
        indices = numpy.argsort(self.costs)
        self.costs = numpy.take(self.costs,indices,0)
        self.population = numpy.take(self.population,indices,0)
        self.pbest = numpy.take(self.population,indices,0)

    def gen_mice(self):
        self.mice = self.population[0:self.mice_shape[0]]
        if self.mice_shape!=self.cat_shape:
            self.mice=numpy.r_[ self.mice, numpy.zeros((1,self.parameters))]

    def gen_cats(self):
        self.cats = self.population[self.mice_shape[0]:]

    def chase(self):
        self.cat_velocities = self.w*self.cat_velocities + self.c*numpy.random.rand(self.cat_shape[0],1)*self.mice

    def flee(self):
        gbest = None
        pbest = None
        gbest = numpy.atleast_2d(self.gbest).repeat(repeats=self.cat_shape[0]-0,axis=0)
        if self.cat_shape!=self.mice_shape:
            pbest = numpy.r_[self.pbest[0:self.mice_shape[0]], numpy.zeros((1,self.parameters))]
        else:
            pbest = self.pbest[0:self.mice_shape[0]]
        pbest_dist = self.calculate_distance(self.mice,pbest)
        gbest_dist = self.calculate_distance(self.mice,gbest)
        cat_dist = self.calculate_distance(self.mice, self.cats)
        sum_dist = pbest_dist + gbest_dist + cat_dist
        fp = pbest_dist/sum_dist
        fg = gbest_dist/sum_dist
        fc = cat_dist/sum_dist
        self.mice_velocities = fc*(self.cat_velocities+self.mice_velocities)+fp*numpy.random.rand(self.cat_shape[0],1)*(pbest-self.mice_velocities)+fg*numpy.random.rand(self.cat_shape[0],1)*(gbest-self.mice_velocities)

    def approach(self):
        self.mice = self.mice + self.mice_velocities
        numpy.clip(self.mice,self.domain[0],self.domain[1],out=self.mice)
        self.cats = self.cats + self.cat_velocities
        numpy.clip(self.cats,self.domain[0],self.domain[1],out=self.cats)
        
    def catch(self):
        for i in range(0,self.population_size//2):
            r = random.randint(0,self.parameters-1)
            self.mice[i][r] = random.random() * ((self.domain[1] - self.domain[0]) + self.domain[0])
        temp_mice_costs = self.calculate_cost(self.mice)
        temp_cat_costs = self.calculate_cost(self.cats)
        caught = numpy.greater(temp_mice_costs,temp_cat_costs)
        for i in range(0,self.population_size//2):
            if numpy.take(caught[i],0):
                self.cats[i]=self.mice[i]
                self.mice[i] = numpy.random.rand(1,self.parameters) * ((self.domain[1] - self.domain[0]) + self.domain[0])
            else:
                self.cat_velocities[i] = numpy.random.rand(1,self.parameters) * ((self.domain[1] - self.domain[0]) + self.domain[0])

        temp_mice_costs = self.calculate_cost(self.mice)
        temp_cat_costs = self.calculate_cost(self.cats)
        for i in range(0,self.population_size//2):
            if numpy.take(temp_mice_costs[i],0)<numpy.take(self.costs[i],0):
                self.population[i] = self.mice[i]
                if numpy.take(temp_mice_costs[i],0) < numpy.take(self.pbest_costs[i],0):
                    self.pbest[i] = self.mice[i]
                    self.pbest_costs[i] = temp_mice_costs[i]
                    if numpy.take(temp_mice_costs[i],0) < self.gbest_cost:
                        self.gbest = self.mice[i] 
                        self.gbest_cost = numpy.take(temp_mice_costs[i],0)
            if numpy.take(temp_cat_costs[i],0)<numpy.take(self.costs[i+self.mice_shape[0]],0):
                self.population[i+self.mice_shape[0]]=self.cats[i]
                if numpy.take(temp_cat_costs[i],0) < numpy.take(self.pbest_costs[i+self.mice_shape[0]],0):
                    self.pbest[i+self.mice_shape[0]] = self.cats[i]
                    self.pbest_costs[i] = temp_cat_costs[i]
                    if numpy.take(temp_cat_costs[i],0) < self.gbest_cost:
                        self.gbest = self.cats[i] 
                        self.gbest_cost = numpy.take(temp_cat_costs[i],0)

    def calculate_distance(self,x,y):
        return numpy.linalg.norm(x-y)

    def start(self):
        self.init()
        x= []
        y= []
        #plt.xlim([0, 0.1])
        for i in range(0,self.iterations):
            self.gen_cats()
            self.gen_mice()
            self.chase()
            self.flee()
            self.approach()
            self.catch()
            self.sort()
            x.append(self.gbest_cost)
            y.append(i)
        x= numpy.append(x,[])
        y = numpy.append(y,[])
        plt.plot(x,y)
        plt.xlabel("Optimized solution cost")
        plt.ylabel("Number of iterations")
        plt.show()

def objective(x):
    return x*x

cmbo = CMBO(objective,1,100,domain=(-99999,99999))
cmbo.start()
print(cmbo.gbest)
print(cmbo.gbest_cost)