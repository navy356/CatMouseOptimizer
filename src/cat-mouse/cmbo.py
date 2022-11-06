import numpy
import random

class CMBO:
    def __init__(self,costfunc,parameters,population_size,iterations=50,domain=(-100,100)):
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

    def init_population(self):
        self.population = numpy.random.rand(*self.shape) * (self.domain[1] - self.domain[0]) + self.domain[0];

    def calculate_cost(self,arr=None):
        if arr is not None:
            return numpy.apply_along_axis(lambda row : self.costfunc(*tuple(row)),1,arr)
        self.costs = numpy.apply_along_axis(lambda row : self.costfunc(*tuple(row)),1,self.population)

    def sort(self):
        indices = numpy.argsort(self.costs)
        self.costs = numpy.take(self.costs,indices,0)
        self.population = numpy.take(self.population,indices,0)

    def gen_mice(self):
        self.mice = self.population[0:self.mice_shape[0]]
        if self.mice_shape!=self.cat_shape:
            self.mice=numpy.r_[ self.mice, numpy.zeros((1,self.parameters))]

    def gen_cats(self):
        self.cats = self.population[self.mice_shape[0]:]

    def update_cats(self):
        r = numpy.random.rand(*self.cat_shape) 
        I = numpy.random.rand(*self.cat_shape)
        temp_cats = self.cats + r*(self.mice - numpy.round_(1+I)*self.cats)
        temp_costs = self.calculate_cost(temp_cats)
        for i in range(0,self.population_size//2):
            if temp_costs[i] < self.costs[self.cat_shape[0]+i]:
                self.cats[i] = temp_cats[i]

    def update_mice(self):
        Haven = self.population


cmbo = CMBO(lambda x,y: x*y,2,11)
cmbo.init_population()
cmbo.calculate_cost()
cmbo.sort()
cmbo.gen_cats()
cmbo.gen_mice()
print(cmbo.costs)
cmbo.update_cats()