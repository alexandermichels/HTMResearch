#--- IMPORT DEPENDENCIES ------------------------------------------------------+

from __future__ import division
import logging as log
import random
import math, sys
from os.path import join
from HTMNetwork import *
from models.ARMAModels import ARMATimeSeries
from models.SimpleSequence import VeryBasicSequence

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    return (x[0]**2/2 +x[1]**2/4)

def func2(x):
    time_series = VeryBasicSequence(pattern=1)
    network = HTM(time_series, x[0], cellsPerMiniColumn=8, verbosity=0)
    return train(network, error_method="Binary")

def func3(x):
    time_series = VeryBasicSequence(pattern=2)
    network = HTM(time_series, x[0], cellsPerMiniColumn=8, verbosity=0)
    return train(network, error_method="Binary")

def func4(x):
    time_series = VeryBasicSequence(pattern=3)
    network = HTM(time_series, x[0], cellsPerMiniColumn=8, verbosity=0)
    return train(network, error_method="Binary")

def func5(x):
    time_series = VeryBasicSequence(pattern=4)
    network = HTM(time_series, x[0], cellsPerMiniColumn=8, verbosity=0)
    return train(network, error_method="Binary")

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]

class PSO():
    def __init__(self,costFunc,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(bounds)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            x0 = []
            for j in range(0,num_dimensions):
                x0.append(random.uniform(min(bounds[j]), max(bounds[j])))
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
        csv_out = open(join("../outputs/", 'swarm_on_params-{}.csv'.format(DATE)), "w+")
        writer = csv.writer(csv_out)
        header_row = ["Iteration"]
        for j in range(0,num_particles):
            header_row.append("Particle {}".format(j))
            for k in range(0,num_dimensions):
                header_row.append("Particle {}'s x[{}] Position".format(j,k))
            header_row.append("Particle {}'s Error".format(j))
        writer.writerow(header_row)

        while i < maxiter:
            print i,err_best_g, pos_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                #print(swarm[j].position_i)
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)

            output_row = ["{}".format(i)]
            for j in range(0,num_particles):
                output_row.append(j)
                for k in range(0,num_dimensions):
                    output_row.append(swarm[j].position_i[k])
                output_row.append(swarm[j].err_i)
            writer.writerow(output_row)
            i+=1

        #print final results
        writer.writerow(["FINAL:"])
        result_descr = []
        for k in range(0,num_dimensions):
            result_descr.append("x[{}] Best Position".format(k))
        result_descr.append("Best Error")
        writer.writerow(result_descr)
        writer.writerow(pos_best_g + [err_best_g])
        csv_out.flush()
        csv_out.close()

        print 'FINAL:'
        print pos_best_g
        print err_best_g

#--- RUN ----------------------------------------------------------------------+
def main():
    #bounds=[(-5,5),(-5,5)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    #PSO(func1,bounds,num_particles=64,maxiter=16)
    bounds=[(0,1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func2,bounds,num_particles=16,maxiter=16)
    bounds=[(0,2)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func3,bounds,num_particles=16,maxiter=16)
    bounds=[(0,2)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func4,bounds,num_particles=16,maxiter=16)
    bounds=[(0,1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func5,bounds,num_particles=16,maxiter=16)


#--- END ----------------------------------------------------------------------+

if __name__ == "__main__":
    main()
