#--- IMPORT DEPENDENCIES ------------------------------------------------------+
""" Standard Packages"""
from __future__ import division
import argparse, csv, copy, itertools, json, math, random, sys
import logging as log
from os.path import join
import logging as log
import multiprocessing as mp

"""My Stuff"""
from HTM import *
from models.ARMAModels import ARMATimeSeries
from models.SimpleSequence import VeryBasicSequence
#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
    return (x[0]**2/2 +x[1]**2/4)

def func2(x):
    time_series = VeryBasicSequence(pattern=1)
    return HTM(time_series, x[0], verbosity=0).train(error_method="binary")

def func3(x):
    time_series = VeryBasicSequence(pattern=2)
    network = HTM(time_series, x[0], verbosity=0)
    return network.train(error_method="binary")

def func4(x):
    time_series = VeryBasicSequence(pattern=3)
    network = HTM(time_series, x[0], verbosity=0)
    return network.train(error_method="binary")

def func5(x):
    time_series = VeryBasicSequence(pattern=4)
    network = HTM(time_series, x[0], verbosity=0)
    return network.train(error_method="binary")

def func22(x):
    time_series = VeryBasicSequence(pattern=1)
    network = HTM(time_series, x[0], verbosity=0)
    return network.train(error_method="binary", sibt=int(x[1]), iter_per_cycle=int(x[2]), max_cycles=int(x[3]))

def func23(x):
    time_series = VeryBasicSequence(pattern=2)
    network = HTM(time_series, x[0], verbosity=0)
    return network.train(error_method="binary", sibt=int(x[1]), iter_per_cycle=int(x[2]), max_cycles=int(x[3]))

def func24(x):
    time_series = VeryBasicSequence(pattern=3)
    network = HTM(time_series, x[0], verbosity=0)
    return network.train(error_method="binary", sibt=int(x[1]), iter_per_cycle=int(x[2]), max_cycles=int(x[3]))

def func25(x):
    time_series = VeryBasicSequence(pattern=4)
    return HTM(time_series, x[0], verbosity=0).train(error_method="binary", sibt=int(x[1]), iter_per_cycle=int(x[2]), max_cycles=int(x[3]))

def sanfunc(x):
    return HTM(VeryBasicSequence(pattern=4, n=1000), x[0], verbosity=0).train(error_method="binary")


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

    def copy_from(self, particle):
        self.position_i=copy.deepcopy(particle.position_i)
        self.velocity_i=copy.deepcopy(particle.velocity_i)
        self.pos_best_i=copy.deepcopy(particle.pos_best_i)
        self.err_best_i=-particle.err_best_i
        self.err_i=-particle.err_I

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weight the previous velocity)
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

def eval_pos_unpacker(args):
    return evaluate_position(*args)

def evaluate_position(swarm, j, costFunc):
    swarm[j].evaluate(costFunc)
    return (j, swarm[j])

class PSO():
    def __init__(self,costFunc,bounds,num_particles,maxiter,processes=mp.cpu_count()-1):
        DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
        self.log_file = join('../logs/', 'swarmp-on_{}-({}particles-{}maxiter-{}processes)-{}.log'.format(costFunc.__name__,num_particles,maxiter,processes,DATE))
        log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = self.log_file, level=log.DEBUG)

        log.debug("...initializing swarm....\n    costFunc: {}\n    num_particles: {}\n    processes: {}\n".format(costFunc.__name__,num_particles,maxiter,processes))

        global num_dimensions
        num_dimensions=len(bounds)
        log.debug("    \n    num_dimensions: {}".format(num_dimensions))
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            x0 = []
            for j in range(0,num_dimensions):
                x0.append(random.uniform(min(bounds[j]), max(bounds[j])))
            swarm.append(Particle(x0))
        log.debug("...swarm intialized:")
        for i in range(0,num_particles):
            log.debug("Particle {}: {}".format(i,swarm[i].position_i))

        # begin optimization loop
        i=0
        csv_out = open(join("../outputs/", 'swarm_on_params-{}.csv'.format(DATE)), "w+")
        writer = csv.writer(csv_out)
        header_row = ["Iteration"]
        for j in range(0,num_particles):
            header_row.append("Particle {}".format(j))
            for k in range(0,num_dimensions):
                header_row.append("Particle {}'s x[{}] Position".format(j,k))
            header_row.append("Particle {}'s Error".format(j))
        writer.writerow(header_row)

        p = mp.Pool(processes = processes)
        while i < maxiter:
            print i,err_best_g, pos_best_g
            log.debug("\n+++++ Beginning Iteration {} +++++".format(i))
            log.debug("    i: {}\n    err_best: {}\n    pos_best {}".format(i, err_best_g, pos_best_g))
            log.debug("...entering pool mapping...")
            results = p.map(eval_pos_unpacker, itertools.izip(itertools.repeat(swarm), range(num_particles), itertools.repeat(costFunc)))
            log.debug("...pool mapping exited...")
            results.sort()
            log.debug("...pool results sorted...")
            results = [r[1] for r in results]
            swarm = copy.deepcopy(results)
            log.debug("...copying results back to main thread's copy...")

            log.debug("...evaluating fitness...")
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            log.debug("New best error: {}\nNew best position: {}".format(err_best_g, pos_best_g))

            log.debug("...updating particle position and velocity...")
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
            csv_out.flush()
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
        log.debug("Final:\n    Position: {}\n    Error: {}".format(pos_best_g,err_best_g))
        log.debug("...closing handlers")
        logger = log.getLogger()
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

#--- RUN ----------------------------------------------------------------------+
def swarm_test():
    bounds=[(-100,100),(-100,100)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func1,bounds,num_particles=6,maxiter=12, processes=6)

def swarmv1():
    bounds=[(0.00001,1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func2,bounds,num_particles=6,maxiter=12, processes=6)
    bounds=[(0.00001,2)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func3,bounds,num_particles=6,maxiter=12, processes=6)
    bounds=[(0.00001,2)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func4,bounds,num_particles=6,maxiter=12, processes=6)
    bounds=[(0.00001,1)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func5,bounds,num_particles=6,maxiter=12, processes=6)

def swarmv2():
    bounds=[(0.00001,1), (0,50), (1,5), (5,20)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func22,bounds,num_particles=16,maxiter=24, processes=16)
    bounds=[(0.00001,2), (0,50), (1,5), (5,20)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func23,bounds,num_particles=16,maxiter=24, processes=16)
    bounds=[(0.00001,2), (0,50), (1,5), (5,20)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func24,bounds,num_particles=16,maxiter=24, processes=16)
    bounds=[(0.00001,1), (0,50), (1,5), (5,20)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(func25,bounds,num_particles=16,maxiter=24, processes=16)

def swarmsan():
    bounds=[(0.00001,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(sanfunc,bounds,num_particles=7,maxiter=8, processes=7)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-m', type=str, required=True,
            dest='mode', help='Which functions to swarm on. {test, v1, v2}')
    args = parser.parse_args()
    if args.mode == "test":
        print("Testing....")
        swarm_test()
    elif args.mode == "v1":
        print("Version 1 selected")
        swarmv1()
    elif args.mode == "v2":
        print("Version 2 selected")
        swarmv2()
    elif args.mode == "san":
        print("Sanity check selected")
        swarmsan()



#--- END ----------------------------------------------------------------------+

if __name__ == "__main__":
    main()
