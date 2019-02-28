#--- IMPORT DEPENDENCIES ------------------------------------------------------+
""" Standard Packages"""
from __future__ import division
import argparse, csv, copy, itertools, json, math, random, signal, sys
import logging as log
from os.path import join
import logging as log
import multiprocessing as mp

"""My Stuff"""
from HTM import *
from models.ARMAModels import ARMATimeSeries
from models.SimpleSequence import VeryBasicSequence

"""Handles Ctrl+C"""
def signal_handler(sig, frame):
        sys.exit(0)


#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def testfunc(x):
    return (x[0]**2/2 +x[1]**2/4)

def vbsfuncv1(args):
    x = args["x"]
    return HTM(VeryBasicSequence(pattern=args["pattern"]), x[0], verbosity=0).train(error_method="binary", logging=True)

def vbsfuncv2(args):
    x = args["x"]
    return HTM(VeryBasicSequence(pattern=args["pattern"]), x[0], verbosity=0).train(error_method="binary", sibt=int(x[1]), iter_per_cycle=int(x[2]))

def vbsfuncv3(args):
    x = args["x"]
    param_dict = { "spParams" : { "potentialPct": x[3], "numActiveColumnsPerInhArea": int(x[4]), "synPermConnected": x[5], "synPermInactiveDec": x[6] }, "tmParams" : { "activationThreshold": int(x[7])}, "newSynapseCount" : int(x[8]) }
    return HTM(VeryBasicSequence(pattern=args["pattern"]), x[0], params=param_dict, verbosity=0).train(error_method="binary", sibt=int(x[1]), iter_per_cycle=int(x[2]))

def vbsfuncv4(args):
    x = args["x"]
    param_dict = { "spParams" : { "potentialPct": x[3], "numActiveColumnsPerInhArea": int(x[4]), "synPermConnected": x[5], "synPermInactiveDec": x[6] }, "tmParams" : { "activationThreshold": int(x[7])}, "newSynapseCount" : int(x[8]) }
    return HTM(VeryBasicSequence(pattern=args["pattern"]), x[0], params=param_dict, verbosity=0).train(error_method="binary", sibt=int(x[1]), iter_per_cycle=int(x[2]), weights={ 1: 1.0, 5: x[9] })

def arfuncv1(args):
    x = args["x"]
    return HTM(ARMATimeSeries(1,0, sigma=args["sigma"], normalize=False), x[0], verbosity=0).train(error_method="rmse", sibt=int(x[1]), iter_per_cycle=int(x[2]))

def arfuncv2(args):
    x = args["x"]
    param_dict = { "spParams" : { "potentialPct": x[3], "numActiveColumnsPerInhArea": int(x[4]), "synPermConnected": x[5], "synPermInactiveDec": x[6] }, "tmParams" : { "activationThreshold": int(x[7])}, "newSynapseCount" : int(x[8]) }
    return HTM(ARMATimeSeries(1,0, sigma=args["sigma"], n=100, normalize=False), x[0], params=param_dict, verbosity=0).train(error_method="rmse", sibt=int(x[1]), iter_per_cycle=int(x[2]))

def arfuncv3(args):
    x = args["x"]
    param_dict = { "spParams" : { "potentialPct": x[3], "numActiveColumnsPerInhArea": int(x[4]), "synPermConnected": x[5], "synPermInactiveDec": x[6] }, "tmParams" : { "activationThreshold": int(x[7])}, "newSynapseCount" : int(x[8]) }
    return HTM(ARMATimeSeries(1,0, sigma=args["sigma"], n=100, normalize=False), x[0], params=param_dict, verbosity=0).train(error_method="rmse", sibt=int(x[1]), iter_per_cycle=int(x[2]), weights={ 1: 1.0, 5: x[9] })

def sanfunc(x):
    return HTM(VeryBasicSequence(pattern=1, n=1000), x[0], verbosity=0).train(error_method="binary")


#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.step = 0

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    def copy_from(self, particle):
        self.position_i=copy.deepcopy(particle.position_i)
        self.velocity_i=copy.deepcopy(particle.velocity_i)
        self.pos_best_i=copy.deepcopy(particle.pos_best_i)
        self.err_best_i=-particle.err_best_i
        self.err_i=-particle.err_i

    # evaluate current fitness
    def evaluate(self,costFunc, func_sel):
        func_sel["x"] = self.position_i
        self.err_i=costFunc(func_sel)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.4 # constant inertia weight (how much to weight the previous velocity)
        if self.step < 12:
            w=1.1+((self.step*-.7)/24) # start it out higher and linearly lower it
        c1=2        # cognitive constant
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
        self.step+=1

def eval_pos_unpacker(args):
    return evaluate_position(*args)

def evaluate_position(particle, j, costFunc, func_sel):
    particle.evaluate(costFunc, func_sel)
    return (j, particle)

class PSO():
    def __init__(self, costFunc, bounds, num_particles, maxiter, func_sel=None,  processes = (mp.cpu_count()-1), descr=None):
        DATE = '{}'.format(strftime('%Y-%m-%d_%H:%M:%S', localtime()))
        self.log_file = join('../logs/', 'swarmp-on_{}-({}particles-{}maxiter-{}processes)-{}.log'.format(costFunc.__name__,num_particles,maxiter,processes,DATE))
        if not func_sel == None:
            sel_string = ""
            for key, value in func_sel.iteritems():
                sel_string+="{}-{}"
            self.log_file = join('../logs/', 'swarmp-on_{}-({})-({}particles-{}maxiter-{}processes)-{}.log'.format(costFunc.__name__,sel_string,num_particles,maxiter,processes,DATE))
        log.basicConfig(format = '[%(asctime)s] %(message)s', datefmt = '%m/%d/%Y %H:%M:%S %p', filename = self.log_file, level=log.DEBUG)

        log.debug("...initializing swarm....\n    costFunc: {}\n    num_particles: {}\n    maxiter: {}\n    func_sel: {}\n    processes: {}\n".format(costFunc.__name__,num_particles,maxiter,func_sel,processes))

        global num_dimensions
        num_dimensions=len(bounds)
        log.debug("    \n    num_dimensions: {}".format(num_dimensions))
        err_best_g=float("inf")                   # best error for group
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
        csv_out = open(join('../outputs/', 'swarmp-on_{}-({})-({}particles-{}maxiter-{}processes)-{}.csv'.format(costFunc.__name__,func_sel,num_particles,maxiter,processes,DATE)), "w+")
        writer = csv.writer(csv_out)
        header_row = ["Iteration"]
        for j in range(0,num_particles):
            if descr == None:
                for k in range(0,num_dimensions):
                    header_row.append("Particle {}'s x[{}] Position".format(j,k))
            else:
                for k in range(0,num_dimensions):
                    header_row.append("Particle {}'s {}".format(j,descr[k]))
            header_row.append("Particle {}'s Error".format(j))
        header_row.append("Average Error")
        writer.writerow(header_row)

        pool = mp.Pool(processes = processes)
        while i < maxiter:
            print i, err_best_g, pos_best_g
            log.debug("\n+++++ Beginning Iteration {} +++++".format(i))
            log.debug("    i: {}\n    err_best: {}\n    pos_best {}".format(i, err_best_g, pos_best_g))
            log.debug("...entering pool mapping...")
            results = pool.map(eval_pos_unpacker, itertools.izip(swarm, range(num_particles), itertools.repeat(costFunc), itertools.repeat(func_sel)))

            log.debug("...pool mapping exited...")
            results.sort()
            log.debug("...pool results sorted...")
            swarm = [r[1] for r in results]
            log.debug("...copying results back to main thread's copy...")

            log.debug("...evaluating fitness...")
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            log.debug("New best error: {}\nNew best position: {}".format(err_best_g, pos_best_g))

            output_row = ["{}".format(i)]
            avg_err = 0
            for j in range(0,num_particles):
                for k in range(0,num_dimensions):
                    output_row.append(swarm[j].position_i[k])
                avg_err+=swarm[j].err_i
                output_row.append(swarm[j].err_i)
            output_row.append(avg_err/num_particles)
            writer.writerow(output_row)
            csv_out.flush()

            log.debug("...updating particle position and velocity...")
            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        pool.close()
        pool.join()
        #print final results
        writer.writerow(["FINAL:"])
        result_descr = ["Best Error", ""]
        if descr == None:
            for k in range(0,num_dimensions):
                result_descr.append("x[{}] Best Position".format(k))
        else:
            for k in range(0,num_dimensions):
                result_descr.append("{} Best Position".format(descr[k]))
        writer.writerow(result_descr)
        result = [ err_best_g, ""] + pos_best_g
        writer.writerow(result)
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
    PSO(testfunc,bounds,num_particles=6, maxiter=24, processes=6, descr=["x-coordinate", "y-coordinate"])

def swarmvbsv1():
    bounds=[(0.00000001,1)]
    for i in range(1,6):
        if i == 2 or i == 3:
            bounds=[(0.00000001,2)]
        elif i == 5:
            bounds=[(0.00000001,4)]
        PSO(vbsfuncv1,bounds,num_particles=6,maxiter=24, func_sel={"pattern":i}, processes=6, descr=["RDSE Resolution"])

def swarmvbsv2():
    descr=["RDSE Resolution", "SIBT", "IterPerCycle"]
    bounds=[(0.0000001,1), (0,50), (1,5)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    for i in range(1,6):
        if i == 2 or i == 3:
            bounds=[(0.0000001,2), (0,50), (1,5)]
        elif i == 5:
            bounds=[(0.00000001,4), (0,50), (1,5)]
        PSO(vbsfuncv2,bounds,num_particles=12,maxiter=24, func_sel={"pattern":i}, processes=12, descr=descr)

def swarmvbsv3():
    descr = ["RDSE Resolution", "SIBT", "IterPerCycle", "potentialPct", "numActiveColumnsPerInhArea", "synPermConnected", "synPermInactiveDec", "activationThreshold", "newSynapseCount"]
    bounds=[(0.0000000001,1), (0,50), (1,5), (.00001, 1), (20, 80), (.00001, 0.5), (.00001, .1), (8, 40), (15, 35)]
    for i in range(1,6):
        if i == 2 or i == 3:
            bounds=[(0.0000000001,2), (0,50), (1,5), (.00001, 1), (20, 80), (.00001, 0.5), (.00001, .1), (8, 40), (15, 35)]
        elif i == 5:
            bounds=[(0.0000000001,4), (0,50), (1,5), (.00001, 1), (20, 80), (.00001, 0.5), (.00001, .1), (8, 40), (15, 35)]
        PSO(vbsfuncv3,bounds,num_particles=12,maxiter=24, func_sel={"pattern":i}, processes=12, descr=descr)

def swarmvbsv4():
    descr = ["RDSE Resolution", "SIBT", "IterPerCycle", "potentialPct", "numActiveColumnsPerInhArea", "synPermConnected", "synPermInactiveDec", "activationThreshold", "newSynapseCount", "fiveWeight"]
    bounds=[(0.0000000001,1), (0,50), (1,5), (.00001, 1), (20, 80), (.00001, 0.5), (.00001, .1), (8, 40), (15, 35), (0,10)]
    for i in range(1,6):
        if i == 2 or i == 3:
            bounds=[(0.0000000001,2), (0,50), (1,5), (.00001, 1), (20, 80), (.00001, 0.5), (.00001, .1), (8, 40), (15, 35), (0,10)]
        elif i == 5:
            bounds=[(0.0000000001,4), (0,50), (1,5), (.00001, 1), (20, 80), (.00001, 0.5), (.00001, .1), (8, 40), (15, 35), (0,10)]
        PSO(vbsfuncv4,bounds,num_particles=12,maxiter=24, func_sel={"pattern":i}, processes=12, descr=descr)

def arswarmv1():
    descr=["RDSE Resolution", "SIBT", "IterPerCycle"]
    bounds=[(0.00000001,1), (0,50), (1,5)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    for i in range(1,6):
        PSO(arfuncv1,bounds,num_particles=12,maxiter=36, func_sel={"sigma":i}, processes=12, descr=descr)

def arswarmv2():
    descr = ["RDSE Resolution", "SIBT", "IterPerCycle", "potentialPct", "numActiveColumnsPerInhArea", "synPermConnected", "synPermInactiveDec", "activationThreshold", "newSynapseCount"]
    bounds=[(0.0000000001,1), (0,50), (1,5), (.00001, 1), (20, 80), (.00001, 0.5), (.00001, .1), (8, 40), (15, 35)]
    for i in range(1,6):
        PSO(arfuncv2,bounds,num_particles=18,maxiter=24, func_sel={"sigma":i}, processes=18, descr=descr)

def arswarmv3():
    descr = ["RDSE Resolution", "SIBT", "IterPerCycle", "potentialPct", "numActiveColumnsPerInhArea", "synPermConnected", "synPermInactiveDec", "activationThreshold", "newSynapseCount", "fiveWeight"]
    bounds=[(0.0000000001,1), (0,50), (1,5), (.00001, 1), (20, 80), (.00001, 0.5), (.00001, .1), (8, 40), (15, 35), (0,10)]
    for i in range(1,6):
        PSO(arfuncv3,bounds,num_particles=18,maxiter=24, func_sel={"sigma":i}, processes=18, descr=descr)

def swarmsan():
    bounds=[(0.00001,4)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...] #CPMC, RDSE resolution,
    PSO(sanfunc,bounds,num_particles=6,maxiter=24, processes=6, descr=["RDSE Resolution"])

def cust():
    pass

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
    elif args.mode == "vbsv1":
        print("Very basic sequence Version 1 selected")
        swarmvbsv1()
    elif args.mode == "vbsv2":
        print("Very basic sequence Version 2 selected")
        swarmvbsv2()
    elif args.mode == "vbsv3":
        print("Very basic sequence version 3 selected")
        swarmvbsv3()
    elif args.mode == "vbsv4":
        print("Very basic sequence version 4 selected")
        swarmvbsv4()
    elif args.mode == "arv1":
        print("Autoregressive version 1 selected")
        arswarmv1()
    elif args.mode == "arv2":
        print("Autoregressive version 2 selected")
        arswarmv2()
    elif args.mode == "arv3":
        print("Autoregressive version 3 selected")
        arswarmv3()
    elif args.mode == "san":
        print("Sanity check selected")
        swarmsan()
    elif args.mode == "cust":
        print("custom setup")
        cust()



#--- END ----------------------------------------------------------------------+

if __name__ == "__main__":
    main()
