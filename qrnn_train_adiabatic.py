# -*- coding: utf-8 -*-
"""
Created on May 2024
 
@author: eacosta
"""
import dimod
from dimod import Binary
from dwave.cloud import Client
import numpy as np
import time

from config import ADIABATIC_REAL, DWAVE_TOKEN, DWAVE_TIMELIMIT
from helpers import qcprint


class QRNN_optimizer_adiabatic():
    ang_div_2 = []
    w = []
    x = []
    y = []
    z = []
    
    blp = dimod.ConstrainedQuadraticModel()
    
    parts = 4
    angles = []
    exp =[]
    log_exp = []
    
    def __init__(self, parts, input_list, y_list):
        qcprint("\nBuilding Adiabatic model partitioning EXP function in " + str(parts) + " parts")
        self.parts=parts
        self.binaryVarsinit()
        self.discretizeExp()
        self.buildQUBO(input_list, y_list)

    #Create Binary variables
    def binaryVarsinit(self):
        #1. Rotation angles theta[x]/2
        self.w = [Binary('w{}'.format(i)) for i in range(self.parts)] #theta[0]
        self.x = [Binary('x{}'.format(i)) for i in range(self.parts)] #theta[1]
        self.y = [Binary('y{}'.format(i)) for i in range(self.parts)] #theta[2]
        self.z = [Binary('z{}'.format(i)) for i in range(self.parts)] #theta[3]
    
    #Discretize exponential function
    def discretizeExp(self):
        # Angles ϴ (radians)
        L = 1 * np.pi
        spoint = 0
        lp = L/self.parts
        # Take mid point of each part
        self.angles = [((spoint-(lp/2)) + ((i+1)*lp)) for i in range(self.parts)]
        qcprint("Angles: " + str(self.angles))

        # Angles ϴ/2
        self.angles_div_2 = [a/2 for a in self.angles]
        qcprint("Angles/2: " + str(self.angles_div_2))
        
        # Exponential coeficients for plain angles ϴ/2 (discretization of-π/2 to π/2)
        self.exp = [np.exp(angle) for angle in self.angles_div_2]
        self.log_exp = [np.log(e) for e in self.exp]
        qcprint("Exps : " + str(self.exp))
        qcprint("Logs : " + str(self.log_exp))

    def operator(self, psi_0):
        if len(psi_0) == 16:
            out = [0.0] * len(psi_0)

            for i in range(self.parts):
                for j in range(self.parts):
                    for k in range(self.parts):
                        for l in range(self.parts):
                            #Product of exp is sum of their lns
                            out[0] = np.log(0.00001 if psi_0[0]==0 else 0.99999)+sum([(self.log_exp[i]*-self.w[i])+(self.log_exp[j]*-self.x[j])+(self.log_exp[k]*-self.y[k])+(self.log_exp[l]*-self.z[l])])
                            out[8] = np.log(0.00001 if psi_0[8]==0 else 0.99999)+sum([(self.log_exp[i]*self.w[i])+(self.log_exp[j]*-self.x[j])+(self.log_exp[k]*-self.y[k])+(self.log_exp[l]*self.z[l])])

            return out

        else:
            return psi_0
            
        return 0
    
    #MSE
    def squared_error(self, psi_0, psi_f):
        # To avoid ln(1.0) or ln(0.0)shift those values 0.00001
        out = np.square(np.log(0.00001 if psi_f[0]==0 else 0.99999)-(self.operator(psi_0)[0])) + \
            np.square(np.log(0.00001 if psi_f[8]==0 else 0.99999)-(self.operator(psi_0)[8]))
        return out
    
    #Set QUBO ojective
    def buildQUBO(self, input_list, y_list):
        self.blp.set_objective(sum([self.squared_error(input_list[i], y_list[i]) for i in range(len(input_list))]))
        
        self.blp.add_constraint(sum(self.w) == 1, "t0 constraint")
        self.blp.add_constraint(sum(self.x) == 1, "t1 constraint")
        self.blp.add_constraint(sum(self.y) == 1, "t2 constraint")
        self.blp.add_constraint(sum(self.z) == 1, "t3 constraint")
    
    def optimize(self):
        qcprint("\nTRAINING QRNN ADIABATICAL approach")
        qcprint("Total number of constraints: " + str(len(self.blp.constraints)))
        qcprint("Number of binary variables:" + str(len(self.blp.variables)))

        start_time = time.time()
        end_time = time.time()
        
        optimal = [0, 0, 0, 0]
        solver = dimod.ExactCQMSolver()
        if ADIABATIC_REAL:
            qcprint("Training on real solver")
            client = Client.from_config(token=DWAVE_TOKEN)
            solver = client.get_solver(supported_problem_types__issubset={'cqm'}) 
            start_time = time.time()
            computation = solver.sample_cqm(self.blp, label="experiment_" + str(self.parts) + "_parts", time_limit=DWAVE_TIMELIMIT)
            end_time = time.time()
            sampleset = computation.sampleset
            qcprint("sampleset:")
            qcprint(sampleset)
            solutions = computation.samples
            qcprint ("solutions :")
            qcprint(solutions)

            if len(solutions) > 0:
                optimal = solutions[0]
                qcprint("Lowest energy solution: " + str(optimal))
            else:
                qcprint("No feasible solution found")
        else:
            qcprint("Training on simulated solver")
            start_time = time.time()
            solutions = solver.sample_cqm(self.blp)
            end_time = time.time()
            feasible_sols = solutions.filter(lambda s: s.is_feasible)
            qcprint("Feasible solutions:  " + str(len(feasible_sols)))
            if len(feasible_sols.samples()) > 0:
                qcprint(feasible_sols.samples()[0])
                optimal = feasible_sols.samples()[0]
            else:
                qcprint("No feasible solution found")
            
        qcprint("Training Adiabatically took %s sec" % (end_time - start_time))
        qcprint("Explored " + str(len(solutions)) + " options in total")

        return optimal
    
    def get_angles(self, optimal):
        qcprint("Get Angles from optimal: " + str(optimal))
        angles4VQC = []
        
        if ADIABATIC_REAL:  # On real mode, optimal solution is an array
            sol_ind = 0
            for i in range(4):  #w, x, y, z
                for j in range(self.parts):
                    if (optimal[sol_ind] == 1):
                        angles4VQC.append(self.angles[j])
                    sol_ind += 1
        else: # On simulations, optimal solution is a dictionary
            for i in range(len(self.w)):
                if optimal['w'+str(i)] == 1:
                    angles4VQC.append(self.angles[i])
                    
            for i in range(len(self.x)):
                if optimal['x'+str(i)] == 1:
                    angles4VQC.append(self.angles[i])
                    
            for i in range(len(self.y)):
                if optimal['y'+str(i)] == 1:
                    angles4VQC.append(self.angles[i])
                    
            for i in range(len(self.z)):
                if optimal['z'+str(i)] == 1:
                    angles4VQC.append(self.angles[i])

    
        qcprint("Angles to update the QRNN (theta0, theta1, theta2, theta3):")
        qcprint(angles4VQC)
        
        return angles4VQC
