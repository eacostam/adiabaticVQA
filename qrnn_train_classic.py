# -*- coding: utf-8 -*-
"""
Created on May 2024
 
@author: eacosta
"""
# Import Libraries
import numpy as np
from qiskit_algorithms.optimizers import SPSA
import matplotlib.pyplot as plt
import time

from qrnn import circuits_execution
from config import MAX_ITER, LOG_DIR
from helpers import qcprint


# QRNN CLASSICAL TRAINING
class QRNN_optimizer_classic():
    def __init__(self, data, labels, qrnn, shots):
        self.data = data
        self.labels = labels
        self.qrnn = qrnn
        self.shots = shots
        
    def cost_function(self, var_parameters):
        classifications = circuits_execution(self.data, var_parameters, self.qrnn, self.shots)

        cost = 0
        for i, classification in enumerate(classifications):
            p = classification.get(self.labels[i])
            cost += -np.log(p + 1e-10)
        cost /= len(self.data)

        return cost
    
    def train(self, init_ansatz_params):
        qcprint("\nTRAINING QRNN CLASSICAL approach for: " + str(MAX_ITER) + " iterations")
        start_time = time.time()
        log = OptimizerLog()
        spsa = SPSA(maxiter=MAX_ITER, callback=log.update)
        result = spsa.minimize(fun=self.cost_function, x0=init_ansatz_params, 
                               bounds=[(-1, 1), (-1, 1), (-1, 1), (-1, 1)])

        opt_var = result.x
        qcprint("Training Classical took %s sec" % (time.time() - start_time))

        # Plot Training Results
        fig = plt.figure()
        plt.plot(log.evaluations, log.costs)
        plt.xlabel('Steps')
        plt.ylabel('Cost')
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.title('QRNN Classical Training and Validation Loss')
        plt.savefig(LOG_DIR+"qrnn_cost_{}.png".format(time.time()))
        plt.show()
        
        return opt_var
    
# Optimizer Log Class
class OptimizerLog:
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
        
    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        qcprint("evaluation: " + str(evaluation) + ", cost: " + str(cost))
        self.costs.append(cost)
