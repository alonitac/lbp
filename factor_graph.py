###############################################################################
# factor graph data structure implementation 
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

import functools
import numpy as np
from factors import *

class FactorGraph:
    def __init__(self, numVar=0, numFactor=0):
        '''
        var list: index/names of variables

        domain list: the i-th element represents the domain of the i-th variable; 
                     for this programming assignments, all the domains are [0,1]

        varToFactor: list of lists, it has the same length as the number of variables. 
                     varToFactor[i] is a list of the indices of Factors that are connected to variable i

        factorToVar: list of lists, it has the same length as the number of factors. 
                     factorToVar[i] is a list of the indices of Variables that are connected to factor i

        factors: a list of Factors

        messagesVarToFactor: a dictionary to store the messages from variables to factors,
                            keys are (src, dst), values are the corresponding messages of type Factor

        messagesFactorToVar: a dictionary to store the messages from factors to variables,
                            keys are (src, dst), values are the corresponding messages of type Factor
        '''
        self.var = [None for _ in range(numVar)]
        self.domain = [[0,1] for _ in range(numVar)]
        self.varToFactor = [[] for _ in range(numVar)]
        self.factorToVar = [[] for _ in range(numFactor)]
        self.factors = []
        self.messagesVarToFactor = {}
        self.messagesFactorToVar = {}
        self.marginals = [None for _ in range(numVar)]
    
    def addFactor(self, factor):
        '''
        :param factor: a Factor object
        '''
        self.factors.append(Factor(factor))
        assert len(self.factors) <= len(self.factorToVar)
        for var_idx in factor.scope:
            self.varToFactor[var_idx].append(len(self.factors) - 1)
        self.factorToVar[len(self.factors) - 1] = factor.scope

    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assigment
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factors:
            output *= f.val[tuple(a[f.scope])]
        return output
    
    def getInMessage(self, src, dst, type="varToFactor"):
        '''
        param - src: the source factor/clique index
        param - dst: the destination factor/clique index
        param - type: type of messages. "varToFactor" is the messages from variables to factors; 
                    "factorToVar" is the message from factors to variables
        return: message from src to dst
        
        In this function, the message will be initialized as an all-one vector (normalized) if 
        it is not computed and used before. 
        '''
        if type == "varToFactor":
            if (src, dst) not in self.messagesVarToFactor:
                inMsg = Factor()
                inMsg.scope = [src]
                inMsg.card = [len(self.domain[src])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesVarToFactor[(src, dst)] = inMsg
            return self.messagesVarToFactor[(src, dst)]

        if type == "factorToVar":
            if (src, dst) not in self.messagesFactorToVar:
                inMsg = Factor()
                inMsg.scope = [dst]
                inMsg.card = [len(self.domain[dst])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesFactorToVar[(src, dst)] = inMsg
            return self.messagesFactorToVar[(src, dst)]

    def runParallelLoopyBP(self, iterations):
        '''
        param - iterations: the number of iterations you do loopy BP
          
        In this method, you need to implement the loopy BP algorithm. The only values 
        you should update in this function are self.messagesVarToFactor and self.messagesFactorToVar. 
        
        Warning: Don't forget to normalize the message at each time. You may find the normalize
        method in Factor useful.

        Note: You can also calculate the marginal MAPs after each iteration here...
        '''
        for it in range(iterations):
            print('.', end='', flush=True)
            if (it+1) % 5 == 0:
                print(it+1, end='', flush=True)

            for factor_i, factor_scope in enumerate(self.factorToVar):
                for var_i in factor_scope:
                    msg = self.factors[factor_i]
                    for var_j in factor_scope:
                        if var_j != var_i:
                            msg = msg.multiply(self.getInMessage(var_j, factor_i, type='varToFactor'))
                    self.messagesFactorToVar[(factor_i, var_i)] = msg.marginalize_all_but([var_i]).normalize()

            for var_i, var_factors in enumerate(self.varToFactor):
                for factor_i in var_factors:
                    var_factors_cpy = var_factors.copy()
                    var_factors_cpy.remove(factor_i)
                    msg = self.factors[var_factors_cpy[0]]
                    for factor_j in var_factors_cpy[1:]:
                        msg = msg.multiply(self.getInMessage(factor_j, var_i, type='factorToVar'))

                    self.messagesVarToFactor[(var_i, factor_i)] = msg.normalize()

    def estimateMarginalProbability(self, var):
        '''
        Estimate the marginal probability of a single variable after running
        loopy belief propagation. (This method assumes runParallelLoopyBP has been run)

        param - var: a single variable index
        return: numpy array of size 2 containing the marginal probabilities 
                that the variable takes the values 0 and 1
        '''
        return self.factors[var].multiply(self.messagesVarToFactor[(var, var)]).marginalize_all_but([var]).val

    def getMarginalMAP(self):
        '''
        In this method, the return value output should be the marginal MAP assignment for each variable.
        You may utilize the method estimateMarginalProbability.
        
        example: (N=2, 2*N=4)
        '''
        y = []
        for var in range(len(self.var)):
            y.append(self.estimateMarginalProbability(var))
        return np.argmax(y, axis=1)

    def print(self):
        print('Variables:')
        for i in range(len(self.var)):
            print('  Variable {}: {}'.format(i, self.var[i]))
            print('     In factors:', self.varToFactor[i])
        print('Factors:')
        for i, f in enumerate(self.factors):
            print('  Factor {}: {}'.format(i, f))
            print('     vars=', self.factorToVar[i])
            print('     scope=', f.scope)
            print('     card=', f.card)
            print('     val=', f.val)
