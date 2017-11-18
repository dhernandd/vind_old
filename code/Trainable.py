"""
Trainable.py
"""
import numpy as np
import collections

import cPickle as pickle
import dill

class Trainable(object):
    """
    TODO
    """
    def __init__(self, filename=None, **kwargs):
        """
        """
        self._costs = collections.OrderedDict([])
        self._validation_costs = collections.OrderedDict([])
        self._iter_count = 0

        self._best_validation_cost = -np.inf

    
    def append_log(self, cost, validation_cost=None, filename=None, iteration_time=None):
        """
        Call after a successful SGD iteration to append to log. 
        TODO: add ability to save parameters & such.
        
        There's nothing forcing you to use this in a coherent way, so take care.
        """
#         print 'Cost: ', cost
        self._costs[self._iter_count] = cost
        improved = False
        if validation_cost is not None:
            self._validation_costs[self._iter_count] = validation_cost
            if validation_cost > self._best_validation_cost:
                improved = True
                # update the patience
                # DANI: This does not make sense! Second condition is always satisfied? TODO?
    #                 if validation_cost > self._best_validation_cost*self._improvement_threshold:
    #                     self._patience = max(self._patience, self._iter_count*self._patience_increase)
                self._best_validation_cost = validation_cost
                self.save_object(filename)

        self._iter_count += 1
        return improved

        
    def save_object(self, filename):
        """
        Save object to file filename
        """
        print('\t Writing model to: %s \n' %filename)
#         dill.dump(self.get_GenModel(), file(filename + '_gen', 'wb'))
#         dill.dump(self.get_RecModel(), file(filename + '_rec', 'wb'))
        dill.dump(self, file(filename, 'wb'))
#         f.close()

