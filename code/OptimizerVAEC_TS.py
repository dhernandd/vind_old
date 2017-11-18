# Copyright 2017 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
from __future__ import print_function

import sys
sys.setrecursionlimit(1500)
# from collections import OrderedDict
# import cPickle as pickle
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np

import theano
import theano.tensor as T
import theano.gof
import theano.compile
theano.config.optimizer = 'fast_compile'

import lasagne
from lasagne.nonlinearities import sigmoid, linear, softmax

from MinibatchIterator import DatasetTrialIterator
from Trainable import Trainable
from LatEvModels import LinearEvolution, LocallyLinearEvolution


def collect_shared_vars(expressions):
    """Returns all shared variables the given expression(s) depend on.
    Parameters
    ----------
    expressions : Theano expression or iterable of Theano expressions
        The expressions to collect shared variables from.
    Returns
    -------
    list of Theano shared variables
        All shared variables the given expression(s) depend on, in fixed order
        (as found by a left-recursive depth-first search). If some expressions
        are shared variables themselves, they are included in the result.
    """
    # wrap single expression in list
    if isinstance(expressions, theano.Variable):
        expressions = [expressions]
    # return list of all shared variables
    return [v for v in theano.gof.graph.inputs(reversed(expressions))
            if isinstance(v, theano.compile.SharedVariable)]
    

def data_shuffle(Y, X):
    C = zip(Y, X)
    shuffle(C)
    Y, X = zip(*C)
    return np.array(Y), np.array(X)

    

class Optimizer_TS(Trainable):
    """
    """
    def __init__(self, ParsDicts, LATCLASS, OBSCLASS, yDim, xDim, RECCLASS=None, Y=None, X=None, common_lat=False):
        """
        """
        Trainable.__init__(self)
        
        self.yDim = yDim
        self.xDim = xDim
        
        self.Y = Y = T.tensor3('Y') if Y is None else Y
        self.X = X = T.tensor3('X') if X is None else X
        self.x = T.matrix('x')

        LatPars = ParsDicts['LatPars']
        self.common_lat = common_lat
        if common_lat:
            self.lat_ev_model = lat_ev_model = LATCLASS(LatPars, xDim, X)
        else:
            lat_ev_model = None
        
        ObsPars = ParsDicts['ObsPars']
        GENLATCLASS = ObsPars['LATCLASS'] if 'LATCLASS' in ObsPars else None
        self.mgen = OBSCLASS(ObsPars, yDim, xDim, Y, X, lat_ev_model=lat_ev_model, LATCLASS=GENLATCLASS)
        
        RecPars = ParsDicts['RecPars']
        RECLATCLASS = RecPars['LATCLASS'] if 'LATCLASS' in RecPars else None
        if RECCLASS is not None: self.mrec = RECCLASS(RecPars, yDim, xDim, Y, X, lat_ev_model, LATCLASS=RECLATCLASS) 
    
       
    def get_GenModel(self):
        return self.mgen
    

    def get_RecModel(self):
        return self.mrec
    
    
    def get_Y(self):
        """
        For outside usage.
        """
        return self.Y


    def get_X(self):
        """
        For outside usage.
        """
        return self.X
    

    def get_Params(self, costname):
        return self.ParamsDict[costname]() + self.lat_ev_model.get_Params() if self.common_lat else self.ParamsDict[costname]() 

        
    def fit(self):
        raise Exception('\nPlease define me. This is an abstract method.')

    
    
    
class VAEC_LDSOptimizer(Optimizer_TS):
    """
    """
    def __init__(self, ParsDicts, LATMODEL, OBSMODEL, yDim, xDim, RECMODEL=None, Y=None, X=None):
        """
        """
        Optimizer_TS.__init__(self, ParsDicts, LATMODEL, OBSMODEL, yDim, xDim, RECCLASS=RECMODEL, Y=Y, X=X)
                            
        OptPars = ParsDicts['OptPars']
        if 'NN_NVIL_baseline' not in OptPars.keys():
            nnodes = 40
            NN_NVIL_baseline = lasagne.layers.InputLayer((None, self.yDim))
            NN_NVIL_baseline = lasagne.layers.DenseLayer(NN_NVIL_baseline, nnodes, nonlinearity=sigmoid)  # store last layer as a property to be able to recover all others
            self.NN_NVIL_baseline = lasagne.layers.DenseLayer(NN_NVIL_baseline, 1, nonlinearity=linear)  # store last layer as a property to be able to recover all others
        self.NVIL_baseline = lasagne.layers.get_output(self.NN_NVIL_baseline, self.Y).flatten()
        
        Ybatch = T.tensor3('Ybatch')
        Xbatch = T.tensor3('Xbatch')
        self.CostsDict = {'GenModelLogDensity' : self.cost_GenModelLogDensity, 'ELBO' : self.cost_ELBO}#, 'ELBO_wNVIL' : self.costs_ELBO_wNVIL}
        self.ParamsDict = {'GenModelLogDensity' : lambda : self.mgen.get_Params(), 
                            'ELBO' : lambda : self.mgen.get_Params() + self.mrec.get_Params()}
#                            'ELBO_wNVIL' : self.mgen.get_Params() + self.mrec.get_Params() + self.get_NVIL_Params()}
        self.CostsInputDict = {'GenModelLogDensity' : [theano.In(self.Y), theano.In(self.X)],
                               'ELBO' : [theano.In(self.Y)], 'ELBO_wNVIL' : [theano.In(self.Y)]}
#         self.ParametersSet = {'ObsCovariance' : self.mgen.get_ObsCovarianceParams(),
#                               'LogDensity' : self.mgen.get_Params(),
#                               'Entropy' : self.mrec.get_Params_Entropy(),
#                               'Baseline' : lasagne.layers.get_all_params(self.NN_NVIL_baseline)}
        
        self.StoreDict = {'costname' : [], 'cost_valid' : []}
                
    
    def cost_GenModelLogDensity(self, padleft=False):
        """
        """
        mgen = self.get_GenModel()
        Nsamps = self.Y.shape[0]
        LogDensity = mgen.compute_LogDensity(self.Y, self.X, padleft=padleft)
        
        costs_func = theano.function(inputs=self.get_CostInputs('GenModelLogDensity'), outputs=[LogDensity/Nsamps])
        
        return LogDensity, costs_func
    
    
    def cost_ELBO(self, padleft=False):
        """
        """
        Nsamps = self.Y.shape[0]
        postX = self.mrec.symbsample_X(self.Y)
        
        LogDensity = self.mgen.compute_LogDensity(self.Y, postX, padleft=padleft)
        Entropy = self.mrec.compute_Entropy(self.Y)
        
        ELBO = LogDensity + Entropy
        
        costs_func = theano.function(inputs=self.get_CostInputs('ELBO'), outputs=[ELBO/Nsamps, LogDensity/Nsamps, Entropy/Nsamps], mode='FAST_RUN')
        
        return ELBO, costs_func

    
    def eval_cost(self, Ydata, Xdata=None, costname='GenModelLogDensity', **kwargs):
        cost_func = self.CostsDict[costname](**kwargs)[1]  # costs_func always appear in the index 1
        if costname in ['GenModelLogDensity']:
            return cost_func(Ydata, Xdata)[0] # Actual cost_func is always at index 0 of costs_func
        else:
            return cost_func(Ydata)[0]
    

    def get_Trainer(self, costname='GenModelLogDensity', method='adam', padleft=False):
        """ 
        """
        lr = T.scalar('lr')

        the_cost = self.CostsDict[costname](padleft=padleft)[0]  # the costname always appears at the index 0 of the cost_... method returned value
        
        print('Params', self.get_Params(costname))
        updates = {'adam' : lasagne.updates.adam(-the_cost, self.get_Params(costname), learning_rate=lr)}
        
        train_fn = theano.function(inputs=self.get_CostInputs(costname) + [theano.In(lr)],  outputs=the_cost, 
                                   updates=updates[method], mode=theano.Mode(linker='vm'), on_unused_input='warn') # 

        return train_fn 


    def fit(self, y_train, x_train=None, y_valid=None, x_valid=None, batch_size=1, method='adam', costname='GenModelLogDensity', max_norm=0,
            max_epochs=20, learning_rate=1e-13, MinibatchIterator=DatasetTrialIterator, adalr=1.0, eps=1e-6, rho=0.95, padleft=False):
        """
        """
        import time
        if costname in ['ELBO_wNVIL']:
            param_updater = self.get_NVIL_SGDTrainer(costname=costname)
        else:
            param_updater = self.get_Trainer(costname=costname, method=method, padleft=padleft) 
                
        epoch = 0
        t0 = time.time()
        self.simplechecks(y_train, Xdata=x_train, Yvalid=y_valid, Xvalid=x_valid, costname=costname)
        print('Simple Checks duration:', time.time() - t0)
        while epoch < max_epochs:
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ max_epochs))
            sys.stdout.flush()
            
            if costname in ['GenModelLogDensity']:
                train_set_iterator = MinibatchIterator(y_train, Xdata=x_train, batch_size=batch_size)
            else:
                train_set_iterator = MinibatchIterator(y_train, batch_size=batch_size)
            
            for data in train_set_iterator:
                t1 = time.time()
                if costname in ['GenModelLogDensity']:                        
                    y_data, x_data = data
                    param_updater(y_data, x_data, lr=learning_rate) # The training step for this data.
                else:
                    y_data = data
                    param_updater(y_data, lr=learning_rate) # The training step for this data.
                print('Training step duration:', time.time() - t1)
            t2 = time.time()
            self.simplechecks(y_train, x_train, Yvalid=y_valid, Xvalid=x_valid, costname=costname)
            print('Simple Checks duration:', time.time() - t2)
            
            epoch += 1 


    def simplechecks(self, Ydata, Xdata=None, Yvalid=None, Xvalid=None, costname='ELBO', param_set='ObsCovariance'):
        """
        Recommended use only for testing.
        """
        mgen = self.get_GenModel()
        if costname == 'ELBO':
            _, costsfunc = self.cost_ELBO()
            ELBO, LD, E = costsfunc(Ydata)
            print('ELBO, LD, E:', ELBO, LD, E) 
            if Yvalid is not None:
                ELBOv, LDv, Ev = costsfunc(Yvalid)
                print('ELBO (validation):', ELBOv, LDv, Ev)
        elif costname == 'GenModelLogDensity':
            LD_Yterms = mgen.eval_LogDensity_Yterms(Ydata, Xdata)
            LD_Xterms = mgen.eval_LogDensity_Xterms(Xdata)
            LD = mgen.eval_LogDensity(Ydata, Xdata)
            LD_valid = mgen.eval_LogDensity(Yvalid, Xvalid)
            print( 'Actual LogDensity:', LD)
            print( 'Actual LogDensity (Xterms):', LD_Xterms[0], LD_Xterms[1], LD_Xterms[2], LD_Xterms[3], LD_Xterms[4]) 
            print( 'Actual LogDensity (Yterms):', LD_Yterms[0], LD_Yterms[1], LD_Yterms[2])
            print( 'Actual LogDensity (Validation):', LD_valid)
            self.StoreDict['costname'].append(LD)
            self.StoreDict['costname'].append(LD_valid)
#             print 'Fitted A:', mgen.eval_A()
#             print 'Grad:', self.eval_grads(Ydata, Xdata, costname, param_set)
        elif costname == 'ELBO_wNVIL':
            ELBO, LD, E, _, _ = self.eval_costs_ELBO_wNVIL(Ydata)
            self.StoreDict['costname'].append(ELBO)
            if Yvalid is not None:
                self.StoreDict['cost_valid'].append(self.eval_costs_ELBO_wNVIL(Yvalid)[0])
            print('ELBOs:', ELBO, LD, E)





class VAEC_NLDSOptimizer(Optimizer_TS):
    """
    """
    def __init__(self, ParsDicts, LATCLASS, OBSCLASS, yDim, xDim, RECCLASS=None, Y=None, X=None, common_lat=False, rslt_dir=None):
        """
        """
        Optimizer_TS.__init__(self, ParsDicts, LATCLASS, OBSCLASS, yDim, xDim, RECCLASS=RECCLASS, Y=Y, X=X, common_lat=common_lat)
        
        OptPars = ParsDicts['OptPars']
        for key in OptPars.keys():
            setattr(self, key, OptPars[key])
        if not hasattr(self, 'NN_NVIL_baseline'):
            nnodes = 40
            NN_NVIL_baseline = lasagne.layers.InputLayer((None, self.yDim))
            NN_NVIL_baseline = lasagne.layers.DenseLayer(NN_NVIL_baseline, nnodes, nonlinearity=sigmoid)  # store last layer as a property to be able to recover all others
            self.NN_NVIL_baseline = lasagne.layers.DenseLayer(NN_NVIL_baseline, 1, nonlinearity=linear)  # store last layer as a property to be able to recover all others
        self.NVIL_baseline = lasagne.layers.get_output(self.NN_NVIL_baseline, self.Y).flatten()
        
        self.rslt_dir = rslt_dir
        
        self.CostsInputDict = {'ELBO' : [theano.In(self.Y), theano.In(self.X)], 'ELBO_wNVIL' : [theano.In(self.Y)], 
                               'Entropy' : [theano.In(self.Y), theano.In(self.X)], 'LogDensity' : [theano.In(self.Y), theano.In(self.X)]}
        
        self.ParamsDict = {'ELBO' : lambda : self.mgen.get_Params() + self.mrec.get_Params(),
                           'Entropy' : lambda : self.mrec.get_Params_Entropy(),
                           'LogDensity' : lambda : self.mgen.get_Params() + self.mrec.get_Params(),
                           'ELBO_wNVIL' : lambda : self.mgen.get_Params() + self.mrec.get_Params() + self.get_NVIL_Params()}
        self.CostsDict = {'LogDensity' : self.cost_LogDensity, 
#                           'ELBO_wNVIL' : self.costs_ELBO_wNVIL, 
                          'ELBO' : self.cost_ELBO,
                          'Entropy' : self.cost_Entropy}
        
        self.StoreDict = {'costname' : [], 'cost_valid' : []}


    def __getstate__(self):
        """
        Terrible hack. Needed it because this fucker was not fucking pickling.
        Max recursion limit exceeded, blah blah blah... Fucker.
        """
        return {key : value for key, value in self.__dict__.iteritems() if key not in ['fit']}


    def get_CostInputs(self, costname):
        return self.CostsInputDict[costname]


    def cost_Entropy(self, Y=None, X=None, padleft=False):
        """
        """
        if Y is None: Y = self.Y
        if X is None: X = self.X
        
        mrec = self.get_RecModel()
        postX = theano.clone(mrec.symbsample_X(Y, X))   # Posterior is computed symbolically inside. OK!            

        Nsamps = Y.shape[0]
        
        Entropy = mrec.compute_Entropy(Y, postX)
        costs_func = theano.function(inputs=self.CostsInputDict['Entropy'], outputs=[Entropy/Nsamps])
        
        return Entropy, costs_func


    def cost_LogDensity(self, Y=None, X=None, padleft=False):
        """
        """
        if Y is None: Y = self.Y
        if X is None: X = self.X
        
        mrec = self.get_RecModel()
        mgen = self.get_GenModel()
        postX = theano.clone(mrec.symbsample_X(Y, X))   # Symbolic posterior

        Nsamps = Y.shape[0]
        
        LogDensity = mgen.compute_LogDensity(Y, postX, padleft=padleft)

        costs_func = theano.function(inputs=self.CostsInputDict['LogDensity'], outputs=[LogDensity/Nsamps])
        
        return LogDensity, costs_func

    
    def get_symb_postX(self, Y, X, sample_strategy='with_symb_noise'):
        """
        """
        if sample_strategy == 'with_symb_noise':
            return theano.clone(self.mrec.symbsample_X(Y, X))
        elif sample_strategy == 'no_noise':
            return theano.clone(self.mrec.postX, replace={self.Y : Y, self.X : X})
        else:
            assert False, 'Sampling strategy not implemented!'
        
        
    def cost_ELBO(self, Y=None, X=None, padleft=False, sample_strategy='with_symb_noise',
                  regularize_evolution_weights=False):
        """
        """
        if Y is None: Y = self.Y
        if X is None: X = self.X
        
        mrec = self.get_RecModel()
        mgen = self.get_GenModel()
        postX = self.get_symb_postX(Y, X, sample_strategy)
        
        if regularize_evolution_weights:
            from lasagne.layers import get_all_layers
            from lasagne.regularization import regularize_layer_params, l2
            lat_ev_layers = get_all_layers(self.lat_ev_model.NNEvolve) 
            lat_weights_regloss = regularize_layer_params(lat_ev_layers[1], l2)

        Nsamps = Y.shape[0]
        LogDensity = mgen.compute_LogDensity(Y, postX, padleft=padleft) 
        Entropy = mrec.compute_Entropy(Y, postX)
        ELBO = (LogDensity + Entropy if not regularize_evolution_weights else 
                LogDensity + Entropy + lat_weights_regloss)
        costs_func = theano.function(inputs=self.CostsInputDict['ELBO'], 
                                     outputs=[ELBO/Nsamps, LogDensity/Nsamps, Entropy/Nsamps])
        
        return ELBO, costs_func

    
    def eval_cost(self, Ydata, Xdata=None, costname='ELBO', **kwargs):
        cost_func = self.CostsDict[costname](**kwargs)[1]  # costs_func always appear in the index 1
        return cost_func(Ydata, Xdata)[0]


    def get_Trainer(self, costname='ELBO', method='adam', padleft=False, 
                    sample_strategy='with_symb_noise'):
        """ 
        Returns a theano.function with updates that trains the statistical model (Adam trainer).
        
        Evaluate this output function as in:
            f(y, lr)
        where y is the data to be fitted and lr is the desired learning rate.
        """
        lr = T.scalar('lr')
        adlr = T.scalar('adlr')
        rho = T.scalar('rho')
        eps = T.scalar('eps')
        
        the_cost = self.CostsDict[costname](sample_strategy=sample_strategy)[0]  # the costname always appears at the index 0 of the cost_... method
        
        
        print('Params', self.get_Params(costname))
        updates = {'adam' : lasagne.updates.adam(-the_cost, self.get_Params(costname), learning_rate=lr),
                   'adadelta' : lasagne.updates.adadelta(-the_cost, self.get_Params(costname), learning_rate=adlr, rho=rho, epsilon=eps)}
        updates_Inputs = {'adam' : [theano.In(lr)],
                          'adadelta' : [theano.In(adlr), theano.In(rho), theano.In(eps)]}
        
        with open('./debugprint', 'wb+') as debugfile:
            so = sys.stdout
            sys.stdout = debugfile
            theano.printing.debugprint(the_cost)
            sys.stdout = so
        train_fn = theano.function(inputs=self.get_CostInputs(costname) + updates_Inputs[method],  outputs=the_cost, 
                                   updates=updates[method], mode=theano.Mode(linker='vm'), on_unused_input='warn') 

        return train_fn 


    def fit(self, y_train, x_train=None, y_valid=None, x_valid=None, batch_size=1, method='adam', costname='ELBO', max_norm=0,
            max_epochs=20, learning_rate=1e-3, MinibatchIterator=DatasetTrialIterator, adlr=1.0, eps=1e-6, rho=0.95, padleft=False,
            rslt_dir=None, end_lr=1e-4):
        """
        """
        method_params_dict = {'adam' : {'lr' : learning_rate}, 
                              'adadelta' : {'adlr' : adlr, 'rho' : rho, 'eps' : eps}}
        method_params = method_params_dict[method]

        self.rslt_dir = rslt_dir
        if costname in ['ELBO_wNVIL']:
            param_updater = self.get_NVIL_SGDTrainer(costname=costname)
        else:
            param_updater = self.get_Trainer(costname=costname, method=method, padleft=padleft)            
                
        Nsamps = y_train.shape[0]
        Tbins = y_train.shape[1]
        X_passed = np.zeros((Nsamps, Tbins, self.xDim))
        if y_valid is not None:
            Nsamps_valid = y_valid.shape[0]
            X_valid_passed = np.zeros((Nsamps_valid, Tbins, self.xDim))
        
        mrec = self.get_RecModel()
        if self.xDim == 2:
            self.lat_ev_model.quiver2D_flow(pause=False)
            plt.savefig(rslt_dir + 'quiver_init')
        
        epoch = 0
        x_init = False
        _, costsfunc = self.cost_ELBO()
        cost = -np.inf
        exp_lr_decrease_rate = learning_rate/end_lr
        while epoch < max_epochs:
            param_updater = param_updater
            batch_counter = 0
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ max_epochs))
            sys.stdout.flush()
            
            X_passed = mrec.eval_Mu(y_train) if not x_init else mrec.eval_postX(y_train, X_passed)
#             print('Computing posterior...')
#             X_passed = mrec.eval_Mu(y_train) if not x_init else Xsol_func(y_train, X_passed)
#             print('postX: ', X_passed[0,0,0])
#             y_train, X_passed = data_shuffle(y_train, X_passed)
            if y_valid is not None: X_valid_passed = (mrec.eval_Mu(y_valid) if not x_init 
                                else mrec.eval_postX(y_valid, mrec.eval_postX(y_valid, X_valid_passed)) )
            else: X_valid_passed = None
            x_init = True
            
            dec_learning_rate = True
            if dec_learning_rate and epoch >= 1:
                new_lr = learning_rate*np.exp(-epoch*np.log(exp_lr_decrease_rate)/max_epochs)
                print('Decreasing learning rate to', new_lr)
                method_params_dict['adam']['lr'] = new_lr
                
            train_set_iterator = MinibatchIterator(y_train, X_passed, batch_size=batch_size)
            for y_data, x_passed in train_set_iterator:
                param_updater(y_data, x_passed, **method_params)    # The training step for this y_data.
#                 if batch_counter % 40 == 0:
#                     print('Batch:', batch_counter, ' of', Nsamps/batch_size)
#                     cost, LD, E = costsfunc(y_data, x_passed)
#                     print('ELBO, LD, E:', cost, LD, E)
                batch_counter += 1
            
            plt_file = rslt_dir + 'datafit' + str(epoch)
            if self.xDim ==2 and epoch % 1 == 0:
                self.plot_2Dquiver_paths(Xdata=X_passed, rslt_file=plt_file, epoch=epoch, skipped=1)
            
            cost, cost_valid = self.simplechecks(y_train, Xdata=X_passed, Yvalid=y_valid, 
                                                 Xvalid=X_valid_passed, costname=costname)            
            self.append_log(cost, cost_valid, filename=rslt_dir + 'sgvb')
            
            epoch += 1 
            
#         bestfit = {'Xtrain' : X_passed, 'Xvalid' : X_valid_passed}
#         pickle.dump(bestfit, open(rslt_dir + 'bestfit', 'wb+'))


    def simplechecks(self, Ydata, Xdata=None, Yvalid=None, Xvalid=None, 
                     costname='ELBO', param_set='ObsCovariance'):
        """
        Recommended use only for testing.
        """
        mgen = self.get_GenModel()
        if costname == 'ELBO':
            _, costsfunc = self.cost_ELBO()
            cost, LD, E = costsfunc(Ydata, Xdata)
            print('ELBO, LD, E:', cost, LD, E)
            if Yvalid is not None:
                cost_valid, LDv, Ev = costsfunc(Yvalid, Xvalid)
                print('ELBO (validation):', cost_valid, LDv, Ev)
            else: cost_valid = None
        elif costname == 'GenModelLogDensity':
            LD_Yterms = mgen.eval_LogDensity_Yterms(Ydata, Xdata)
            LD_Xterms = mgen.eval_LogDensity_Xterms(Xdata)
            LD = mgen.eval_LogDensity(Ydata, Xdata)
            LD_valid = mgen.eval_LogDensity(Yvalid, Xvalid)
            print('Actual LogDensity:', LD)
            print('Actual LogDensity (Xterms):', LD_Xterms[0], LD_Xterms[1], LD_Xterms[2], LD_Xterms[3], LD_Xterms[4]) 
            print('Actual LogDensity (Yterms):', LD_Yterms[0], LD_Yterms[1], LD_Yterms[2])
            print('Actual LogDensity (Validation):', LD_valid)
            self.StoreDict['costname'].append(LD)
            self.StoreDict['costname'].append(LD_valid)
#             print 'Fitted A:', mgen.eval_A()
#             print 'Grad:', self.eval_grads(Ydata, Xdata, costname, param_set)
        elif costname == 'ELBO_wNVIL':
            ELBO, LD, E, _, _ = self.eval_costs_ELBO_wNVIL(Ydata)
            self.StoreDict['costname'].append(ELBO)
            if Yvalid is not None:
                self.StoreDict['cost_valid'].append(self.eval_costs_ELBO_wNVIL(Yvalid)[0])
            print('ELBOs:', ELBO, LD, E)
            
        return cost, cost_valid


    def plot_2Dquiver_paths(self, Xdata, rslt_file, epoch, withInflow=False, skipped=1):
        """
        """
        mlat = self.lat_ev_model
        axes = mlat.plot2D_sampleX(Xdata=Xdata, pause=False, draw=False, newfig=True, skipped=skipped)
        x1range, x2range = axes.get_xlim(), axes.get_ylim()
        axes.set_title(str(epoch), fontdict={'fontsize':40})
        s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        mlat.quiver2D_flow(pause=False, x1range=x1range, x2range=x2range, 
                           scale=s, newfig=False, withInflow=withInflow)
        plt.savefig(rslt_file)
        plt.close()
