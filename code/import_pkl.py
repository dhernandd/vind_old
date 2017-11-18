'''
Code for importing the required packages for fitting NN models. can be used for any main code
'''
from __future__ import absolute_import

import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import lasagne       # the library we're using for NN's
# import the nonlinearities we might use 
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from numpy.random import *

#import cPickle
import sys
import os

# source_folder = '/Users/danielhernandez/Work/nlds/yj_newcode/flds/'
# sys.path.append(source_folder + 'code/lib/')
# sys.path.append(source_folder + 'code/')
# sys.path.append(source_folder + 'data/')
sys.path.append('./lib/')
from loadmat import loadmat

# I always initialize random variables here. 
msrng = RandomStreams(seed=20150503)
mnrng = np.random.RandomState(20150503)


# from lib.Trainable import *             # Class that gives SGVB object all the tools for training
# from lib.GenerativeModel import *       # Class file for generative models. 
# from lib.Recognitionmodel import *      # Class file for recognition models
from SGVB import SGVB                  # The meat of the algorithm - define the ELBO and initialize Gen/Rec model
# from lib.MiniBatchIterator import DatasetTrialIterator, DatasetTrialIteratorBatch

# This is a workaround for a weird bug in saving files. I don't even remember what it is. Just use it for now. 
sys.modules['mtrand'] = np.random.mtrand

theano.config.optimizer = 'fast_compile' 

