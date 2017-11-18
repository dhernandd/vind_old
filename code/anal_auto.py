"""
anal_auto.py
"""
import sys
from optparse import OptionParser
import numpy as np
# import cPickle as pickle
import dill

import theano
import theano.tensor as T

import lasagne

from theano.tensor.shared_randomstreams import RandomStreams
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify, sigmoid
import matplotlib.pyplot as plt

##### INTERNAL PACKAGES #####

from GMTimeSeries import NLDSv1_wWhiteNoise
from GaussianObsTS import GaussianObsTS_wDS
from PoissonObsTS import PoissonObsTS_wDS
GaussianObsTSGM = GaussianObsTS_wDS(NLDSv1_wWhiteNoise)
PoissonObsTSGM = PoissonObsTS_wDS(NLDSv1_wWhiteNoise)

from RMTimeSeries import SmoothingNLDSTimeSeries
from OptimizerVAEC_TS import VAEC_NLDSOptimizer

##### PREPARATORY CODE #####

theano.config.optimizer = 'None'

parser = OptionParser()

parser.add_option("--cluster", dest='cluster', default=1, type='int')  # set to 1 for running in the cluster
parser.add_option("--savedir", dest='savedir', default='oscillatordata001')
parser.add_option("--savefile", dest='savefile', default='oscillatordatadict')
parser.add_option("--mainobjfile", dest='mainobjfile', default='sgvb')

args = sys.argv
(options, args) = parser.parse_args(args)

sys.path.append('/vega/stats/users/dh2832/general_libs/') if options.cluster else sys.path.append('/Users/danielhernandez/Work/time_series/vae_nlds_rec_algo/') 

savedir = '/vega/stats/users/dh2832/general_libs/rslts/nldsTS/v3.0/' if options.cluster else '/Users/danielhernandez/Work/time_series/vae_nlds_rec_algo/rslts/nldsTS/v3.0/'
this_savedir = savedir + options.savedir + '/'
mainobjfile = this_savedir + options.mainobjfile
MainObj = dill.load(open(mainobjfile, 'rb+'))
mrec = MainObj.get_RecModel()
mgen = MainObj.get_GenModel()

datadir = '/vega/stats/users/dh2832/general_libs/data/' if options.cluster else '/Users/danielhernandez/Work/time_series/vae_nlds_rec_algo/data'
this_datadir = datadir + options.datadir + '/'
datafile = this_datadir + options.datafile
Data = dill.load(open(datafile, 'rb+'))

x1range = x2range = (-35.0, 35.0)
lattice = mgen.defineLattice(x1range, x2range)
Tbins = lattice.shape[0]
lattice = np.reshape(lattice, [1, Tbins, mgen.xDim])
quiver_f = mgen.eval_nextX(lattice, withInflow=False).reshape(Tbins-1, mgen.xDim)
quiver_i = lattice[:,:-1,:].reshape(Tbins-1, mgen.xDim)

Ytrain = Data['Ytrain']
OneDict = {'Xtrain' : MainObj.X_passed, 'Xvalid' : MainObj.X_valid_passed, 'Cost' : MainObj._costs, 'Cost_Valid' : MainObj._validations_costs,
           'Quiver_f' : quiver_f, 'Quiver_i' : quiver_i}

savefile = options.savefile
dill.dump(OneDict, open(this_savedir + 'onedict', 'wb+'))