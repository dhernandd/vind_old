# Copyright 2017, Daniel Hernandez Diaz, Columbia University. 
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
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
from optparse import OptionParser
import cPickle as pickle

import numpy as np
import theano
import lasagne
from lasagne.nonlinearities import softmax, linear, softplus

# Local imports
from LatEvModels import LocallyLinearEvolution
from ObservationModels import GaussianObsTSGM, PoissonObsTSGM

from datetools import addDateTime
cur_date = addDateTime()

# ==============================================================================

LOCAL_RLT_ROOT = "/Users/danielhernandez/Work/time_series/vae_nlds_rec_algo_v2/data/"
RLT_DIR = "poisson_data_008/"
LOAD_DIR = LOCAL_RLT_ROOT + RLT_DIR # "/Users/danielhernandez/Work/time_series/vae_nlds_rec_algo_v2/data/gaussian_data_002/"
LOAD_FILE = "datadict"

YDIM = 10
XDIM = 2
TBINS = 30
NSAMPS = 100
OBSERVATIONS = 'Poisson'
NNODES = 60
ALPHA = 0.0

parser = OptionParser()

### DATA  PATHS, ETC ###
parser.add_option("--local_rlt_root", dest='local_rlt_root', default=LOCAL_RLT_ROOT,
                  help="The local directory containing all of your datasets.")
parser.add_option("--rltdir", dest='rlt_dir', default=RLT_DIR)  # fakedata_fit
parser.add_option("--load_dir", dest='load_dir', default=LOAD_DIR)  # fakedata_fit
parser.add_option("--load_file", dest='load_file', default=LOAD_FILE)  # fakedata_fit


# OPTIONS
parser.add_option("--ydim", dest='ydim', default=YDIM, type='int')
parser.add_option("--xdim", dest='xdim', default=XDIM, type='int')
parser.add_option("--tbins", dest='tbins', default=TBINS, type='int')
parser.add_option("--nsamps", dest='nsamps', default=NSAMPS, type='int')
parser.add_option("--obs", dest='observations', default=OBSERVATIONS,
                  help='The observation model. Types currently allowed are "Poisson" and "Gaussian".')

parser.add_option("--alpha", dest='alpha', default=ALPHA, type='float',
                  help='The scale of the nonlinearity. This hyperparameter is crucial. If set to 0.0, \
                  the code simply fits a linear dynamical system')

parser.add_option("--nnodes", dest='nnodes', default=NNODES, type='int')
parser.add_option("--genWscale", dest='genWscale', default=5.0, type='float')
parser.add_option("--genAscale", dest='genAscale', default=-0.1, type='float')
parser.add_option("--genrotscale", dest='genrotscale', default=0.1, type='float')
parser.add_option("--genangle", dest='genangle', default=90.0, type='float')
parser.add_option("--genQCholscale", dest='genQCholscale', default=2.0, type='float')
parser.add_option("--gennodes", dest='gennodes', default=40, type='float')
parser.add_option("--genbaserate", dest='genbaserate', default=5.0, type='float')

args = sys.argv
(options, args) = parser.parse_args(args)

# ==============================================================================

def npdict_to_theanodict(d):
    """
    Petit func to convert a dictionary of numpy arrays to one of theano shared variables initialized to the corresponding array values.
    """
    newd = {}
    for key in d.keys():
        if isinstance(d[key], np.ndarray):
#             newd[key] = theano.shared(value=d[key].astype(theano.config.floatX), name=key, borrow=True)
            newd[key] = theano.shared(value=d[key], name=key, borrow=True)
        else:
            newd[key] = d[key]
    return newd  # comprehension here? {a if P(i) else b for i in somerange}, meh


def write_option_file(rlt_dir):
    """
    """
    option_list = [option for option in dir(options) if not option.startswith('_')]
    with open(rlt_dir + 'options.txt', 'wb') as option_file:
        for option in option_list:
            if option not in ['ensure_value', 'read_file', 'read_module']:
                option_file.write(option + ' ' + str(getattr(options, option)) + '\n')


def generate_fake_data():
    """
    """    
    print('Generating some fake data, yay!\n')

    rlt_dir = options.local_rlt_root + options.rlt_dir
    if not os.path.exists(rlt_dir): os.makedirs(rlt_dir)

    write_option_file(rlt_dir)
    
    xdim = options.xdim
    ydim = options.ydim
 
    obs = options.observations
    nnodes = options.nnodes
    genNNEvolve = lasagne.layers.InputLayer((None, None, xdim))
    genNNEvolve = lasagne.layers.DenseLayer(genNNEvolve, nnodes, nonlinearity=softmax, 
                                            W=lasagne.init.Orthogonal(), num_leading_axes=2)
    genNNEvolve = lasagne.layers.DenseLayer(genNNEvolve, xdim**2, nonlinearity=linear, 
                                            W=lasagne.init.Uniform(0.9), num_leading_axes=2)
    
    latpars = dict([
#                     ('QInvChol', genQCholscale*np.diag(np.ones(xdim))),
#                     ('Q0InvChol', 1.0*np.diag(np.ones(xdim))),
#                     ('NNEvolve', genNNEvolve),
                    ('Alinear', np.array([[1.0, 0.1],[-0.1,1.0]])),
                    ('alpha', options.alpha)])
    LatParsDict = npdict_to_theanodict(latpars)
    lat_model = LocallyLinearEvolution(LatParsDict, xdim)
    

    genObsnl = {'Gaussian' : linear, 'Poisson' : softplus}
    genModel = {'Gaussian' : GaussianObsTSGM, 'Poisson' : PoissonObsTSGM}
    genNNMuY = lasagne.layers.InputLayer((None, None, xdim))
    genNNMuY = lasagne.layers.DenseLayer(genNNMuY, nnodes, nonlinearity=softplus, 
                                         W=lasagne.init.Orthogonal(), num_leading_axes=2)
    genNNMuY.W.set_value(3.0*genNNMuY.W.get_value())
    genNNMuY = lasagne.layers.DenseLayer(genNNMuY, ydim, nonlinearity=genObsnl[obs], 
                                         W=lasagne.init.Orthogonal(), num_leading_axes=2)
    
    ObsParsDict = dict([('NNMuY', genNNMuY),
                        ('NNMuY_W', 5.0),
                        ('is_out_positive', True) 
                        ])
    
    ObsParsDict = npdict_to_theanodict(ObsParsDict)
    gen_model = genModel[obs](ObsParsDict, ydim, xdim, lat_ev_model=lat_model)
    
    # Generate some data
    Tbins = options.tbins
    Nsamps = options.nsamps
    nsamps_train = 2*Nsamps/3
    valid_test = 5*Nsamps/6
    Ydata, Xdata = gen_model.sample_XY(Nsamps=Nsamps, Tbins=Tbins, withInflow=True)
    Yrate = gen_model.Rate.eval({gen_model.X : Xdata})
    print(Yrate[0])
    
#     print('Rates:', gen_model.Rate.eval({gen_model.X : Xdata})[0])
#     print('W:', gen_model.NNMuY.W.get_value())
    
    Ytrain, Xtrain = Ydata[:nsamps_train], Xdata[:nsamps_train]
    Yvalid, Xvalid = Ydata[nsamps_train:valid_test], Xdata[nsamps_train:valid_test]
    Ytest, Xtest = Ydata[valid_test:], Xdata[valid_test:]
     
    LD = gen_model.eval_LogDensity(Ydata, Xdata)
    print('LogDensity, LogDensity (X terms):', LD)
        
    lat_model.quiver2D_flow(pause=False, withInflow=False, savefile='linear_flow.jpg', scale=250)
    lat_model.plot_2Dquiver_paths(Xdata, rlt_dir + 'gen_paths', withInflow=True)
    
    with open(rlt_dir + 'datadict', 'wb+') as rlt_file:
        datadict = {'Ytrain' : Ytrain, 'Yvalid' : Yvalid, 'Xtrain' : Xtrain, 'Xvalid' : Xvalid,
                    'Yrate' : Yrate, 'Y_test' : Ytest, 'Xtest' : Xtest}
        pickle.dump(datadict, rlt_file)
        pickle.dump(gen_model, open(rlt_dir + 'gen_model', 'wb+'))


def load_data():
    """
    """
    load_dir = options.load_dir
    load_file = options.load_file
    
    with open(load_dir + load_file, 'rb+') as lfile:
        datadict = pickle.load(lfile)
        Ytrain = datadict['Ytrain']
        Xtrain = datadict['Xtrain']
        print(Xtrain[0,:,:])
        print(Ytrain[0])
        
        print(np.isnan(Ytrain).any())
        print(np.mean(Ytrain))


if __name__ == '__main__':
    
    generate_fake_data()
    load_data()