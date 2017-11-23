# Copyright 2017, Daniel Hernandez Diaz, Columbia University. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
from __future__ import absolute_import

# exec(open("import_pkl.py").read())
import sys
sys.setrecursionlimit(100000)
sys.path.append('../')

import os
from optparse import OptionParser
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')  #  Use this in the server

import numpy as np
import theano
theano.config.optimizer = 'fast_compile'
import lasagne
from lasagne.nonlinearities import softmax, linear, softplus, tanh, sigmoid
from lasagne.layers import InputLayer, DenseLayer
from lasagne.init import Orthogonal, Uniform, Normal

# Local imports
import vind
from code.ObservationModels import GaussianObsTSGM, PoissonObsTSGM
from code.LatEvModels import LocallyLinearEvolution

from code.RecognitionModels import SmoothingNLDSTimeSeries
from code.OptimizerVAEC_TS import VAEC_NLDSOptimizer

# from MinibatchIterator import DatasetTrialIterator

from code.datetools import addDateTime
cur_date = addDateTime()

#=== GLOBALS ===#
# AS A USER YOU SHOULD BE ABLE TO GET THIS RUNNING JUST BY CHANGING THESE
# PLEASE READ THE HELP ON EACH OF THESE OPTIONS IN THE PARSER.

### HARDWARE, META
CLUSTER = 0
EXTERNAL_DATA = 1

### DATA  PATHS, ETC ###
LOCAL_DATA_ROOT = "data/"
THIS_DATA_DIR = "poisson_data_002" #  lorenzdata004
IS_DICT = 1
DATA_FILE = 'datadict'
TEST_FILE = None
LOCAL_RLT_ROOT = 'rslts/'
RLT_DIR = 'poisson002_fit'

CLUSTER_DATA_ROOT = "./" 


### MODEL HYPERPARAMETERS  AND GRAPH DEFINITION ###
OBSERVATIONS = 'Poisson'
ALPHA = 0.1
XDIM = 2
MUX_DEPTH = 2
LAMBDAX_DEPTH = 2
MUY_DEPTH = 2
IS_OUT_POSITIVE = 0
ACT_FUNC_HLS = 'softplus'
OBS_OUTPUT_NL = 'linear'
NNODES = 60

### OPTIMIZATION ###
LR = 3e-3
BATCH_SIZE = 1
EPOCHS = 200
METHOD = 'adam'
COMMON_LAT = 1

# =================================================================

parser = OptionParser()

### HARDWARE, META ###
parser.add_option("--cluster", dest='cluster', default=CLUSTER, type='int',
				  help='Only for developer use. Leave this to 0 if you are a user.')  # set to 1 for running in the cluster
parser.add_option("--extdata", dest='extdata', default=EXTERNAL_DATA, type='int',
				  help='Would you like this code to generate its own data (0) or \
				  are you providing your own data, external to this code (1)? \
				  DATA GENERATION MOVED TO A SEPARATE SCRIPT! SEE generate_data.py')

### DATA  PATHS, ETC ###
# The code assumes the data is organized locally in a directory hierarchy of the form
# LOCAL_DATA_ROOT/THIS_DATA_DIR/datafile
parser.add_option("--local_data_root", dest='local_data_root', default=LOCAL_DATA_ROOT,
				  help="The local directory containing all of your datasets.")
# Options whose name start with cluster are for the convenience of the developer only. 
# If you are a user, just set CLUSTER=0 and forget about the rest of them. 
parser.add_option("--cluster_data_root", dest='cluster_data_root', default=CLUSTER_DATA_ROOT,
				  help="The cluster directory containing all of your datasets.")
parser.add_option("--datadir", dest='datadir', default=THIS_DATA_DIR,
				help="The directory inside your local_data_root that contains the \
				specific pickled dataset to be used in this run. Datasets should \
				consist of pickled numpy tensors or python dictionaries with numpy \
				tensors as values (see below).")

parser.add_option("--is_dict", dest='is_dict', default=IS_DICT, type='int',
				help="The preferred way of feeding the data is a dictionary with \
				keys 'Ytrain', 'Yvalid' and optionally 'Ytest'; corresponding to \
				the training, validation and test data respectively. To be absolutely \
				clear, your datafile below should almost always be a pickled dictionary \
				at least, like so {'Ytrain' : Ytraindata, 'Yvalid' : Yvaliddata} where \
				Ytraindata and Yvaliddata are numpy tensors. These tensors shall be 3D, \
				with shape [N, T, d_X] where N is the number of trials, T is the number \
				of time bins and d_X is the data dimension.")
parser.add_option("--datafile", dest='datafile', default=DATA_FILE,
				help="The file with the pickled data. Preferably a dictionary as \
				detailed above. However, a pickled 3D tensor of just training data can also \
				be passed here, in which case you should set is_dict=0.") # train_data
parser.add_option("--vdatafile", dest='vdatafile', default='valid_data',
				help="For use in case your training data and your validation data \
				are pickled numpy tensors in separate files. This is not the preferred \
				way. Pass all the data in a dictionary when possible!")
parser.add_option("--test_file", dest='test_file', default=TEST_FILE)

parser.add_option("--local_rlt_root", dest='local_rlt_root', default=LOCAL_RLT_ROOT,
				  help="The local directory that contains the results for all your runs.")
parser.add_option("--rltdir", dest='rlt_dir', default=RLT_DIR, 
				help="The specific directory inside local_rlt_root where the results \
				of this run will be saved. At runtime, this directory will be created \
				and the current date will be appended to its name.")


### MODEL HYPERPARAMETERS AND GRAPH DEFINITION ###
parser.add_option("--nlrecog", dest='NLRecog', default=1, type='int',
				help="Set to 1 for nonlinear recognition. Most likely that is\
				what you want.")
parser.add_option("--commonlat", dest='commonlat', default=COMMON_LAT, type='int',
				  help="Are the Recognition Model and the Generative Model sharing \
				  the Latent Dynamical System? If yes set to 1. \
				  (COMMON_LAT = 0 NOT IMPLEMENTED!")	 	# TODO:
parser.add_option("--xdim", dest='xdim', default=XDIM, type='int',
				  help='The dimension of the latent space.')
parser.add_option("--obs", dest='observations', default=OBSERVATIONS,
				  help="The observation model. Currently allowed are Poisson' \
				  and 'Gaussian''.")
parser.add_option("--is_out_positive", dest='is_out_positive', default=IS_OUT_POSITIVE, type='int',
				  help="Use in conjunction with Poisson observations. If set to 1 the output of the NN \
				  is positive definite (via a softplus nonlinearity) and can be used directly as a \
				  Poisson rate. Otherwise, if set to 0, an exponential layer has to be added.")

parser.add_option("--alpha", dest='alpha', default=ALPHA, type='float',
				  help='The scale of the nonlinearity. If set to 0.0, \
				  the code simply fits a linear dynamical system. When set to a value different \
				  from 0.0, it plays a role akin to that of the learning rate for nonlinear \
				  dynamics training. This hyperparameter is CRUCIAL. If your code is not \
				  converging a too large or too small value here is probably the culprit. \
				  Also check the learning rate.')

parser.add_option("--act_func_hls", dest='act_func_hls', default=ACT_FUNC_HLS,
				  help="The activation function for the hidden layers of the NNs.")
parser.add_option("--output_nl", dest='output_nl', default=OBS_OUTPUT_NL,
				  help="The nonlinearity of the last layer for the Generative model. \
				  For Poisson observations this can be set to softplus to generate a \
				  positive rate.")
parser.add_option("--mux_depth", dest='mux_depth', default=MUX_DEPTH, type='int',
				  help='The depth of the recognition NN, mapping the data at time t, Y_t \
				  to the mean of the value X_t of the corresponding latent state.')
parser.add_option("--lbdaxdepth", dest='lbdaxdepth', default=LAMBDAX_DEPTH, type='int',
				  help='The depth of the recognition NN, mapping the data at time t, Y_t \
				  to the variance \Lambda_t of the corresponding latent state.')
parser.add_option("--muydepth", dest='muydepth', default=MUY_DEPTH, type='int')

parser.add_option("--nnodesMevlv", dest='nnodesMevlv', default=200, type='int')
parser.add_option("--outbgenscale", dest='outbgenscale', default=1.0, type='float')
parser.add_option("--outWgenscale", dest='outWgenscale', default=1.0, type='float')
parser.add_option("--nnodesMrec", dest='nnodesMrec', default=NNODES, type='int')
parser.add_option("--outWrecscale", dest='outWrecscale', default=1.0, type='float')


### OPTIMIZATION ###
parser.add_option("--lr", dest='learning_rate', default=LR, type='float',
				  help='The starting learning rate for adam and the like SGD methods. \
				  Typically a value in the range [1e-2, 1e-4]. This parameter is CRUCIAL.\
				  If your code is not converging, a too large or too small value here \
				  is probably the culprit. Also check alpha.')
parser.add_option("--bsize", dest='batch_size', default=BATCH_SIZE, type='int',
				  help='The batch size used during training.')
parser.add_option("--maxepochs", dest='maxepochs', default=EPOCHS, type='int',
				  help='The number of epochs to run the gradient descent.')
parser.add_option("--method", dest='method', default=METHOD,
				  help='The GD algorithm. Currently "adam", "adadelta" are implemented.')
parser.add_option("--adalr", dest='adalr', default=0.99, type='float',
				  help='Adagrad learning rate.')
parser.add_option("--eps", dest='eps', default=1e-6, type='float')
parser.add_option("--rho", dest='rho', default=0.95, type='float')
parser.add_option("--cost", dest='cost', default='ELBO',
				  help='The code can optimize w.r.t. different costs. This is \
				  mostly useful for debugging. For applications, just set to "ELBO".')
parser.add_option("--fititers", dest='fititers', default=1, type='int')


args = sys.argv
(options, args) = parser.parse_args(args)

#========================================================================


# This is a workaround for a weird bug in saving files. I don't even remember
# what it is. Just use it for now.
sys.modules['mtrand'] = np.random.mtrand


def build_hprms_dict(options):
	"""
	Builds a dictionary from the options gathered by the parser.
	Args:
		options: Options object
	Output:
		d: A dictionary with the options as key value pairs.
		
	TODO: Finish this
	"""
	d = {}

	### DATA  PATHS, ETC
	d['local_data_root'] = options.local_data_root
	d['datadir'] = options.datadir
	d['local_rlt_root'] = options.local_rlt_root
	d['rlt_dir'] = options.rlt_dir
	
	d['cluster_data_root'] = options.cluster_data_root
	
	### MODEL HYPERPARAMETERS AND GRAPH DEFINITION ###
	d['xDim'] = options.xdim
	d['obs'] = options.observations
	
	d['mux_depth'] = options.mux_depth
	d['act_func_hls'] = options.act_func_hls
	d['output_nl'] = options.output_nl
	d['nnodesMevlv'] = options.nnodesMevlv
	d['nnodesMrec'] = options.nnodesMrec
	d['muydepth'] = options.muydepth
	d['outWgenscale'] = options.outWgenscale
	d['outbgenscale'] = options.outbgenscale
	d['is_out_positive'] = options.is_out_positive
	d['lbdaxdepth'] = options.lbdaxdepth
	
	### OPTIMIZATION ###
	d['learning_rate'] = options.learning_rate
	d['maxepochs'] = options.maxepochs
	d['cost'] = options.cost
	d['method'] = options.method
	d['batch_size'] = options.batch_size
	
	return d


class hprms_dict_to_obj(dict):
	"""
	Makes a simple object out of a dictionary to mimic object behavior. 
	Now we can access values as attributes.
	
	(I thought this way of handling hyperparameters from the LFADS code by David
	Sussillo was useful.)
	"""
	def __getattr__(self, key):
		if key in self:
			return self[key]
		else:
			assert False, ('%s is not a key.' %key)
			
	def __setattr__(self, key, value):
		self[key] = value


def load_data(d):
	"""
	"""
	local_data_root = d.local_data_root
	if options.cluster: cluster_data_root = d.cluster_data_root
	
	this_data_dir = d.datadir
	
	data_folder = cluster_data_root + this_data_dir + '/' if options.cluster else local_data_root + this_data_dir + '/'

	data_file = options.datafile
	data_path = data_folder + data_file
	if options.is_dict:
		data_dict = pickle.load(open(data_path, 'rb+'))
		
		Ytrain = data_dict['Ytrain']
		if 'Yvalid' in data_dict:
			Yvalid = data_dict['Yvalid']
	else:
		Ytrain = pickle.load(open(data_path, 'rb+'))
		vdata_file = options.vdatafile
		test_file = options.test_file
		if vdata_file:
			vdata_path = data_folder + vdata_file
			Yvalid = pickle.load(open(vdata_path, 'rb+'))
		else:
			Yvalid = None
		if test_file:
			testdata_path = data_folder + test_file
			Ytest = pickle.load(open(testdata_path, 'rb+'))
		else:
			Ytest = None
		data_dict = {'Ytrain' : Ytrain, 'Yvalid' : Yvalid, 'Ytest' : Ytest}
	
	print 'Ytrain shape:', Ytrain.shape
	
	return data_dict


def make_rlt_dir(d):
	"""
	Makes the results dir if it is not present and returns its path.
	
	Arguments:
		d:			Parameters object
	Returns:
		rlt_dir:	Path to the directory that will store the results.
	"""
	local_rlt_root = d.local_rlt_root
	local_rlt_dir = local_rlt_root + d.rlt_dir +  cur_date + '/'
	
	cluster_rlt_root = '/vega/stats/users/dh2832/general_libs/v2.0/rslts/'
	cluster_rlt_dir = cluster_rlt_root + d.rlt_dir +  cur_date + '/'
	
	rlt_dir = cluster_rlt_dir if options.cluster else local_rlt_dir
	
	if not os.path.exists(rlt_dir): os.makedirs(rlt_dir)
	
	return rlt_dir


def npdict_to_theanodict(d):
	"""
	Petite function to convert a dictionary of numpy arrays to one of theano shared
	variables initialized to the corresponding array values.
	"""
	newd = {}
	for key in d.keys():
		if isinstance(d[key], np.ndarray):
			newd[key] = theano.shared(value=d[key], name=key, borrow=True)
		else:
			newd[key] = d[key]
	return newd  # comprehension here? {a if P(i) else b for i in somerange}, meh, I'm bored



def init_layers(model_layers, replica_layers):
	"""
	Initializes the parameters of a set of layers using the parameters of an
	identical architecture, presumably already partially trained. 
	
	Args:
		model_layers: 	The set of layers whose parameters we want to replicate.
	Output:
		replica_layers:	The layers whose parameters we want to initialize.
	"""
	for l, m in zip(model_layers, replica_layers):
		m.W.set_value(l.W.get_value())
		m.b.set_value(l.b.get_value())


def define_lat_pars_dict(d):
	"""
	Defines the parameters and neural networks pertaining the Latent Evolution
	Model
	
	Args:
		d:	The object-dictionary
	Output:
		LatParsDict:		A dictionary containing all these hyperparameters and NNs, to be
						fed to the Latent Evolution Model
	"""
	xDim = d.xDim
	nnodesEvlv = d.nnodesMevlv
	
	NNEvolve = InputLayer((None, xDim), name='Ev_IL')
	NNEvolve = DenseLayer(NNEvolve, nnodesEvlv, nonlinearity=softmax, W=Orthogonal(), 
						num_leading_axes=1, name='Ev_HL1')
	NNEvolve = DenseLayer(NNEvolve, xDim**2, nonlinearity=linear, W=Uniform(0.9), 
						num_leading_axes=1, name='Ev_OL')
	
	cmn_dct = dict([('NNEvolve', NNEvolve),
					('alpha', options.alpha)
					])
	LatParsDict = npdict_to_theanodict(cmn_dct)
	
	return LatParsDict


def define_obs_pars_dict(d, MuY_layers_In=None):
	"""
	Defines the parameters and neural networks pertaining the Observation Model.
	
	Args:
		d:	The object-dictionary
	Output:
		ObsParsDict:		A dictionary containing all these hyperparameters and NNs, to be
						fed to the Observation Model
	"""
	act_func_dict = {'softplus' : softplus, 'tanh' : tanh}
	output_nl_dict = {'linear' : linear, 'softplus'  : softplus}
	
	output_nl = output_nl_dict[d.output_nl]
	nl = act_func_dict[d.act_func_hls]
	
	nnodesMrec = d.nnodesMrec
	MuY_dpth = d.muydepth
	
	NNMuY = InputLayer((None, None, d.xDim), name='MuY_IL')
	for i in range(MuY_dpth):
		NNMuY = DenseLayer(NNMuY, nnodesMrec, nonlinearity=nl, W=Orthogonal(), 
						num_leading_axes=2, name='MuY_HL' + str(i))
	NNMuY = DenseLayer(NNMuY, d.yDim, nonlinearity=output_nl, W=Orthogonal(), 
					num_leading_axes=2, name='MuY_OL')
	
	# Initialize layers weight if so desired.
# 	if d.loadfit:
# 		MuY_layers = lasagne.layers.get_all_layers(NNMuY)[1:]
# 		init_layers(MuY_layers_In, MuY_layers)
	
	ObsParsDict = {'NNMuY' : NNMuY, 
				   'NNMuY_W' : d.outWgenscale, 
				   'NNMuY_b' : d.outbgenscale, 
				   'LATCLASS' : LocallyLinearEvolution,
				   'is_out_positive' : d.is_out_positive}
	
	return ObsParsDict


def define_rec_pars_dict(d, MuX_layers_In=None, LX_layers_In=None):
	"""
	Defines the parameters and neural networks pertaining the Recognition Model.
	
	Args:
		d:	The object-dictionary
	Output:
		RecParsDict:		A dictionary containing all these hyperparameters and NNs, to be
						fed to the Recognition Model
	"""
#	 outWrecscale = d.outWrecscale
	mux_dpth = d.mux_depth
	nnodesMrec = d.nnodesMrec
	act_func_dict = {'softplus' : softplus, 'tanh' : tanh}
	nl = act_func_dict[d.act_func_hls]

	NNMuX = InputLayer((None, None, d.yDim))
	for i in range(mux_dpth):
		NNMuX = DenseLayer(NNMuX, nnodesMrec, nonlinearity=nl, W=Normal(0.5), 
						num_leading_axes=2, name='MuX_HL' + str(i))
	NNMuX = DenseLayer(NNMuX, d.xDim, nonlinearity=linear, W=Orthogonal(), 
					num_leading_axes=2, name='MuX_OL')
# 	if d.loadfit:
# 		assert MuX_layers_In is not None, "List of layers not provided."
# 		MuX_layers = lasagne.layers.get_all_layers(NNMuX)[1:]
# 		init_layers(MuX_layers_In, MuX_layers)
	
	
	LambdaX_dpth = d.lbdaxdepth
	NNLambdaX = lasagne.layers.InputLayer((None, None, d.yDim))
	for i in range(LambdaX_dpth):
		NNLambdaX = DenseLayer(NNLambdaX, nnodesMrec, nonlinearity=nl, W=Normal(0.5), 
							num_leading_axes=2, name='LX_HL' + str(i))
	NNLambdaX = DenseLayer(NNLambdaX, d.xDim**2, nonlinearity=linear, W=Orthogonal(), 
						num_leading_axes=2, name='LX_OL')
# 	if options.loadfit:
# 		assert LX_layers_In is not None, "List of layers not provided."
# 		LX_layers = lasagne.layers.get_all_layers(NNLambdaX)[1:]
# 		init_layers(LX_layers_In, LX_layers)
	
	RecParsDict = {'LATCLASS' : LocallyLinearEvolution, 
				   'NNMuX' : NNMuX, 
				   'NNLambdaX' : NNLambdaX}
	
	return RecParsDict


def define_opt_pars_dict(d):
	"""
	Defines optional parameters pertaining to the optimizer.
	
	Args:
		d:	The object-dictionary
	Output:
		OptParsDict:		A dictionary containing all these hyperparameters and NNs, to be
						fed to Optimizer object.
	"""
	return {}


def initialize_optimizer(d, rlt_dir):
	"""
	Initializes the optimizer.
	
	Args:
		d:	The object-dictionary.
		rlt_dir:		The directory to store the results
	Output:
		optzer: The optimizer instance.
	"""
	LatParsDict = define_lat_pars_dict(d)
	ObsParsDict = define_obs_pars_dict(d)
	RecParsDict = define_rec_pars_dict(d)
	OptParsDict = define_opt_pars_dict(d)

	print '\nInitializing SGVB Optimizer....'
	ObsModelDict = {'Gaussian' : GaussianObsTSGM, 'Poisson' : PoissonObsTSGM}
	ParsDicts = {'LatPars' : LatParsDict, 'ObsPars' : ObsParsDict, 
				'RecPars' : RecParsDict, 'OptPars' : OptParsDict}
	optzer = VAEC_NLDSOptimizer(ParsDicts, LocallyLinearEvolution, ObsModelDict[d.obs], 
								d.yDim, d.xDim, SmoothingNLDSTimeSeries, rslt_dir=rlt_dir, 
								common_lat=options.commonlat)
	
	optzer.save_object(rlt_dir + 'sgvb')
	
	return optzer


def train(d, dataset):
	"""
	"""
	rlt_dir = make_rlt_dir(d)
	option_list = [option for option in dir(options) if not option.startswith('_')]
	with open(rlt_dir + 'options.txt', 'wb') as option_file:
		for option in option_list:
			if option not in ['ensure_value', 'read_file', 'read_module']:
				option_file.write(option + ' ' + str(getattr(options, option)) + '\n')
	
	# Initialize optimizer
	optzer = initialize_optimizer(d, rlt_dir)

	# Train
	Ytrain = dataset['Ytrain']
	Yvalid = dataset['Yvalid']
	
	lr = d.learning_rate
	maxeps = d.maxepochs
	costname = d.cost
	method = d.method
	batch_size = d.batch_size
	for _ in range(options.fititers):
		optzer.fit(Ytrain, y_valid=Yvalid, learning_rate=lr, max_epochs=maxeps, 
				   costname=costname, method=method, rslt_dir=rlt_dir, 
				   batch_size=batch_size)
	
	print '  \n Done. As in finished.\n'


def main():
	"""
	Fly baby, fly.
	"""
	d = build_hprms_dict(options)
	d = hprms_dict_to_obj(d)
	
	dataset = load_data(d)
	
	d.train_samps, d.tbins, d.yDim = dataset['Ytrain'].shape
	
	train(d, dataset)


if __name__ == "__main__":
	main()



