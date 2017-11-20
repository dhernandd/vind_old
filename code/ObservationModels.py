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
import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tnla

import lasagne
from lasagne.nonlinearities import softplus, linear, sigmoid
from lasagne.layers import InputLayer, DenseLayer
from lasagne.init import Orthogonal

from LatEvModels import LocallyLinearEvolution



class PoissonObsTSGM(object):
    """
    """     
    def __init__(self, ObsPars, yDim, xDim, Y=None, 
                 X=None, lat_ev_model=None, LATCLASS=LocallyLinearEvolution):
        """
        """
        self.yDim = yDim
        self.xDim = xDim
        self.Y = Y = T.tensor3('Y') if Y is None else Y
        
        if lat_ev_model is None:
            self.common_lat = False
            self.X = X = T.tensor3('X') if X is None else X
            self.lat_ev_model = lat_ev_model = LATCLASS({}, xDim, X, nnname='GenEv')
        else:
            self.common_lat = True
            self.lat_ev_model = lat_ev_model
            self.X = lat_ev_model.get_X() if X is None else X
                            
        for key in ObsPars.keys():
            setattr(self, key, ObsPars[key])
        
        if not hasattr(self, 'is_out_positive'): self.is_out_positive = True
        if not hasattr(self, 'NNMuY_W'): self.NNMuY_W = 1.0
        if not hasattr(self, 'NNMuY_b'): self.NNMuY_b = 1.0
        if not hasattr(self, 'NNMuY'):
            NNMuY = InputLayer((None, None, self.xDim), name='MuY_IL')
            self.NNMuY = DenseLayer(NNMuY, self.yDim, nonlinearity=softplus, 
                                    W=Orthogonal(), num_leading_axes=2, name='MuY_OL')
            self.NNMuY.W.set_value(self.NNMuY_W*self.NNMuY.W.get_value()) 
            self.NNMuY.b.set_value(self.NNMuY_b*np.ones(self.yDim) + 2.0*(np.random.rand(yDim)))
        inv_tau = 0.2
        self.Rate = ( lasagne.layers.get_output(self.NNMuY, self.X) if self.is_out_positive else 
                     T.exp(inv_tau*lasagne.layers.get_output(self.NNMuY, self.X)) )
    
    
    def get_LatModel(self):
        return self.lat_ev_model
            
    
    def get_Params(self):
        return lasagne.layers.get_all_params(self.NNMuY) if self.common_lat else \
            self.lat_ev_model.get_Params() + lasagne.layers.get_all_params(self.NNMuY)
    
    
    def eval_Rate(self, Xdata):
        """
        """
        return self.Rate.eval({self.X : Xdata})
                       
     
    def sample_XY(self, Nsamps=1, Tbins=30, X0data=None, Xdata=None, withInflow=False):
        """
        TODO: Write docstring
        """
        if Xdata is None:
            Xdata = self.lat_ev_model.sample_X(X0data=X0data, Nsamps=Nsamps, Tbins=Tbins, withInflow=withInflow)
        else:
            Nsamps = Xdata.shape[0]
            Tbins = Xdata.shape[1]
         
        Rate = self.eval_Rate(Xdata)
        Ydata = np.random.poisson(Rate)
        
        return Ydata, Xdata
             
     
    def compute_LogDensity_Yterms(self, Y=None, X=None, padleft=False, persamp=False):
        """
        
        TODO: The persamp option allows this function to return a list of the costs
        computed for each sample. This is useful for implementing more
        sophisticated optimization procedures such as NVIL. TO BE IMPLEMENTED...
         
        NOTE: Please accompany a compute function with an eval function that
        allows evaluation from an external program. compute functions assume by
        default that the 0th dimension of the data arrays is the trial
        dimension. If you deal with a single trial and the trial dimension is
        omitted, set padleft to False to padleft.
        """
        if Y is None: Y = self.Y
        if X is None: X = self.X
         
        if padleft: Y = T.shape_padleft(Y, 1)
        
        Yprime = theano.clone(self.Rate, replace={self.X : X})
        Density = T.sum(Y*T.log(Yprime) - Yprime - T.gammaln(Y+1))
         
        return Density
     
     
    def compute_LogDensity(self, Y=None, X=None, padleft=False, persamp=False):
        """
        NOTE: Every compute function must be accompanied by an eval function
        that allows evaluation from an external program.
        """
        LX = self.lat_ev_model.compute_LogDensity_Xterms(X, padleft=padleft, persamp=persamp)
        LY = self.compute_LogDensity_Yterms(Y, X, padleft=padleft, persamp=persamp)
         
        return LX[0] + LY
             
     
    def eval_LogDensity_Yterms(self, Ydata, Xdata, persamp=False):
        L = self.compute_LogDensity_Yterms(persamp=persamp)
        Nsamps = Ydata.shape[0]
        return L.eval({self.Y : Ydata, self.X : Xdata})/Nsamps
     
     
    def eval_LogDensity(self, Ydata, Xdata, padleft=False, persamp=False):
        Nsamps = Ydata.shape[0]
        return self.compute_LogDensity(padleft=padleft, persamp=persamp).eval({self.Y : Ydata, self.X : Xdata})/Nsamps
    


class GaussianObsTSGM(object):
    """
    A Generative Model for Time Series with Gaussian Observations.
    P(Y|X) ~ Normal(MuY(X), Sigma(X))
    P(X) ~ Normal(0, I)
    Bayes rule holds: P(Y, X) = P(Y|X)P(X)
    One trial here refers to one observation of the set {Y, X}.
     
    NOTE: Obs Generative Model classes model a complete likelihood P(Y, X). They
    typically provide both a sample_XY and a compute_LogDensity method based on
    P(Y, X). The latent model is passed as an attribute, not inherited from.
    """
    def __init__(self, ObsPars, yDim, xDim, Y=None, X=None, 
                 lat_ev_model=None, LATCLASS=LocallyLinearEvolution):
        """
        """
        self.yDim = yDim
        self.xDim = xDim
        self.Y = Y = T.tensor3('Y') if Y is None else Y
        
        if lat_ev_model is None:
            self.common_lat = False
            self.X = X = T.tensor3('X') if X is None else X
            self.lat_ev_model = lat_ev_model = LATCLASS({}, xDim, X, nnname='GenEv')
        else:
            self.common_lat = True
            self.lat_ev_model = lat_ev_model
            self.X = lat_ev_model.get_X() if X is None else X
      
        for key in ObsPars.keys():
            setattr(self, key, ObsPars[key])
         
        gennodes = 60
        if not hasattr(self, 'NNMuY_W'): self.NNMuY_W = 5.0
        if not hasattr(self, 'NNMuY_b'): self.NNMuY_b = 1.0
        if not hasattr(self, 'NNMuY'):
            NNMuY = InputLayer((None, None, self.xDim))
            NNMuY = DenseLayer(NNMuY, 60, nonlinearity=softplus, W=Orthogonal(), 
                        num_leading_axes=2, name='MuY_HL')
            # Gaussian obs seem to work best if we hike the initial scale of W here
            NNMuY.W.set_value(self.NNMuY_W*NNMuY.W.get_value())
            self.NNMuY = DenseLayer(NNMuY, self.yDim, nonlinearity=linear, W=Orthogonal(), 
                                    num_leading_axes=2)  # store last layer as a property to be able to recover all others
        self.NNMuY.W.set_value(self.NNMuY_W*self.NNMuY.W.get_value())
        self.NNMuY.b.set_value(self.NNMuY_b*np.ones(yDim) + 1.0*(np.random.rand(yDim) - 0.5))
        self.MuY = lasagne.layers.get_output(self.NNMuY, self.X)
        
        if not hasattr(self, 'NLObsNoise'): self.NLObsNoise = False 
        if not hasattr(self, 'SigmaInvChol'):
            if not self.NLObsNoise:
                self.SigmaInvChol = theano.shared(1.0*np.eye(self.yDim), 'SigmaInvChol')
            else:   # TODO: This is not right, this is not a Cholesky decomposition. Fix.
                assert False, 'X-dependent generative variance not implemented!'
                NN_SigmaInvChol = lasagne.layers.InputLayer((None, self.xDim))
                NN_SigmaInvChol = lasagne.layers.DenseLayer(NN_SigmaInvChol, gennodes, nonlinearity=sigmoid)
                self.NN_SigmaInvChol = lasagne.layers.DenseLayer(NN_SigmaInvChol, self.yDim**2, nonlinearity=linear)
                self.SigmaInvChol = lasagne.layers.get_output(self.NN_SigmaInvChol, self.X)
        self.SigmaChol = Tnla.matrix_inverse(self.SigmaInvChol)
        self.SigmaInv = T.dot(self.SigmaInvChol, self.SigmaInvChol.T)
        self.Sigma = T.dot(self.SigmaChol, self.SigmaChol.T)
 
    # TODO: Implement this as a property, ugh
    def get_LatModel(self):
        return self.latm
     
     
    def get_Params(self):
        params_MuY = lasagne.layers.get_all_params(self.NNMuY) 
        if not self.NLObsNoise:
            return  params_MuY + [self.SigmaInvChol] if self.common_lat else \
                self.lat_ev_model.get_Params() + params_MuY
        else:
            assert False, 'Not implemented at the moment. Please set self.NLObsNoise to False.'

     
    def eval_Mu(self, Xdata):
        """
        """
        return self.MuY.eval({self.X : Xdata})
          
      
    def eval_SigmaInvChol(self):
        """
        """
        return self.SigmaInvChol.eval()
           
      
    def sample_XY(self, X0data=None, Nsamps=1, Tbins=30, Xdata=None, withInflow=False):
        """
        TODO: Write docstring
        """
        if Xdata is None:
            Xdata = self.lat_ev_model.sample_X(X0data=X0data, 
                                               Nsamps=Nsamps, 
                                               Tbins=Tbins, 
                                               withInflow=withInflow)
        else:
            Nsamps = Xdata.shape[0]
            Tbins = Xdata.shape[1]
            
        SigmaChol = T.tile(self.SigmaChol, (Nsamps*Tbins, 1, 1))
        SigmaCholN = T.batched_dot(np.random.randn(Nsamps*Tbins, self.yDim), SigmaChol)
         
        Musymb = theano.clone(self.MuY, replace={self.X : Xdata})  
        Musymb = T.reshape(Musymb, (Nsamps*Tbins, self.yDim))  
         
        Ysymb = SigmaCholN + Musymb
        Ysymb = T.reshape(Ysymb, (Nsamps, Tbins, self.yDim))
          
        Ydata = Ysymb.eval()
        return Ydata, Xdata
      
      
    def compute_LogDensity_Yterms(self, Y=None, X=None, padleft=False, persamp=False):
        """
        TODO: Write docstring
        
        The persamp option allows this function to return a list of the
        costs computed for each sample. This is useful for implementing more
        sophisticated optimization procedures such as NVIL.
          
        NOTE: Please accompany every compute function with an eval function
        that allows evaluation from an external program. 
        
        compute functions assume by default that the 0th dimension of the data
        arrays is the trial dimension. If you deal with a single trial and the
        trial dimension is omitted, set padleft to False to padleft.
        """
        if Y is None:
            Y = self.Y
        if X is None:
            X = self.X
          
        if padleft:
            Y = T.shape_padleft(Y, 1)
          
        Nsamps = Y.shape[0]
        Tbins = Y.shape[1]
          
        Mu = theano.clone(self.MuY, replace={self.X : X})
        DeltaY = Y - Mu
        
        # TODO: Implement SigmaInv dependent on X
        if persamp:
            L1 = -0.5*T.sum(DeltaY*T.dot(DeltaY, self.SigmaInv), axis=(1,2))
            L2 = 0.5*T.log(Tnla.det(self.SigmaInv))*Tbins
        else:
            L1 = -0.5*T.sum(DeltaY*T.dot(DeltaY, self.SigmaInv))
            L2 = 0.5*T.log(Tnla.det(self.SigmaInv))*Nsamps*Tbins
        L = L1 + L2
          
        return L, L1, L2
          
          
    def compute_LogDensity(self, Y=None, X=None, padleft=False, persamp=False):
        """
        TODO: Write docstring
  
        NOTE: Please accompany every compute function with an eval function
        that allows evaluation from an external program. 
        
        compute functions assume by default that the 0th dimension of the data
        arrays is the trial dimension. If you deal with a single trial and the
        trial dimension is omitted, set padleft to False to padleft.
        """
        LX = self.lat_ev_model.compute_LogDensity_Xterms(X, padleft=padleft, persamp=persamp)
        LY = self.compute_LogDensity_Yterms(Y, X, padleft=padleft, persamp=persamp)
          
        return LX[0] + LY[0]
                  
          
    def eval_LogDensity_Yterms(self, Ydata, Xdata, persamp=False):
        L, L1, L2 = self.compute_LogDensity_Yterms(persamp=persamp)
        Nsamps = Ydata.shape[0]
        return L.eval({self.Y : Ydata, self.X : Xdata})/Nsamps, L1.eval({self.Y : Ydata, self.X : Xdata})/Nsamps, L2.eval({self.Y : Ydata})/Nsamps
          
          
    def eval_LogDensity(self, Ydata, Xdata, padleft=False, persamp=False):
        Nsamps = Ydata.shape[0]
        return self.compute_LogDensity(padleft=padleft, persamp=persamp).eval({self.Y : Ydata, self.X : Xdata})/Nsamps
         
