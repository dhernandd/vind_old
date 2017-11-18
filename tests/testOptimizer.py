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
import unittest
import os

import numpy as np

from LatEvModels import LocallyLinearEvolution
from ObservationModels import PoissonObsTSGM
from RecognitionModels import SmoothingNLDSTimeSeries
from OptimizerVAEC_TS import Optimizer_TS, VAEC_NLDSOptimizer

from datetools import addDateTime
rslt_dir = '/Users/danielhernandez/Work/time_series/vae_nlds_rec_algo_v2/rslts/test_fit' + addDateTime() + '/'
if not os.path.exists(rslt_dir):
    os.makedirs(rslt_dir)


class OptimizerTest(unittest.TestCase):
    """
    """
    xDim = 2
    yDim = 10
    latpars, optpars = {}, {}
    obspars = {'LATCLASS' : LocallyLinearEvolution}
    recpars = {'LATCLASS' : LocallyLinearEvolution}
    ParsDicts = {'LatPars' : latpars, 'OptPars' : optpars, 'ObsPars' : obspars, 'RecPars' : recpars}
    xdata1 = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 15.0], [50.0, 50.0]])
    Xdata1 = np.array([[[0.0, 0.0], [1.0, 1.0], [50.0, 50.0], [10.0, 15.0]]])
    
    lm = LocallyLinearEvolution(optpars, xDim)
    obsm2 = PoissonObsTSGM(obspars, yDim, xDim, lat_ev_model=lm)


    opt1 = VAEC_NLDSOptimizer(ParsDicts, LocallyLinearEvolution, PoissonObsTSGM, yDim, xDim, SmoothingNLDSTimeSeries)
    opt2 = VAEC_NLDSOptimizer(ParsDicts, LocallyLinearEvolution, PoissonObsTSGM, yDim, xDim, SmoothingNLDSTimeSeries, common_lat=True)
    
    def test_simple(self):
        mgen1 = self.opt1.get_GenModel()
        mgen2 = self.opt2.get_GenModel()

        Ydata, Xdata = mgen1.sample_XY(Nsamps=20, Tbins=25)
        print 'Some Ydata:\n', Ydata[:2]
        print 'Some Xdata:\n', Xdata[:2]
    
        Ydata, Xdata = mgen2.sample_XY(Nsamps=20, Tbins=25)
        print 'Some Ydata:\n', Ydata[:2]
        print 'Some Xdata:\n', Xdata[:2]

    def test_cost_LogDensity(self):
        mgen1 = self.opt1.get_GenModel()
        mgen2 = self.opt2.get_GenModel()
        
        Ydata1, Xdata1 = mgen1.sample_XY(Nsamps=20, Tbins=25)
        Ydata2, Xdata2 = mgen2.sample_XY(Nsamps=20, Tbins=25)
        
        cn = 'LogDensity'
        print 'Cost, Data 1:', self.opt1.eval_cost(Ydata1, Xdata1, costname=cn)
        print 'Cost, Data 2:', self.opt2.eval_cost(Ydata2, Xdata2, costname=cn)
        print 'Cost, Wrong Data:', self.opt2.eval_cost(Ydata1, Xdata1, costname=cn)
    
    def test_cost_ELBO(self):
        mgen = self.opt1.get_GenModel()
        Ydata, Xdata = mgen.sample_XY(Nsamps=20, Tbins=30)
        cn = 'ELBO'
        print 'ELBO:', self.opt1.eval_cost(Ydata, Xdata, costname=cn)

    def test_fit_ELBO_Poisson_wCA(self):
        Nsamps = 100
        Ntrain = 4*Nsamps/5
        Ydata, Xdata = self.obsm2.sample_XY(Nsamps=Nsamps, Tbins=30, withInflow=True)
        print 'Ydata[0,0]:', Ydata[0,0]

        mrec = self.opt2.get_RecModel()        
        print 'Actual LogDensity:', self.obsm2.eval_LogDensity(Ydata, Xdata)
        Ytrain = Ydata[:Ntrain]
        Yvalid = Ydata[Ntrain:]
        
        print 'Ydata mean, Xdata mean:', Ydata.mean(), Xdata.mean()
        print 'Inferred X mean at epoch 0:', mrec.eval_Mu(Ydata).mean()
        self.opt2.fit(Ytrain, y_valid=None, learning_rate=3e-3, max_epochs=150, costname='ELBO', rslt_dir=rslt_dir)

    def test_fit_ELBO_Poisson(self):
        Nsamps = 100
        Ntrain = 4*Nsamps/5
        Ydata, Xdata = self.obsm2.sample_XY(Nsamps=Nsamps, Tbins=30, withInflow=True)
        print 'Ydata[0,0]:', Ydata[0,0]
        print 'Actual LogDensity:', self.obsm2.eval_LogDensity(Ydata, Xdata)

        Ytrain = Ydata[:Ntrain]
        Yvalid = Ydata[Ntrain:]
        
        mrec = self.opt1.get_RecModel()                
        print 'Ydata mean, Xdata mean:', Ydata.mean(), Xdata.mean()
        print 'Inferred X mean at epoch 0:', mrec.eval_Mu(Ydata).mean()
        self.opt1.fit(Ytrain, y_valid=None, learning_rate=3e-3, max_epochs=150, costname='ELBO', rslt_dir=rslt_dir)



if __name__ == '__main__':
    suiteO = unittest.TestSuite()
#     suiteO.addTest(OptimizerTest("test_simple"))                        # OK!
#     suiteO.addTest(OptimizerTest("test_cost_LogDensity"))               # OK!
#     suiteO.addTest(OptimizerTest("test_cost_ELBO"))                     # OK!
    suiteO.addTest(OptimizerTest("test_fit_ELBO_Poisson_wCA"))       # OK!
#     suiteO.addTest(OptimizerTest("test_fit_ELBO_Poisson"))
    
    runner = unittest.TextTestRunner()
    
    runner.run(suiteO)
