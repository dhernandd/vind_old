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

import numpy as np
import sys
sys.path.append('../../')

import vind

__package__ = 'vind.tests'
from ..code.LatEvModels import LocallyLinearEvolution
from ..code.ObservationModels import PoissonObsTSGM, GaussianObsTSGM



class PoissonObsTest(unittest.TestCase):
    """
    """
    optpars = {}
    xDim = 2
    xdata1 = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 15.0], [50.0, 50.0]])
    Xdata1 = np.array([[[0.0, 0.0], [1.0, 1.0], [50.0, 50.0], [10.0, 15.0]]])

    lm = LocallyLinearEvolution(optpars, xDim)
    Xdata2 = lm.sample_X(5, 10, withInflow=True)


    yDim = 10
    obspars = {}
    obsm1 = PoissonObsTSGM(obspars, yDim, xDim)
    obsm2 = PoissonObsTSGM(obspars, yDim, xDim, lat_ev_model=lm)
    Ydata3, Xdata3 = obsm1.sample_XY(Nsamps=5, Tbins=10, withInflow=True)
    Ydata4, Xdata4 = obsm2.sample_XY(Nsamps=5, Tbins=10, withInflow=True)
    
    def test_simple(self):
        print 'Rate (obsm1):', self.obsm1.eval_Rate(self.Xdata1)
        print 'Rate (obsm2):', self.obsm2.eval_Rate(self.Xdata2)
        
    def test_sample_XY(self):
        Ydata, _ = self.obsm1.sample_XY(Nsamps=5, Tbins=7, withInflow=False)
        print 'Ydata (obsm1):', Ydata
        
    def test_eval_LogDensity_Yterms(self):
        print 'LogDensity (Yterms):', self.obsm1.eval_LogDensity_Yterms(self.Ydata3, self.Xdata3)
        print 'LogDensity (Yterms, wrong Xdata):', self.obsm1.eval_LogDensity_Yterms(self.Ydata3, self.Xdata2)
        print 'If using the wrong Xdata, abs(LogDensity) should be much higher.'
        
    def test_eval_LogDensity(self):
        print 'LogDensity:', self.obsm1.eval_LogDensity(self.Ydata3, self.Xdata3)
        print 'LogDensity (wrong latent DS):', self.obsm1.eval_LogDensity(self.Ydata4, self.Xdata4)
        print 'LogDensity (with Latent Model defined outside):', self.obsm2.eval_LogDensity(self.Ydata4, self.Xdata4)


class GaussianObsTest(unittest.TestCase):
    """
    """
    optpars = {}
    xDim = 2
    Xdata1 = np.random.rand(1, 5, xDim)
    
    lm = LocallyLinearEvolution(optpars, xDim)
    Xdata2 = lm.sample_X(5, 10, withInflow=True)

    yDim = 10
    obspars = {}
    obsm1 = GaussianObsTSGM(obspars, yDim, xDim)
    obsm2 = GaussianObsTSGM(obspars, yDim, xDim, lat_ev_model=lm)

    Ydata3, Xdata3 = obsm1.sample_XY(Nsamps=5, Tbins=10, withInflow=True)
    Ydata4, Xdata4 = obsm2.sample_XY(Nsamps=5, Tbins=10, withInflow=True)

    def test_simple(self):
        print 'Mu (obsm1):', self.obsm1.eval_Mu(self.Xdata1)
        print 'Mu (obsm2):', self.obsm2.eval_Mu(self.Xdata2)
        
    def test_sample_XY(self):
        Ydata, _ = self.obsm1.sample_XY(Nsamps=5, Tbins=7, withInflow=True)
        print 'Ydata (obsm1):', Ydata

    def test_eval_LogDensity_Yterms(self):
        print 'LogDensity (Yterms):', self.obsm1.eval_LogDensity_Yterms(self.Ydata3, self.Xdata3)
        print 'LogDensity (Yterms, wrong Xdata):', self.obsm1.eval_LogDensity_Yterms(self.Ydata3, self.Xdata2)
        print 'If using the wrong Xdata, abs(LogDensity) should be much higher.'

    def test_eval_LogDensity(self):
        print 'LogDensity:', self.obsm1.eval_LogDensity(self.Ydata3, self.Xdata3)
        print 'LogDensity (wrong latent DS):', self.obsm1.eval_LogDensity(self.Ydata4, self.Xdata4)
        print 'LogDensity (with Latent Model defined outside):', self.obsm2.eval_LogDensity(self.Ydata4, self.Xdata4)



if __name__ == '__main__':
    suitePO = unittest.TestSuite()
    suitePO.addTest(PoissonObsTest("test_simple"))                          # OK!
    suitePO.addTest(PoissonObsTest("test_sample_XY"))                       # OK!
    suitePO.addTest(PoissonObsTest("test_eval_LogDensity_Yterms"))          # OK!
    suitePO.addTest(PoissonObsTest("test_eval_LogDensity"))                 #  OK!

    suiteGO = unittest.TestSuite()
    suiteGO.addTest(GaussianObsTest("test_simple"))
    suiteGO.addTest(GaussianObsTest("test_sample_XY"))
    suiteGO.addTest(GaussianObsTest("test_eval_LogDensity_Yterms"))
    suiteGO.addTest(GaussianObsTest("test_eval_LogDensity"))
    
    
    runner = unittest.TextTestRunner()
    runner.run(suitePO)
    runner.run(suiteGO)