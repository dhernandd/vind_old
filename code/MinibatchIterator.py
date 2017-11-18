"""
The MIT License (MIT)
Copyright (c) 2016 Evan Archer & Daniel Hernandez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""    

import numpy as np


class DatasetTrialIterator(object):
    """
    """
    def __init__(self, Ydata, Xdata=None, batch_size=1, trialD=True):
        self.Y = Ydata
        self.X = Xdata
        self.batch_size = batch_size
        self.trialD = trialD

        
    def __iter__(self):
        """
        """
        shape_padleft = lambda Z : np.reshape(Z, (1,) + Z.shape)    # Padleft the shape. This is used when batch_size=1  and the trial dimension is not included
                                                                    # to create a matrix instead of a vector. 
        y_len = len(self.Y)
        if self.batch_size == 1:
            if self.X is None:
                for i in range(y_len):
                    yield shape_padleft(self.Y[i]) if self.trialD else self.Y[i]
            else:
                for i in range(y_len):
                    yield (shape_padleft(self.Y[i]), shape_padleft(self.X[i])) if self.trialD else (self.Y[i], self.X[i])
        else:
#             rem_indices = np.random.permutation(y_len)
            rem_indices = range(y_len)
            if self.X is None:
                while y_len > self.batch_size:
                    inds = rem_indices[:self.batch_size]
                    rem_indices = rem_indices[self.batch_size:]
                    y_len = len(rem_indices)
                    yield self.Y[inds]
                yield shape_padleft(self.Y[inds]) if y_len == 1 else self.Y[inds]
            else:
                while y_len > self.batch_size:
                    inds = rem_indices[:self.batch_size]
                    rem_indices = rem_indices[self.batch_size:]
                    y_len = len(rem_indices)
                    yield self.Y[inds], self.X[inds]
#                 yield (shape_padleft(self.Y[inds]), shape_padleft(self.X[inds])) if y_len == 1 else 
                yield (self.Y[inds], self.X[inds])
                