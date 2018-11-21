from fastfeature.transformations.NumericUnaryTransformation import NumericUnaryTransformation
from fastfeature.candidates.CandidateFeature import CandidateFeature
from typing import List
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class NanMinMaxScalingTransformation(NumericUnaryTransformation):
    def __init__(self):
        name = 'NanMinMaxScaling'
        NumericUnaryTransformation.__init__(self, name)

    def fit(self, data):
        self.minmaxscaler = MinMaxScaler()
        d = pd.DataFrame({'A': np.matrix(data).A1})
        null_index = d['A'].isnull()
        self.minmaxscaler.fit(d.loc[~null_index, ['A']])



    def transform(self, data):
        d = pd.DataFrame({'A': np.matrix(data).A1})
        null_index = d['A'].isnull()
        d.loc[~null_index, ['A']] = self.minmaxscaler.transform(d.loc[~null_index, ['A']])
        return d['A'].values
