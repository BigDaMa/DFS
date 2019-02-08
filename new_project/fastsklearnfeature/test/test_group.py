import numpy as np
from fastsklearnfeature.reader.Reader import Reader
from fastsklearnfeature.splitting.Splitter import Splitter
from fastsklearnfeature.configuration.Config import Config
from fastsklearnfeature.transformations.GroupByThenTransformation import GroupByThenTransformation
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature



f0 = RawFeature('col0', 0, {})
f1 = RawFeature('col1', 1, {})



training = np.array([[6, 1], [5,1], [4,2], [3,2]])


print(training[0,1])

print(training.shape)



c = CandidateFeature(GroupByThenTransformation(np.sum, 2), [f0, f1])

c.fit(training)


print(c.transform(training))

'''
raw_features[1].fit(training)
print(raw_features[1].transform(training))

raw_features[0].fit(training)
print(raw_features[0].transform(training))
'''
