from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.AvgWord2VecTransformer import AvgWord2VecTransformer
import pandas as pd
import numpy as np
import re

d = ['Felix', 'car', 'house', 'This is my house.']
df = pd.DataFrame(data=d)

print df

params = {'column_id': 0, 'word2vec_model': None}

my_class = AvgWord2VecTransformer

t = my_class(params)
res = t.transform(df, range(len(d)))


print res.shape


def testfun(mystring):
    wordList = re.sub("[^\w]", " ", mystring).split()

    my_vector = np.zeros(300)

    count = 0
    for word in wordList:
        try:
            my_vector += np.ones(300) * len(word)
            count += 1
        except KeyError:
            pass

    my_vector /= float(count)

    return my_vector


testfun = np.vectorize(testfun, otypes=[np.ndarray])

mat = df[df.columns[0]].values
res = testfun(mat)

print np.matrix(np.stack(res)).shape