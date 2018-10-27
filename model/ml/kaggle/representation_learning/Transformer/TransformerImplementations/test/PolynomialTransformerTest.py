from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.PolynomialTransformer import PolynomialTransformer
import pandas as pd
import numpy as np

d = [-1, 1, np.NaN, 1, 3, -100]
df = pd.DataFrame(data=d)

print df

t = PolynomialTransformer(0)
print t.fit(df, range(len(d)))
print t.transform(df, range(len(d)))
print t.transform(df, range(len(d)))