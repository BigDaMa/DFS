from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.NgramTransformer import NgramTransformer
import pandas as pd
import numpy as np

#d = ['hallo', 'hallo this is me']
d = [1, 2]
df = pd.DataFrame(data=d)

print df

t = NgramTransformer(0, analyzer='char')
print t.fit(df, range(len(d)))
print t.transform(df, range(len(d)))