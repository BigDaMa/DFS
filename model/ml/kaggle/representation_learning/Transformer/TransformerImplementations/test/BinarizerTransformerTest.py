from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.BinarizerTransformer import BinarizerTransformer
import pandas as pd

d = [-1, 1, 0, 1, 3, -100]
df = pd.DataFrame(data=d)

print df

t = BinarizerTransformer(0)
print t.transform(df, range(len(d))).shape
print t.transform(df, range(len(d)))