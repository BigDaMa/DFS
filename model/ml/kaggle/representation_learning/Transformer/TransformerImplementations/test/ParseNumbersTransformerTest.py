from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.ParseNumbersTransformer import ParseNumbersTransformer
import pandas as pd

d = ['12m', '12:33p.m.', '12 oz', '12.03.1990']
df = pd.DataFrame(data=d)

print df

t = ParseNumbersTransformer(0)
print t.transform(df, range(len(d)))
