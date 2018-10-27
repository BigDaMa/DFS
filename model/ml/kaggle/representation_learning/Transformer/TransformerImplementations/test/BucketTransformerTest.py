from ml.kaggle.representation_learning.Transformer.TransformerImplementations.numeric.BucketTransformer import BucketTransformer
import pandas as pd

d = [1,1,1,1,1,1,1,1,222,222]
df = pd.DataFrame(data=d)

print df

t = BucketTransformer(0)
print t.fit(df, range(len(d)))

print "bins: " + str(t.bins)


print t.transform(df, range(len(d))).shape
print t.transform(df, range(len(d)))