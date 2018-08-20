from ml.kaggle.representation_learning.Transformer.Mapper.all.AllMapper import AllMapper
from kaggle.representation_learning.Transformer.TransformerImplementations.all.NgramTransformer import NgramTransformer

class NgramMapper(AllMapper):

    def __init__(self, analyzer='word', lowercase=False, ngrams=1, use_tf_idf=True):
        AllMapper.__init__(self, NgramTransformer, {'analyzer': analyzer, 'lowercase': lowercase, 'ngrams': ngrams, 'use_tf_idf': use_tf_idf})
