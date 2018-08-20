from ml.kaggle.representation_learning.Transformer.Mapper.all.AllMapper import AllMapper
from kaggle.representation_learning.Transformer.TransformerImplementations.all.AvgWord2VecTransformer import AvgWord2VecTransformer
import gensim

class AvgWord2VecMapper(AllMapper):

    def __init__(self, word2vec_model=None):
        if word2vec_model == None:
            word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
                '/home/felix/FastFeatures/embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

        AllMapper.__init__(self, AvgWord2VecTransformer, {'word2vec_model': word2vec_model})
