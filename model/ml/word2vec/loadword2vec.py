import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('/home/felix/FastFeatures/embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)

print model.get_vector('Felix')