import numpy as np
import gensim
import re

class AvgWord2VecTransformer():

    def __init__(self, column_id, word2vec_model=None):
        self.column_id = column_id
        self.word2vec_model = word2vec_model
        self.applicable = True

        if word2vec_model == None:
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
                '/home/felix/FastFeatures/embeddings/word2vec/GoogleNews-vectors-negative300.bin.gz', binary=True)


        def word2vec(mystring):
            wordList = re.sub("[^\w]", " ", mystring).split()

            my_vector = np.zeros(self.word2vec_model.vector_size)

            count = 0
            for word in wordList:
                try:
                    my_vector += self.word2vec_model.get_vector(word)
                    count +=1
                except KeyError:
                    pass

            my_vector /= float(count)

            return my_vector

        self.word2vec = np.vectorize(word2vec, otypes=[np.ndarray])



    def fit(self, dataset, ids):
        pass

    def transform(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1

        return np.matrix(np.stack(self.word2vec(column_data)))

    def get_feature_names(self, dataset):
        internal_names = []

        for class_i in range(self.word2vec_model.vector_size):
            internal_names.append(
                str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + "word2vec_" + str(class_i))

        return internal_names

    def get_involved_columns(self):
        return [self.column_id]
