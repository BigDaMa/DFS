import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import operator
from ml.kaggle.representation_learning.Transformer.TransformerImplementations.all.AllTransformer import AllTransformer

class NgramTransformer(AllTransformer):

    def __init__(self, column_id, analyzer='word', lowercase=False, ngrams=1, use_tf_idf=True):
        AllTransformer.__init__(self, column_id, "ngram")
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.ngrams = ngrams
        self.use_tf_idf = use_tf_idf

        if use_tf_idf:
            self.pipeline = Pipeline([('vect', CountVectorizer(analyzer=analyzer, lowercase=lowercase, ngram_range=(1, ngrams), stop_words=None)),
                                 ('tfidf', TfidfTransformer())
                                 ])
        else:
            self.pipeline = Pipeline(
                [('vect', CountVectorizer(analyzer=analyzer, lowercase=lowercase, ngram_range=(1, ngrams)))])


    def fit(self, dataset, ids):
        column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
        try:
            self.pipeline.fit(column_data)
            self.output_space_size = len(self.pipeline.named_steps['vect'].vocabulary_)

        except ValueError:
            self.applicable = False
            self.output_space_size = 0


    def transform(self, dataset, ids):
        if self.applicable:
            column_data = np.matrix(dataset.values[ids, self.column_id], dtype='str').A1
            return self.pipeline.transform(column_data)
        else:
            return None

    def get_feature_names(self, dataset):
        listed_tuples = sorted(self.pipeline.named_steps['vect'].vocabulary_.items(), key=operator.itemgetter(1))
        feature_names = [str(self.column_id) + '#' + str(dataset.columns[self.column_id]) + "#" + str(self.analyzer) + "_ngram_"+ str(self.ngrams) + "#" + tuple_dict_sorted[0] + "#" for tuple_dict_sorted in
             listed_tuples]

        return feature_names

    def __str__(self):
        return self.__class__.__name__ + "_analyzer_" + str(self.analyzer) + "_ngrams_" + str(self.ngrams) + "_use_tfidf_" + str(self.use_tf_idf) + "_lowercase_" + str(self.lowercase)
