import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import operator

class NgramTransformer():

    def __init__(self, column_id, analyzer='word', lowercase=False, ngrams=1, use_tf_idf=True):
        self.column_id = column_id
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.ngrams= ngrams
        self.use_tf_idf =use_tf_idf
        self.applicable = True

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
        except ValueError:
            self.applicable = False

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

    def get_involved_columns(self):
        return [self.column_id]
