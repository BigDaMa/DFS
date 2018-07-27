from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class TFIDF():

    '''
    #countvectorizer
      analyzer = "word" #'word', 'char', 'char_wb'
      lowercase = False
      ngrams = 1
      stop_words = None # 'english'
      max_df = 1.0 # 0.999, 0.99, 0.9, 0.8
      min_df = 1 # 0.001, 0.01, 0.1, 0.2

      #tfidf
      norm = 'l2' # 'l1', None
      use_idf = True #False
      smooth_idf = True #False
      sublinear_tf = False #True
    '''
    def __init__(self, analyzer="word", ngrams=1, lowercase=False, stop_words=None, max_df=1.0, min_df=1, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):

        self.pipeline = Pipeline([('vect', CountVectorizer(analyzer=analyzer,
                                                      lowercase=lowercase,
                                                      ngram_range=(1, ngrams),
                                                      stop_words=stop_words,
                                                      max_df=max_df,
                                                      min_df=min_df
                                                           )
                                  ),
                             ('tfidf', TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)),
                             # ('scale', StandardScaler(with_mean=False))
                             ])


    def featurize(self, data_train_x, data_train_y, data_test_x, data_test_y):
        self.pipeline.fit(data_train_x)
        x = self.pipeline.transform(data_train_x)
        y = data_train_y

        X_test = self.pipeline.transform(data_test_x)
        y_test = data_test_y

        return (x, y, X_test, y_test)
