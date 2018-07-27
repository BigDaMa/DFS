from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans
from library.cluster_selection.UCB1_ZOMBIE import UCB1
from sets import Set
from sklearn.metrics import accuracy_score
import time
from library.Data.News import News
from library.feature.TFIDF import TFIDF

from library.stat_model.SVC_Model import SVC_Model
from library.stat_model.NaiveBayes import NaiveBayes
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('svmnewsgroup')

ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='ML_Experiments_DB'))


@ex.config
def cfg():
  #classifier
  alpha = 0.01

  #bag of words hyperparameter

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

  #try
  analyzer = 'char_wb'
  lowercase = True
  max_df = 1.0
  min_df = 0.001
  ngrams = 1
  norm = "l2"
  smooth_idf = True
  stop_words = 'english'
  sublinear_tf = False
  use_idf = True
  '''



@ex.automain
#def run(alpha, analyzer, ngrams, lowercase, stop_words, max_df, min_df, norm, use_idf, smooth_idf, sublinear_tf):
def run(analyzer, ngrams, lowercase, stop_words, max_df, min_df, norm, use_idf, smooth_idf, sublinear_tf):
    def train_model(X_train, y_train, stat_model):
        #stat_model.optimize_hyperparameters(X_train, y_train, folds=5)
        # params = stat_model.best_params

        params = {'kernel': 'rbf', 'C': 10, 'probability': True, 'gamma': 0.1}
        #params = {'kernel': 'rbf', 'C': 1, 'probability': True, 'gamma': 0.001} #with scale
        #params = {'alpha': alpha}
        #print params

        stat_model.train(X_train, y_train, params)




    def test_model(stat_model, X_test, y_test, fscore_list):
        y_true, y_pred = y_test, stat_model.predict(X_test)

        fscore_list.append(accuracy_score(y_true, y_pred))
        print(fscore_list[-1])

    # load data
    # data
    data = News()
    #data = Spam()
    data_train_x, data_train_y = data.get_train()
    data_test_x, data_test_y = data.get_test()

    #print newsgroups_train

    featurizer = TFIDF(analyzer, ngrams, lowercase, stop_words, max_df, min_df, norm, use_idf, smooth_idf, sublinear_tf)
    (x, y, X_test, y_test) = featurizer.featurize(data_train_x, data_train_y, data_test_x, data_test_y)
    #incremental training

    stat_model = SVC_Model()
    #stat_model = NaiveBayes()

    X_train = None
    y_train = []

    accuracy_list = [0]

    start_time = time.time()


    #print x.shape[0]

    # first model

    train_model(x, y, stat_model)
    test_model(stat_model, X_test, y_test, accuracy_list)


    #print featurizer.pipeline.get_params()


    print("--- %s seconds ---" % (time.time() - start_time))

    return stat_model.model.score(X_test, y_test)



'''
for alpha_param in [0.01]:#[0.001, 0.01, 0.1, 1.0]:
    for analyzer_param in ['word', 'char', 'char_wb']:
        for lowercase_param in [True, False]:
            for ngrams_param in [1,2]:
                for stop_words_param in [None, 'english']:
                    for max_df_param in [1.0, 0.999, 0.99]:
                        for min_df_param in  [1, 0.001, 0.01]:
                            for norm_param in ['l2']:#['l2','l1', None]:
                                for use_idf_param in [True]:#[True,False]:
                                    for smooth_idf_param in [True]:#[True,False]:
                                        for sublinear_tf_param in [False]:#[True,False]:
                                            ex.run(config_updates={'alpha': alpha_param,
                                                                   'analyzer': analyzer_param,
                                                                   'lowercase': lowercase_param,
                                                                   'ngrams': ngrams_param,
                                                                   'stop_words': stop_words_param,
                                                                   'max_df': max_df_param,
                                                                   'min_df': min_df_param,
                                                                   'norm': norm_param,
                                                                   'use_idf': use_idf_param,
                                                                   'smooth_idf': smooth_idf_param,
                                                                   'sublinear_tf': sublinear_tf_param
                                                                   })
'''