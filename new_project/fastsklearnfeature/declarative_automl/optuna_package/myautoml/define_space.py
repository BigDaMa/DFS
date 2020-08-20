from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.MyIdentity import IdentityTransformation

from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.categorical_encoding.FrequencyEncodingOptuna import FrequencyEncodingOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.categorical_encoding.OneHotEncoderOptuna import OneHotEncoderOptuna

from fastsklearnfeature.declarative_automl.optuna_package.classifiers.RandomForestClassifierOptuna import RandomForestClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.AdaBoostClassifierOptuna import AdaBoostClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.BernoulliNBOptuna import BernoulliNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.DecisionTreeClassifierOptuna import DecisionTreeClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.ExtraTreesClassifierOptuna import ExtraTreesClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.GaussianNBOptuna import GaussianNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.HistGradientBoostingClassifierOptuna import HistGradientBoostingClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.KNeighborsClassifierOptuna import KNeighborsClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.LinearSVCOptuna import LinearSVCOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.MultinomialNBOptuna import MultinomialNBOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.PassiveAggressiveOptuna import PassiveAggressiveOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.QuadraticDiscriminantAnalysisOptuna import QuadraticDiscriminantAnalysisOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.SGDClassifierOptuna import SGDClassifierOptuna
from fastsklearnfeature.declarative_automl.optuna_package.classifiers.SVCOptuna import SVCOptuna

from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.scaling.MinMaxScalerOptuna import MinMaxScalerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.scaling.NormalizerOptuna import NormalizerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.scaling.QuantileTransformerOptuna import QuantileTransformerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.scaling.RobustScalerOptuna import RobustScalerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.data_preprocessing.scaling.StandardScalerOptuna import StandardScalerOptuna

from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.FastICAOptuna import FastICAOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.FeatureAgglomerationOptuna import FeatureAgglomerationOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.KernelPCAOptuna import KernelPCAOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.NystroemOptuna import NystroemOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.PCAOptuna import PCAOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.PolynomialFeaturesOptuna import PolynomialFeaturesOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.RandomTreesEmbeddingOptuna import RandomTreesEmbeddingOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.RBFSamplerOptuna import RBFSamplerOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.SelectPercentileOptuna import SelectPercentileOptuna
from fastsklearnfeature.declarative_automl.optuna_package.feature_preprocessing.TruncatedSVDOptuna import TruncatedSVDOptuna

classifier_list = [RandomForestClassifierOptuna(),
                                AdaBoostClassifierOptuna(),
                                BernoulliNBOptuna(),
                                DecisionTreeClassifierOptuna(),
                                ExtraTreesClassifierOptuna(),
                                GaussianNBOptuna(),
                                HistGradientBoostingClassifierOptuna(),
                                KNeighborsClassifierOptuna(),
                                LinearSVCOptuna(),
                                MultinomialNBOptuna(),
                                PassiveAggressiveOptuna(),
                                QuadraticDiscriminantAnalysisOptuna(),
                                SGDClassifierOptuna(),
                                SVCOptuna()
                                ]
preprocessor_list = [IdentityTransformation(),
                                  FastICAOptuna(),
                                  FeatureAgglomerationOptuna(),
                                  KernelPCAOptuna(),
                                  NystroemOptuna(),
                                  PCAOptuna(),
                                  PolynomialFeaturesOptuna(),
                                  RandomTreesEmbeddingOptuna(),
                                  RBFSamplerOptuna(),
                                  SelectPercentileOptuna(),
                                  TruncatedSVDOptuna()]
'''
scaling_list = [IdentityTransformation(),
                             MinMaxScalerOptuna(),
                             NormalizerOptuna(),
                             QuantileTransformerOptuna(),
                             RobustScalerOptuna(),
                             StandardScalerOptuna()]
'''

scaling_list = [QuantileTransformerOptuna()]

categorical_encoding_list = [OneHotEncoderOptuna(),
                                          FrequencyEncodingOptuna()]