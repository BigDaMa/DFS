import autograd.numpy as anp
import numpy as np
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
import numpy as np
import pickle
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from fastsklearnfeature.candidates.RawFeature import RawFeature
from fastsklearnfeature.interactiveAutoML.feature_selection.ForwardSequentialSelection import ForwardSequentialSelection
from fastsklearnfeature.transformations.OneHotTransformation import OneHotTransformation
from typing import List, Dict, Set
from fastsklearnfeature.interactiveAutoML.CreditWrapper import run_pipeline
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.algorithms.nsga2 import NSGA2
import matplotlib.pyplot as plt
from fastsklearnfeature.interactiveAutoML.Runner import Runner
import copy
from sklearn.linear_model import LogisticRegression
from fastsklearnfeature.candidates.CandidateFeature import CandidateFeature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from fastsklearnfeature.transformations.IdentityTransformation import IdentityTransformation
from fastsklearnfeature.transformations.MinMaxScalingTransformation import MinMaxScalingTransformation
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time

from art.metrics import RobustnessVerificationTreeModelsCliqueMethod
from art.metrics import loss_sensitivity
from art.metrics import empirical_robustness


from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from art.classifiers import XGBoostClassifier, LightGBMClassifier, SklearnClassifier


from fastsklearnfeature.interactiveAutoML.feature_selection.RunAllKBestSelection import RunAllKBestSelection
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import fcbf
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import variance
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import model_score
from fastsklearnfeature.interactiveAutoML.feature_selection.BackwardSelection import BackwardSelection
from sklearn.model_selection import train_test_split

from fastsklearnfeature.interactiveAutoML.new_bench import my_global_utils1

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier

from fastsklearnfeature.configuration.Config import Config

from skrebate import ReliefF
from fastsklearnfeature.interactiveAutoML.feature_selection.fcbf_package import my_fisher_score


sensitive_attribute = "sex"

n_estimators = 5

df = pd.read_csv(Config.get('data_path') + '/adult/dataset_183_adult.csv', delimiter=',', header=0)
y = df['class']
del df['class']
X = df
one_hot = True

limit = 1000

X_train, X_test, y_train, y_test = train_test_split(X.values[0:limit,:], y.values[0:limit], test_size=0.5, random_state=42)
xshape = X_train.shape[1]
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train = encoder.fit_transform(X_train)
xshape = X_train.shape[1]
print(xshape)
X_test = encoder.transform(X_test)

print(encoder.get_feature_names(input_features=X.columns))

import os

import tensorflow as tf
import numpy as np

from cleverhans.attacks import FastGradientMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import model_eval

#FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
TRAIN_DIR = '/home/felix/phd/test_hans'
FILENAME = 'mnist.ckpt'
LOAD_MODEL = False


train_start=0
train_end=60000
test_start=0
test_end=10000
nb_epochs=NB_EPOCHS
batch_size=BATCH_SIZE
learning_rate=LEARNING_RATE
train_dir=TRAIN_DIR,
filename=FILENAME
load_model=LOAD_MODEL
testing=False
label_smoothing=0.1

tf.keras.backend.set_learning_phase(0)

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

if tf.keras.backend.image_data_format() != 'channels_last':
    raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

# Create TF session and set as Keras backend session
sess = tf.Session()
tf.keras.backend.set_session(sess)

# Get MNIST test data
mnist = MNIST(train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end)

nb_classes = 2

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(X_train.shape[0], X_train.shape[1]))
y = tf.placeholder(tf.float32, shape=(nb_classes))

model = tf.keras.models.Sequential()

layers = [tf.keras.layers.Dense(128),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(nb_classes)]

for layer in layers:
    model.add(layer)

model.add(tf.keras.layers.Activation('softmax'))

preds = model(x)


def evaluate():
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, X_test, y_test, args=eval_params)
    report.clean_train_clean_eval = acc
#        assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate examples: %0.4f' % acc)

# Train an MNIST model
train_params = {
  'nb_epochs': nb_epochs,
  'batch_size': batch_size,
  'learning_rate': learning_rate,
  'train_dir': train_dir,
  'filename': filename
}

rng = np.random.RandomState([2017, 8, 30])
#if not os.path.exists('/home/felix/phd/test_hans'):
#    os.mkdir(train_dir)

ckpt = None#tf.train.get_checkpoint_state(train_dir)
print(train_dir, ckpt)
ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
wrap = KerasModelWrapper(model)

if load_model and ckpt_path:
    saver = tf.train.Saver()
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    print("Model loaded from: {}".format(ckpt_path))
    evaluate()
else:
    print("Model was not loaded, training from scratch.")
    loss = CrossEntropy(wrap, smoothing=label_smoothing)
    train(sess, loss, X_train, y_train, evaluate=evaluate,
          args=train_params, rng=rng)





'''
  # Define TF model graph
  model = cnn_model(img_rows=img_rows, img_cols=img_cols,
                    channels=nchannels, nb_filters=64,
                    nb_classes=nb_classes)
  preds = model(x)
  print("Defined TensorFlow model graph.")

  def evaluate():
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
    report.clean_train_clean_eval = acc
#        assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate examples: %0.4f' % acc)

  # Train an MNIST model
  train_params = {
      'nb_epochs': nb_epochs,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'train_dir': train_dir,
      'filename': filename
  }

  rng = np.random.RandomState([2017, 8, 30])
  if not os.path.exists(train_dir):
    os.mkdir(train_dir)

  ckpt = tf.train.get_checkpoint_state(train_dir)
  print(train_dir, ckpt)
  ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
  wrap = KerasModelWrapper(model)

  if load_model and ckpt_path:
    saver = tf.train.Saver()
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    print("Model loaded from: {}".format(ckpt_path))
    evaluate()
  else:
    print("Model was not loaded, training from scratch.")
    loss = CrossEntropy(wrap, smoothing=label_smoothing)
    train(sess, loss, x_train, y_train, evaluate=evaluate,
          args=train_params, rng=rng)

  # Calculate training error
  if testing:
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, x_train, y_train, args=eval_params)
    report.train_clean_train_clean_eval = acc

  # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
  fgsm = FastGradientMethod(wrap, sess=sess)
  fgsm_params = {'eps': 0.3,
                 'clip_min': 0.,
                 'clip_max': 1.}
  adv_x = fgsm.generate(x, **fgsm_params)
  # Consider the attack to be constant
  adv_x = tf.stop_gradient(adv_x)
  preds_adv = model(adv_x)

  # Evaluate the accuracy of the MNIST model on adversarial examples
  eval_par = {'batch_size': batch_size}
  acc = model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_par)
  print('Test accuracy on adversarial examples: %0.4f\n' % acc)
  report.clean_train_adv_eval = acc

  # Calculating train error
  if testing:
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, x_train,
                     y_train, args=eval_par)
    report.train_clean_train_adv_eval = acc

  print("Repeating the process, using adversarial training")
  # Redefine TF model graph
  model_2 = cnn_model(img_rows=img_rows, img_cols=img_cols,
                      channels=nchannels, nb_filters=64,
                      nb_classes=nb_classes)
  wrap_2 = KerasModelWrapper(model_2)
  preds_2 = model_2(x)
  fgsm2 = FastGradientMethod(wrap_2, sess=sess)

  def attack(x):
    return fgsm2.generate(x, **fgsm_params)

  preds_2_adv = model_2(attack(x))
  loss_2 = CrossEntropy(wrap_2, smoothing=label_smoothing, attack=attack)

  def evaluate_2():
    # Accuracy of adversarially trained model on legitimate test inputs
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds_2, x_test, y_test,
                          args=eval_params)
    print('Test accuracy on legitimate examples: %0.4f' % accuracy)
    report.adv_train_clean_eval = accuracy

    # Accuracy of the adversarially trained model on adversarial examples
    accuracy = model_eval(sess, x, y, preds_2_adv, x_test,
                          y_test, args=eval_params)
    print('Test accuracy on adversarial examples: %0.4f' % accuracy)
    report.adv_train_adv_eval = accuracy

  # Perform and evaluate adversarial training
  train(sess, loss_2, x_train, y_train, evaluate=evaluate_2,
        args=train_params, rng=rng)

  # Calculate training errors
  if testing:
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds_2, x_train, y_train,
                          args=eval_params)
    report.train_adv_train_clean_eval = accuracy
    accuracy = model_eval(sess, x, y, preds_2_adv, x_train,
                          y_train, args=eval_params)
    report.train_adv_train_adv_eval = accuracy

  return report
'''
