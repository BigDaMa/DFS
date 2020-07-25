import numpy as np
from art.classifiers.classifier import Classifier

class ScikitlearnRegressor(Classifier):
    """
    Wrapper class for scikit-learn classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn classifier model.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: scikit-learn classifier model.
        :type model: `sklearn.base.BaseEstimator`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(ScikitlearnRegressor, self).__init__(clip_values=clip_values, defences=defences, preprocessing=preprocessing)

        self._model = model
        self._input_shape = self._get_input_shape(model)

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn` classifier and will be passed to this function as such.
        :type kwargs: `dict`
        :return: `None`
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)
        y_preprocessed = np.argmax(y_preprocessed, axis=1)

        self._model.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._input_shape = self._get_input_shape(self._model)
        self._nb_classes = self._get_nb_classes()

    def predict(self, x, **kwargs):
        return self._model.predict(x)

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int` or `None`
        """
        if hasattr(self._model, 'n_classes_'):
            _nb_classes = self._model.n_classes_
        else:
            _nb_classes = None
        return _nb_classes

    def save(self, filename, path=None):
        import pickle
        with open(filename + '.pickle', 'wb') as file_pickle:
            pickle.dump(self._model, file=file_pickle)

    def _get_input_shape(self, model):
        if hasattr(model, 'n_features_'):
            _input_shape = (model.n_features_,)
        elif hasattr(model, 'feature_importances_'):
            _input_shape = (len(model.feature_importances_),)
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                _input_shape = (model.coef_.shape[0],)
            else:
                _input_shape = (model.coef_.shape[1],)
        elif hasattr(model, 'support_vectors_'):
            _input_shape = (model.support_vectors_.shape[1],)
        elif hasattr(model, 'steps'):
            _input_shape = self._get_input_shape(model.steps[0][1])
        elif hasattr(model, 'sigma_'):
            _input_shape = (model.sigma_.shape[1],)
        else:
            logger.warning('Input shape not recognised. The model might not have been fitted.')
            _input_shape = None
        return _input_shape

