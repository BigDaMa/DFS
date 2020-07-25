from art.classifiers.scikitlearn import ScikitlearnClassifier
from art.classifiers.scikitlearn import ClassifierGradients
from fastsklearnfeature.test.test_robustness.ScikitlearnRegressor import ScikitlearnRegressor

import numpy as np

class ScikitlearnLinearRegression(ScikitlearnRegressor, ClassifierGradients):
    """
    Wrapper class for scikit-learn Logistic Regression models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn Logistic Regression model.

        :param model: scikit-learn LogisticRegression model
        :type model: `sklearn.linear_model.LogisticRegression`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """

        super(ScikitlearnLinearRegression, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                                            preprocessing=preprocessing)
        self._model = model

    def nb_classes(self):
        raise NotImplementedError('hopefully not called')

    def class_gradient(self, x, label=None, **kwargs):
        raise NotImplementedError('hopefully not called')

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        | Paper link: http://cs229.stanford.edu/proj2016/report/ItkinaWu-AdversarialAttacksonImageRecognition-report.pdf
        | Typo in https://arxiv.org/abs/1605.07277 (equation 6)

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # pylint: disable=E0001
        from sklearn.utils.class_weight import compute_class_weight

        if not hasattr(self._model, 'coef_'):
            raise ValueError("""Model has not been fitted. Run function `fit(x, y)` of classifier first or provide a
            fitted model.""")

        # Apply preprocessing
        #x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        num_samples, _ = x.shape
        gradients = np.zeros(x.shape)
        y_pred = self._model.predict(X=x)

        for i_sample in range(num_samples):
            gradients[i_sample, :] = 2 * x[i_sample] * (y[i_sample] - y_pred[i_sample])


        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients