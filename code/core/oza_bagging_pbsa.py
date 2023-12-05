"""
This file is related to the base JIT-SDP model PBSA.
Cong Teng write it based on oza_bagging_oob.py.
"""
import copy as cp
import collections

from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.utils.utils import *
from skmultiflow.utils import check_random_state


class OzaBaggingClassifier_PBSA(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ PBSA bagging ensemble classifier

    Reference:
    .. G. CCabral, L. Minku. "Towards reliable online just-in-time software defect prediction". TSE, 2022.

    This class is modified by Cong based on Liyan Song's OOB codes.
    """

    def __init__(self, base_estimator=KNNADWINClassifier(), n_estimators=10, random_state=None, theta_imb=0.9, p=0.25,
                 m=1.5, th=0.3):
        super().__init__()
        # default values
        self.ensemble = None
        self.actual_n_estimators = None
        self.classes = None
        self._random_state = None  # This is the actual random_state object used internally
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.__configure()
        ''' Liyan, 2021-7-22 
        rho0: the size of the clean class (0), corresponding to w- in Table 2 of Shuo paper.
        rho1: the size of the defect-inducing class (+1), corresponding to w+ in Table 2 of Shuo paper.
        theta_imb: w_new = theta_imb * w_old + (1 - theta_imb) * [(x, c)]
        '''
        self.rho0 = 0.5
        self.rho1 = 0.5
        self.theta_imb = theta_imb  # factor to update tho0 and tho1 at each test time step
        ''' TC, 2022-11-7
        l0 and l1: control the maximum boosting factor values (the boosting factors varies from 1 to 1 + l0 and 1 + l1)
        m: determines the growth of the exponential function
        th: stands for the threshold that indicates which class must be boosted
        ma_size: window size to calculate moving average of predictions
        '''
        self.l0 = 10
        self.l1 = 12
        self.m = m
        self.th = th
        self.ma_size = 100
        ''' TC, 2022-11-14
        at: interval of the concept drift recovering mechanism
        p: percentage of allowed deviation from th
        alpha and beta: the shape of the beta distribution
        '''
        self.at = 30
        self.p = p
        self.alpha = 5
        self.beta = 2
        ''' TC, 2022-11-14
        ma: history ma values
        ma_window: prediction window
        ave_ma: average of ma values
        '''
        self.ma = collections.deque(maxlen=20)
        self.ma_window = collections.deque(maxlen=self.ma_size)
        self.ave_ma = 0
        ''' TC, 2022-11-14
        last_clean_ts: last time for recovering from clean
        last_defect_ts: last time for recovering from defect
        pool_def: maintain the defect data
        pool_unlabel: maintain the unlabel data
        '''
        self.last_clean_ts = 0
        self.last_defect_ts = 0
        self.pool_def = []
        self.pool_unlabel = []

    def __calculate_ma(self, y_pre):
        """ TC 2022-11-7
        This method used to maintain the variables about moving average of predictions
        :param y_pre: the prediction label
        :return: None
        """
        self.ma_window.append(y_pre)
        self.ma.append(np.mean(self.ma_window))
        self.ave_ma = np.mean(self.ma)

    def __configure(self):
        if hasattr(self.base_estimator, "reset"):
            self.base_estimator.reset()
        self.actual_n_estimators = self.n_estimators
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.actual_n_estimators)]
        self._random_state = check_random_state(self.random_state)

    def reset(self):
        self.__configure()
        return self

    def pbsa_flow(self, X, y, ts, new_unlabel, pool_def, data_ind_reset, classes=None, sample_weight=None):
        """ TC:
        The core part of PBSA.

        Parameters
        ----------
        :param X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        :param y: numpy.ndarray of shape (n_samples)
            An array-like with the class yy of all XX in XX.

        :param ts: int
            Current time step.

        :param new_unlabel: numpy.ndarray of shape (n_samples, n_features)
            Unlabeled data at ts.

        :param pool_def: numpy.ndarray of shape (n_samples, time + n_features + vl + label)
            The coming defect data.

        :param data_ind_reset: numpy.ndarray
            Index of a complete data, which contains (id_time, id_vl, id_y, id_X_np: np, n_fea)

        :param classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class yy. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        :param sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending
            on the base estimator.

        """

        """TC: update pool_unlabel and pool_def"""
        self.pool_unlabel = new_unlabel
        for each in pool_def:
            np.append(self.pool_def, each)
        """TC: use X and y to train the classifier"""
        self.train_model(X, y, classes, sample_weight)
        """TC: update th0 and th1"""
        th0 = self.th + (1 - self.th) * self.p
        th1 = self.th - self.th * self.p
        """TC: if the classifier became exceedingly skewed towards class 1 (defect-inducing), recover from it"""
        if (self.ave_ma > th0) and (self.last_defect_ts + self.at) < ts:
            self.recover_from_defect(classes)
            self.last_defect_ts = ts
        """TC: if the classifier became exceedingly skewed towards class 0 (clean), recover from it"""
        if (self.ave_ma < th1) and (self.last_clean_ts + self.at) < ts:
            self.recover_from_clean(classes, data_ind_reset)
            self.last_clean_ts = ts

    def train_model(self, X, y, classes, sample_weight):
        """ Liyan:
        The only function that had been changed on Oza bagging for OOB in Liyan's code

        The below is the original documentation.
        =========================================

        Partially (incrementally) fit the model.

        Parameters
        ----------
        :param X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        :param y: numpy.ndarray of shape (n_samples)
            An array-like with the class yy of all XX in XX.

        :param classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class yy. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        :param sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending
            on the base estimator.

        Raises
        ------
        ValueError
            A ValueError is raised if the 'classes' parameter is not passed in the first
            partial_fit call, or if they are passed in further calls but differ from
            the initial classes list passed.

        Returns
        -------
        OzaBaggingClassifier
            self

        Notes
        -----
        Since it's an ensemble learner, if XX and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Each sample is trained by each classifier a total of K times, where K
        is drawn by a Poisson(1) distribution.

        """
        if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes

        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError("The classes passed to the partial_fit function differ from those passed earlier.")

        self.__adjust_ensemble_size()

        """Liyan revision is mainly in this part 2021/8/31"""
        r, _ = get_dimensions(X)
        for j in range(r):
            '''Liyan'''
            self.rho1 = self.theta_imb * self.rho1 + (1 - self.theta_imb) * (1 if y[j] == 1 else 0)
            self.rho0 = self.theta_imb * self.rho0 + (1 - self.theta_imb) * (1 if y[j] == 0 else 0)

            """Liyan: compute the lambda of Poisson distribution
            2021/8/31 found the huge bug revised as below
            """
            if y[j] == 1 and self.rho0 > self.rho1:
                lambda_poisson = self.rho0 / self.rho1
            elif y[j] == 0 and self.rho0 < self.rho1:
                lambda_poisson = self.rho1 / self.rho0
            else:
                lambda_poisson = 1

            """Liyan: get K from Poisson(lambda_poisson) distribution"""
            for i in range(self.actual_n_estimators):  # all base learners
                k = self._random_state.poisson(lambda_poisson)  # core revision from Oza_bagging to OOB
                # k = self._random_state.poisson()  # original oza code
                if k > 0:
                    self.ensemble[i].partial_fit([X[j]], [y[j]], classes, [k])
        return self

    def recover_from_clean(self, classes, data_ind_reset):
        """ TC:

        Recover from the exceedingly skewed towards class 0 (clean).

        Parameters
        ----------
        :param classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class yy. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        :param data_ind_reset: numpy.ndarray
            Index of a complete data, which contains (id_time, id_vl, id_y, id_X_np: np, n_fea)

        """
        arr_test = []
        if len(self.pool_unlabel) > self.ma_size:
            for i in range(len(self.pool_unlabel) - 1, len(self.pool_unlabel) - self.ma_size, -1):
                arr_test.append(self.pool_unlabel[i])
        else:
            arr_test = self.pool_unlabel.copy()
        old_avg = 0
        count_noincrease = 0
        avg = 0
        if len(self.pool_def) > 1:
            self.pool_def = np.array(self.pool_def)
            idx_for_sort = self.pool_def[:, data_ind_reset.id_time].argsort()
            self.pool_def = self.pool_def[idx_for_sort]
        while True:
            if len(self.pool_def) == 0:
                break
            b = self._random_state.beta(self.alpha, self.beta)
            idx = int(b * (len(self.pool_def) - 1))
            x = self.pool_def[idx:idx + 1]
            x = x[data_ind_reset.id_X_np]
            self.train_for_recover(x, [1], classes)
            pre = []
            for each in arr_test:
                pre.append(self.predict_for_recover(each))
            if old_avg > avg or count_noincrease > 100:
                break
            else:
                if avg == old_avg:
                    count_noincrease = count_noincrease + 1
                else:
                    old_avg = avg
            if avg > self.th:
                break

    def recover_from_defect(self, classes):
        """ TC:

        Recover from the exceedingly skewed towards class 1 (defect-inducing).

        Parameters
        ----------
        :param classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class yy. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        """
        arr_test = []
        if len(self.pool_unlabel) > self.ma_size:
            for i in range(len(self.pool_unlabel) - 1, len(self.pool_unlabel) - self.ma_size, -1):
                arr_test.append(self.pool_unlabel[i])
        else:
            arr_test = self.pool_unlabel.copy()
        old_avg = 1
        count_noincrease = 0
        avg = 0
        while True:
            if len(self.pool_unlabel) == 0:
                break
            b = self._random_state.beta(self.alpha, self.beta)
            idx = int(b * (len(self.pool_unlabel) - 1))
            x = self.pool_unlabel[idx:idx + 1]
            self.train_for_recover(x, [0], classes)
            pre = []
            for each in arr_test:
                pre.append(self.predict_for_recover(each))
            avg = np.mean(pre)
            if old_avg < avg or count_noincrease > 100:
                break
            else:
                if avg == old_avg:
                    count_noincrease = count_noincrease + 1
                else:
                    old_avg = avg
            if avg < self.th:
                break

    def train_for_recover(self, X, y, classes):
        """ TC:

        Partially (incrementally) fit the model when recovering without sample_weight.

        Parameters
        ----------
        :param X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        :param y: numpy.ndarray of shape (n_samples)
            An array-like with the class yy of all XX in XX.

        :param classes: numpy.ndarray, optional (default=None)
            Array with all possible/known class yy. This is an optional parameter, except
            for the first partial_fit call where it is compulsory.

        """
        lambda_poisson = 1

        for i in range(self.actual_n_estimators):  # all base learners
            k = self._random_state.poisson(lambda_poisson)  # core revision from Oza_bagging to OOB
            # k = self._random_state.poisson()  # original oza code
            if k > 0:
                self.ensemble[i].partial_fit(X, y, classes, [k])

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.base_estimator))
                    self.actual_n_estimators += 1

    def predict_for_recover(self, X):
        """ Predict classes for the passed features. It will not update moving average.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of Fea14_org XX to predict the class yy for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the XX in XX.

        Notes
        -----
        The predict function will average the predictions from all its learners
        to find the most likely prediction for the sample matrix XX.

        """
        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        predictions = []
        if proba is None:
            return None
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        pre = np.asarray(predictions)
        return pre

    def predict(self, X):
        """ Predict classes for the passed features. It will update moving average.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The set of Fea14_org XX to predict the class yy for.

        Returns
        -------
        A numpy.ndarray with all the predictions for the XX in XX.

        Notes
        -----
        The predict function will average the predictions from all its learners
        to find the most likely prediction for the sample matrix XX.

        """
        r, c = get_dimensions(X)
        proba = self.predict_proba(X)
        predictions = []
        if proba is None:
            return None
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        pre = np.asarray(predictions)
        for i in range(len(pre)):
            self.__calculate_ma(pre[i])
        return pre

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        return self

    def predict_proba(self, X):
        """ Estimates the probability of each sample in XX belonging to each of the class-yy.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The matrix of XX one wants to predict the class probabilities for.

        Returns
        -------
        A numpy.ndarray of shape (n_samples, n_labels), in which each outer entry is associated with the XX entry of the
        same index. And where the list in index [i] contains len(self.target_values) elements, each of which represents
        the probability that the i-th sample of XX belongs to a certain class-label.

        Raises
        ------
        ValueError: A ValueError is raised if the number of classes in the base_estimator
        learner differs from that of the ensemble learner.

        """
        proba = []
        r, c = get_dimensions(X)
        try:
            for i in range(self.actual_n_estimators):
                partial_proba = self.ensemble[i].predict_proba(X)
                if len(partial_proba[0]) > max(self.classes) + 1:
                    raise ValueError("The number of classes in the base learner is larger than in the ensemble.")

                if len(proba) < 1:
                    for n in range(r):
                        proba.append([0.0 for _ in partial_proba[n]])

                for n in range(r):
                    for l in range(len(partial_proba[n])):
                        try:
                            proba[n][l] += partial_proba[n][l]
                        except IndexError:
                            proba[n].append(partial_proba[n][l])
        except ValueError:
            return np.zeros((r, 1))
        except TypeError:
            return np.zeros((r, 1))

        # normalizing probabilities
        sum_proba = []
        for l in range(r):
            sum_proba.append(np.sum(proba[l]))
        aux = []
        for i in range(len(proba)):
            if sum_proba[i] > 0.:
                aux.append([x / sum_proba[i] for x in proba[i]])
            else:
                aux.append(proba[i])
        return np.asarray(aux)
