"""
This file is core code.
It contains the following core context:
The implementation of AIO, Filtering, CroPS and Multi-CroPS;
Parameter tuning for JIT-SDP models and CP methods;
The online running process of JIT-SDP models and CP methods;
"WP+1CP" experiments for RQ1;
The experiments for RQ2 and RQ3.
"""

import collections
import copy
import itertools
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from sklearn.decomposition import PCA
from sklearn import preprocessing
from datetime import datetime
from collections import Counter
from itertools import product
from sklearn.metrics import mean_squared_error

from evaluate.evaluation_online import compute_online_PF

from data.real_data_stream import data_id_2name, set_test_stream, set_train_stream
from data.real_data_stream import class_data_ind_org, class_data_ind_reset
from data.real_data_preprocess import real_data_preprocess

from DenStream.DenStream import DenStream
from DenStream.DenStream_new import DenStream_new
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta.oza_bagging import OzaBaggingClassifier
from core.oza_bagging_oob import OzaBaggingClassifier_OOB
from core.oza_bagging_ooc import OzaBaggingClassifier_OOC
from core.oza_bagging_pbsa import OzaBaggingClassifier_PBSA
import pickle as pkl
from utility import check_random_state, cvt_day2timestamp

# auto para
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.metrics import silhouette_score, recall_score
import detecta
from fs_by_ga import load_selected_features
import time

"""global variables"""
invalid_val, label_val = -1, [0, 1]
dir_rslt_save = "../results/rslt.save/"
with_pretrain = 0


def multi_run(project_id, wait_days):
    """
    This is a method to run the core part named sdp_runs.
    This is used by main_multi_runs.py to run code in multithreading.
    It is deprecated now.

    Args:
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
    """
    sdp_runs("pbsa", project_id=project_id, nb_para_tune=0, nb_test=-1, wait_days=wait_days,
             seed_lst=range(30), verbose_int=1, pca_plot=False)

    print("Succeed in sdp_study() on %s" % datetime.today())


def multi_run_more(project_id, wait_days, clf):
    """
    This is a method to run the core part named sdp_runs.
    This is used by main_multi_runs.py to run code in multithreading.

    Args:
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        clf (string): The name of base JIT-SDP model and the CP method.
    """
    sdp_runs(clf, project_id=project_id, nb_para_tune=1000, nb_test=-1, wait_days=wait_days,
             seed_lst=range(20), verbose_int=1, pca_plot=True)
    print("Succeed in sdp_study() on %s" % datetime.today())


def run_window_similarity(clf, project_id):
    """
    This method is used to run sdp_runs_window_similarity.

    Args:
        clf (string): The name of base JIT-SDP model and the CP method.
        project_id (int): The index of target project (WP).
    """
    print("start: "+str(project_id))
    sdp_runs_window_similarity(clf, project_id=project_id, nb_para_tune=1000, nb_test=5010, wait_days=15,
                               seed_lst=range(1), verbose_int=1, pca_plot=False)


def filtering_cross_data(test_data, waiting_days, norm_scaler, project_id, nb_para_tune,
                         window_size, K, max_dist, discard_size):
    """
    This method is related to Filtering (A state-of-the-art online CP method).
    This method is used to select the CP data which is similar to recent WP data.
    We use Filtering on whole data stream first to reduce the calculation time.
    In this process, we guarantee the same results as when using the Filtering method in real online scenario.

    Args:
        test_data (list): The whole data stream.
        waiting_days (int): The waiting time in online JIT-SDP.
        norm_scaler (object): An object used to normalize the data.
        project_id (int): The index of target project (WP).
        nb_para_tune (int): The number of WP data used to do parameter tuning, including the CP data before them.
        window_size (int): A parameter of Filtering, which is the size of the sliding window.
        K (int): A parameter of Filtering, which is the number of WP data used to calculate the distance of CP data.
        max_dist (int): A parameter of Filtering, which is a threshold to select CP data.
        discard_size (int): A parameter of Filtering, which is the size of the queue of discarded CP data.

    Returns:
        data (list): The data of WP and filtered CP data.
    """
    data = test_data.copy()
    data[:, 1:-3] = norm_scaler.my_transform(data[:, 1:-3])
    WP_window = collections.deque(maxlen=window_size)
    CP_discard = collections.deque(maxlen=discard_size)
    CP_discard_index = collections.deque(maxlen=discard_size)
    usedata = []

    index = 0
    CP_count = 0
    CP_count_0 = 0
    CP_count_1 = 0

    for i in range(index, len(data)):
        current_time = data[i][0]
        if data[i][-1] != project_id:
            use = check_CP_use(data[i], WP_window, window_size, K, max_dist, current_time, waiting_days)
            if use == 1:
                usedata.append(i)
                CP_count = CP_count + 1
                if data[i][-2] == 0:
                    CP_count_0 = CP_count_0 + 1
                else:
                    CP_count_1 = CP_count_1 + 1
            else:
                put_into_CP_discard(i, data[i], CP_discard, CP_discard_index)
        else:
            maintain_WP_window(data[i], WP_window)
            usedata.append(i)
            check_CP_discard(CP_discard, WP_window, CP_discard_index, discard_size, window_size, K, max_dist, usedata,
                             current_time, waiting_days)
        # check_CP_discard(CP_discard, WP_window, CP_discard_index, discard_size, window_size, K, max_dist, usedata)
    usedata = np.array(usedata)
    usedata = usedata.astype(int)
    np.unique(usedata)
    idx = usedata.argsort()
    usedata = usedata[idx]
    data = test_data[usedata, :]
    # print(CP_count)
    # print(CP_count_0)
    # print(CP_count_1)
    return data


def filtering_cross_data_for_para(test_data, waiting_days, norm_scaler, project_id, nb_para_tune,
                                  window_size, K, max_dist, discard_size):
    """
    This method is only used for parameter tuning about Filtering (A state-of-the-art online CP method).
    This method is used to select the CP data which is similar to recent WP data.
    We will use Filtering on the data stream before the WP data with time step nb_para_tune.

    Args:
        test_data (list): The whole data stream.
        waiting_days (int): The waiting time in online JIT-SDP.
        norm_scaler (object): An object used to normalize the data.
        project_id (int): The index of target project (WP).
        nb_para_tune (int): The number of WP data used to do parameter tuning, including the CP data before them.
        window_size (int): A parameter of Filtering, which is the size of the sliding window.
        K (int): A parameter of Filtering, which is the number of WP data used to calculate the distance of CP data.
        max_dist (int): A parameter of Filtering, which is a threshold to select CP data.
        discard_size (int): A parameter of Filtering, which is the size of the queue of discarded CP data.

    Returns:
        data (list): The data of WP and filtered CP data.
    """
    data = test_data.copy()
    data[:, 1:-3] = norm_scaler.my_transform(data[:, 1:-3])
    WP_window = collections.deque(maxlen=window_size)
    CP_discard = collections.deque(maxlen=discard_size)
    CP_discard_index = collections.deque(maxlen=discard_size)
    usedata = []

    index = 0
    CP_count = 0
    CP_count_0 = 0
    CP_count_1 = 0

    count_target = 0
    for i in range(index, len(data)):
        if count_target == nb_para_tune:
            break
        current_time = data[i][0]
        if data[i][-1] != project_id:
            use = check_CP_use(data[i], WP_window, window_size, K, max_dist, current_time, waiting_days)
            if use == 1:
                usedata.append(i)
                CP_count = CP_count + 1
                if data[i][-2] == 0:
                    CP_count_0 = CP_count_0 + 1
                else:
                    CP_count_1 = CP_count_1 + 1
            else:
                put_into_CP_discard(i, data[i], CP_discard, CP_discard_index)
        else:
            count_target = count_target + 1
            maintain_WP_window(data[i], WP_window)
            usedata.append(i)
            check_CP_discard(CP_discard, WP_window, CP_discard_index, discard_size, window_size, K, max_dist, usedata,
                             current_time, waiting_days)
        # check_CP_discard(CP_discard, WP_window, CP_discard_index, discard_size, window_size, K, max_dist, usedata)
    usedata = np.array(usedata)
    usedata = usedata.astype(int)
    np.unique(usedata)
    idx = usedata.argsort()
    usedata = usedata[idx]
    data = test_data[usedata, :]
    # print(CP_count)
    # print(CP_count_0)
    # print(CP_count_1)
    return data


def check_CP_discard(CP_discard, WP_window, CP_discard_index, discard_size, window_size, K, max_dist, usedata,
                     current_time, waiting_days):
    """
    This method is part of the Filtering.
    This method is used to maintain the discard queue.

    Args:
        CP_discard (queue): The queue of discard CP data.
        WP_window (queue): The sliding window of recent WP data.
        CP_discard_index (queue): The queue of index of discard CP data.
        discard_size (int): A parameter of Filtering, which is the size of CP_discard.
        window_size (int): A parameter of Filtering, which is the size of WP_window.
        K (int): A parameter of Filtering, which is the number of WP data used to calculate the distance of CP data.
        max_dist (int): A parameter of Filtering, which is a threshold to select CP data.
        usedata (list): The index of the selected data.
        current_time (int): The time stamp of current WP data.
        waiting_days (int): The waiting time in online JIT-SDP.

    """
    if len(CP_discard) < 1:
        return
    temp_CP_discard = collections.deque(maxlen=discard_size)
    temp_CP_discard_index = collections.deque(maxlen=discard_size)
    for i in range(len(temp_CP_discard)):
        CP_data = CP_discard.pop()
        CP_data_index = CP_discard_index.pop()
        use = check_CP_use(CP_data, WP_window, window_size, K, max_dist, current_time, waiting_days)
        if use == 1:
            usedata.append(CP_data_index)
        else:
            temp_CP_discard.append(CP_data)
            temp_CP_discard_index.append(CP_data_index)
        CP_discard = temp_CP_discard.copy()
        CP_discard_index = temp_CP_discard_index.copy()


def check_CP_use(data, WP_window, window_size, K, max_dist, current_time, waiting_days):
    """
    This method is part of the Filtering.
    This method is used to check whether the CP data can be selected.

    Args:
        data (list): The CP data (only one data).
        WP_window (queue): The sliding window of recent WP data.
        window_size (int): A parameter of Filtering, which is the size of WP_window.
        K (int): A parameter of Filtering, which is the number of WP data used to calculate the distance of CP data.
        max_dist (int): A parameter of Filtering, which is a threshold to select CP data.
        current_time (int): The time stamp of current WP data.
        waiting_days (int): The waiting time in online JIT-SDP.

    Returns:
        boolean (int): Can this CP data be selected.
    """
    temp_window = WP_window.copy()
    distance = np.zeros(len(temp_window))
    if data[-2] == 1 and current_time >= (data[0] + cvt_day2timestamp(data[-3]) - cvt_day2timestamp(waiting_days)):
        data_label = 1
    else:
        data_label = 0
    for i in range(len(temp_window)):
        WP_data = temp_window.pop()
        distance[i] = cal_distance(WP_data, data, current_time, data_label, waiting_days)
    idx = distance.argsort()
    distance = distance[idx]
    if len(distance) >= K and np.mean(distance[:K]) <= max_dist:
        return 1
    else:
        return 0


def put_into_CP_discard(i, data, CP_discard, CP_discard_index):
    """
    This method is part of the Filtering.
    This method is used to put the discard CP data into discard queue.

    Args:
        i (int): The index of the CP data in whole data stream.
        data (list): The whole data stream.
        CP_discard (queue): The queue of discard CP data.
        CP_discard_index (queue): The queue of index of discard CP data.
    """
    CP_discard.append(data)
    CP_discard_index.append(i)


def cal_distance(WP_data, data, current_time, data_label, waiting_days):
    """
    This method is part of the Filtering.
    This method is used to calculate the distance between the CP data and a recent WP data.

    Args:
        WP_data (list): A WP data in the sliding window (only one data).
        data (list): The CP data (only one data).
        current_time (int): The time stamp of current WP data.
        data_label (int): The class label of the CP data.
        waiting_days (int): The waiting time in online JIT-SDP.

    Returns:
        distance (float): The distance between the CP data and the recent WP data.
    """
    distance = 0
    if WP_data[-2] == 1 and current_time >= (
            WP_data[0] + cvt_day2timestamp(WP_data[-3]) - cvt_day2timestamp(waiting_days)):
        WP_label = 1
    else:
        WP_label = 0
    if WP_label == data_label:
        for i in range(1, 13):
            distance = distance + (WP_data[i] - data[i]) ** 2
        return math.sqrt(distance)
    else:
        return sys.maxsize


def maintain_WP_window(data, WP_window):
    """
    This method is part of the Filtering.
    This method is used to maintain the sliding window of recent WP data.

    Args:
        data (list): A recent WP data (only one data).
        WP_window (queue): The sliding window of recent WP data.
    """
    WP_window.append(data)


def sbp_initial(all_project=range(23), window_size=500):
    """
    This method is part of the CroPS.
    This method is used to initial the sliding window of each project.

    Args:
        all_project (list): The list of index of all projects.
        window_size (int): A parameter of CroPS, which is the size of the sliding window of each project.

    Returns:
        project_window (list): The list of sliding windows of all projects.
        all_project (list): The list of index of all projects.
    """
    project_window = []
    for i in all_project:
        i_window = collections.deque(maxlen=window_size)
        project_window.append(i_window)
    return project_window, all_project


def calculate_spearman_correlation(target_window, cp_window, data_ind_reset):
    """
    This method is part of the CroPS.
    This method is used to calculate the spearman correlation and JS divergence between WP and CP.

    Args:
        target_window (list): The sliding window of WP.
        cp_window (list): The sliding window of a CP.
        data_ind_reset (object): An object to find the index of JIT-SDP features in data.

    Returns:
        speraman_correlation (float): The spearman correlation between WP and the CP.
        js_divergence (float): The JS divergence between WP and the CP.
    """
    target_X, target_y = [], []
    for each in target_window:
        target_y.append(each[data_ind_reset.id_y])
        target_X.append(each[data_ind_reset.id_X_np[1]])
    target_X = np.array(target_X)
    target_y = np.array(target_y)
    target_correlation = []
    for i in range(len(target_X[0])):
        target_correlation.append(scipy.stats.spearmanr(target_X[:, i], target_y)[0])
    k = 3
    index_k = []
    temp_target_correlation = []
    for i in range(k):
        index_i = target_correlation.index(max(target_correlation))
        index_k.append(index_i)
        temp_target_correlation.append(target_correlation[index_i])
        target_correlation[index_i] = -float('inf')

    cp_X, cp_y = [], []
    for each in cp_window:
        cp_y.append(each[data_ind_reset.id_y])
        cp_X.append(each[data_ind_reset.id_X_np[1]])
    cp_X = np.array(cp_X)
    cp_y = np.array(cp_y)
    cp_correlation = []
    for i in index_k:
        cp_correlation.append(scipy.stats.spearmanr(cp_X[:, i], cp_y)[0])

    target_correlation = []
    cp_correlation = []
    a, b, c = index_k[0], index_k[1], index_k[2]

    target_correlation.append(scipy.stats.spearmanr(target_X[:, a], target_X[:, b])[0])
    target_correlation.append(scipy.stats.spearmanr(target_X[:, a], target_X[:, c])[0])
    target_correlation.append(scipy.stats.spearmanr(target_X[:, b], target_X[:, c])[0])
    cp_correlation.append(scipy.stats.spearmanr(cp_X[:, a], cp_X[:, b])[0])
    cp_correlation.append(scipy.stats.spearmanr(cp_X[:, a], cp_X[:, c])[0])
    cp_correlation.append(scipy.stats.spearmanr(cp_X[:, b], cp_X[:, c])[0])
    target_correlation = np.array(target_correlation)
    cp_correlation = np.array(cp_correlation)

    length = min(len(target_X), len(cp_X))
    target_X = target_X[:length]
    cp_X = cp_X[:length]
    M = (target_X + cp_X) / 2
    js = 0.5 * scipy.stats.entropy(target_X, M, base=2) + 0.5 * scipy.stats.entropy(cp_X, M, base=2)
    return np.nansum(np.absolute(target_correlation - cp_correlation)) / 2, np.nansum(js) / 12


def sdp_runs_window_similarity(clf_name="odasc", project_id=6, nb_para_tune=500, nb_test=5000, wait_days=15,
                               seed_lst=range(20), verbose_int=0, pca_plot=True, just_run=False):
    """
    This method is related to the similarity calculation between projects.
    This method is used to calculate the similarity of the data based metrics across time and save them into csv file.

    Args:
        clf_name (string): The name of base JIT-SDP model and the CP method.
        project_id (int): The index of target project (WP).
        nb_para_tune (int): The number of WP data used to do parameter tuning.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        wait_days (int): The waiting time in online JIT-SDP.
        seed_lst (list): The list of random seeds used when running.
        verbose_int (int): A number to control the print of running information. "-1" means no print; a larger value
            means deeper and more detailed "print".
        pca_plot (boolean): A parameter to control whether plot the result.
        just_run (boolean): If True, this method will not load or save results, for safety reason.
    """
    clf_name = clf_name.lower()
    if pca_plot:
        x_lim, y_lim = None, None
    project_name = data_id_2name(project_id)
    info_run = "%s: %s, wtt=%d, #seed=%d" % (clf_name, project_name, wait_days, len(seed_lst))
    if just_run:  # revise the print level to the most detailed level
        verbose_int = 3

    """prepare test data stream"""
    report_nb_test = nb_test
    if clf_name == "oza" or clf_name == "oob" or clf_name == "odasc" or clf_name == "orb" or clf_name == "pbsa":
        test_stream = set_test_stream(project_name)
        test_stream.X = np.hstack(
            (test_stream.X, (np.ones(len(test_stream.X)) * project_id).reshape(len(test_stream.X), 1)))
        X_org = test_stream.X[class_data_ind_org().id_X_np]
        # convert fea14 to fea13 and the test data stream
        XX, use_data = real_data_preprocess(X_org)
        yy = test_stream.y[use_data]
        time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
        vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
        target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

        # handle negative nb_test
        n_data_all, n_fea = XX.shape[0], XX.shape[1]  # after fea conversion for jit-sdp
        assert n_fea == 12, "# transformed fea should be 13. Sth. is wrong."
        if nb_test < 0:
            nb_test += n_data_all
            if verbose_int >= 2:
                print("actual nb_test=%d" % nb_test)
        assert nb_para_tune < nb_test, "nb_pre=%d should be smaller than nb_data=%d" % (nb_para_tune, nb_test)

        # fea normalizer based on all test data used for DenStream
        norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
        norm_scaler.my_fit(XX)
        # print('std:', np.std(norm_scaler.my_transform(XX), axis=0))

        # prepare all test samples
        test_data_all = np.hstack((time, XX, vl, yy, target))  # col=4+12 ~ (time, fea12, vl, yy, target)
        data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                              n_fea=n_fea)

        if nb_test == -1:
            nb_test_WP = len(test_data_all)
        else:
            nb_test_WP = nb_test
        nb_pre = nb_para_tune
        target_idx_para = 0
        target_idx = test_data_all[:, data_ind_reset.id_target] == project_id
    elif clf_name == "odasc_aio" or clf_name == "oob_aio" or clf_name == "orb_aio" or clf_name == "pbsa_aio" \
            or clf_name == "odasc_sbp" or clf_name == "oob_sbp" or clf_name == "orb_sbp" or clf_name == "pbsa_sbp":
        test_stream = set_test_stream(project_name)
        test_stream.X = np.hstack(
            (test_stream.X, (np.ones(len(test_stream.X)) * project_id).reshape(len(test_stream.X), 1)))
        X_org = test_stream.X[class_data_ind_org().id_X_np]
        # convert fea14 to fea13 and the test data stream
        XX, use_data = real_data_preprocess(X_org)
        yy = test_stream.y[use_data]
        time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
        vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
        target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

        # handle negative nb_test
        n_data_all, n_fea = XX.shape[0], XX.shape[1]  # after fea conversion for jit-sdp
        assert n_fea == 12, "# transformed fea should be 13. Sth. is wrong."
        if nb_test < 0:
            nb_test += n_data_all
            if verbose_int >= 2:
                print("actual nb_test=%d" % nb_test)
        assert nb_para_tune < nb_test, "nb_pre=%d should be smaller than nb_data=%d" % (nb_para_tune, nb_test)

        norm_data = XX

        # prepare all test samples
        test_data_all = np.hstack((time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
        data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                              n_fea=n_fea)
        """add cross project data"""
        for i in range(23):
            if i != project_id:
                project_name_cp = data_id_2name(i)
                test_stream = set_test_stream(project_name_cp)
                test_stream.X = np.hstack(
                    (test_stream.X, (np.ones(len(test_stream.X)) * i).reshape(len(test_stream.X), 1)))
                X_org = test_stream.X[class_data_ind_org().id_X_np]
                # convert fea14 to fea13 and the test data stream
                XX, use_data = real_data_preprocess(X_org)
                yy = test_stream.y[use_data]
                time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
                vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
                target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

                test_data_temp = np.hstack((time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
                test_data_all = np.vstack([test_data_all, test_data_temp])

                norm_data = np.vstack([norm_data, XX])

        idx = test_data_all[:, 0].argsort()
        test_data_all = test_data_all[idx]

        # fea normalizer based on all test data used for DenStream
        norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
        norm_scaler.my_fit(norm_data)

        # find the index that contains nb_test target data
        count = 0
        nb_pre = nb_para_tune
        for i in range(len(test_data_all)):
            if test_data_all[i][-1] == project_id:
                count = count + 1
                if nb_test == -1:
                    nb_test_WP = i
            if count == nb_pre:
                nb_para_tune = i
            if count == nb_test and nb_test != -1:
                nb_test_WP = i
                break

        target_idx = test_data_all[:, data_ind_reset.id_target] == project_id
        target_idx_para = target_idx[:nb_para_tune]
        if with_pretrain == 1:
            target_idx = target_idx[nb_para_tune + 1:nb_test_WP + 1]
        else:
            target_idx = target_idx[1:nb_test_WP + 1]

    data_ptrn = test_data_all[:nb_para_tune]
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]

    """main parts across seeds"""
    for ss, seed in enumerate(seed_lst):
        if 1:

            """[core] test-then-training process:
            at each test step, only one test data arrives, while maybe no or several training data become available
            """
            if with_pretrain == 1:
                nb_test_act = nb_test_WP - nb_para_tune
            else:
                nb_test_act = nb_test_WP
            test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
            cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
            cl_train_lst, use_cluster_lst = [], []
            CP_positive_data = []
            CP_positive_size = 50
            pre_r1, pre_r2, pre_gmean = 0, 0, 0

            if with_pretrain == 1:
                prev_test_time, data_buffer, nb_train_data = data_ptrn[-1, data_ind_reset.id_time], None, 0  # vip
            else:
                prev_test_time, data_buffer, nb_train_data = test_data_all[0, data_ind_reset.id_time], None, 0  # vip
            notadd = 0
            # project_window, selected_project = sbp_initial(range(14), 500)
            project_window, selected_project = sbp_initial(range(23), 500)
            use_sbp = 1
            n_commit = np.zeros(len(project_window))
            count_target = 0
            last_count_target = 0
            for tt in range(nb_test_act):
                # get the test data
                if with_pretrain == 1:
                    test_step = tt + nb_para_tune
                else:
                    test_step = tt
                new_1data = test_data_all[test_step, :].reshape((1, -1))
                test_X = new_1data[data_ind_reset.id_X_np]
                test_time[tt] = new_1data[:, data_ind_reset.id_time]
                test_y_tru[tt] = new_1data[:, data_ind_reset.id_y]
                if new_1data[0, data_ind_reset.id_target] == project_id:
                    count_target = count_target + 1

                if use_sbp == 1 and count_target % 5000 == 0 and count_target != last_count_target:
                    last_count_target = count_target
                    metrics_dis, spearman_cor, js_div = calculate_U_similarity(project_id, project_window,
                                                                               data_ind_reset, n_commit)
                    if not just_run:
                        to_dir_csv = "../results/rslt.report/"
                        os.makedirs(to_dir_csv, exist_ok=True)
                        to_flnm_csv = to_dir_csv + "pf_bst_ave%d_p%d_n%d_window_similarity.csv" % (
                            len(seed_lst), nb_pre, report_nb_test)
                        with open(to_flnm_csv, "a+") as fh2:
                            if not os.path.getsize(to_flnm_csv):  # header
                                print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
                                    "target_project", "time_steps", "method", "cp_id", "spearman_cor",
                                    "js_div", "defect_ratio", "median_feature", "maximum_feature", "std_feature",
                                    "n_commit"), file=fh2)
                            for z in range(len(project_window)):
                                print("%s,%d,%s,%d,%f,%f,%f,%f,%f,%f,%f" % (
                                    project_name, count_target, clf_name, z, spearman_cor[z], js_div[z],
                                    metrics_dis[z][0],
                                    metrics_dis[z][1], metrics_dis[z][2], metrics_dis[z][3], metrics_dis[z][4]),
                                      file=fh2)

                if new_1data[0, data_ind_reset.id_target] == project_id:
                    target_idx[tt] = True
                    if use_sbp == 1:
                        n_commit[project_id] = n_commit[project_id] + 1

                    """get the new train data batch"""
                    data_buffer, new_train_defect, new_train_clean, new_train_unlabeled = \
                        set_train_stream(prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer,
                                         wait_days)

                    if 1:
                        for each in new_train_clean:
                            each[data_ind_reset.id_y] = 0
                            project_window[int(each[data_ind_reset.id_target])].append(each)
                        for each in new_train_defect:
                            each[data_ind_reset.id_y] = 1
                            project_window[int(each[data_ind_reset.id_target])].append(each)
                        stay_clean = np.in1d(new_train_clean[:, data_ind_reset.id_target], selected_project)
                        stay_defect = np.in1d(new_train_defect[:, data_ind_reset.id_target], selected_project)
                        new_train_clean = new_train_clean[stay_clean]
                        new_train_defect = new_train_defect[stay_defect]
                        # note the order (clean, defect)
                        cmt_time_train = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_time], new_train_defect[:, data_ind_reset.id_time]))
                        use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                        X_train = np.concatenate(
                            (new_train_clean[data_ind_reset.id_X_np], new_train_defect[data_ind_reset.id_X_np]))
                        y_train_obv = np.concatenate(
                            (np.zeros(new_train_clean.shape[0]), np.ones(new_train_defect.shape[0])))
                        y_train_tru = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_y], new_train_defect[:, data_ind_reset.id_y]))
                        y_train_target = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_target],
                             new_train_defect[:, data_ind_reset.id_target]))
                        nb_train_data += y_train_obv.shape[0]

                    prev_test_time = test_time[tt]  # update VIP
                else:
                    target_idx[tt] = False
                    if use_sbp == 1:
                        n_commit[int(new_1data[0, data_ind_reset.id_target])] = n_commit[int(
                            new_1data[0, data_ind_reset.id_target])] + 1
                    test_y_pre[tt] = -1
                    if new_1data.ndim == 1:  # debug
                        new_1data = new_1data.reshape((1, -1))
                    if new_1data[0, data_ind_reset.id_y] == 0:
                        new_1data[0, data_ind_reset.id_vl] = np.inf
                    # set data_buffer, (ts, XX, vl)
                    if data_buffer is None:  # initialize
                        data_buffer = new_1data
                    else:
                        data_buffer = np.vstack((data_buffer, new_1data))
    return 0


def calculate_U_similarity(target_id, project_window, data_ind_reset, n_commits):
    """
    This method is related to the similarity calcualtion.
    This method is used to calculate the simialrities between WP and CPs.

    Args:
        target_id (int): The index of WP.
        project_window (list): The list of sliding windows of all projects.
        data_ind_reset (object): An object to find the index of JIT-SDP features in data.
        n_commits (list): The list of the number of existing data of each project.

    Returns:
        metrics_distance (list): The distance of all project-level metrics except spearman correlation and js divergence
            between WP and CPs.
        spearman_correlation (list): The spearman correlation between WP and CPs.
        js_divergence (list): The js divergence between WP and CPs.
    """
    defect_ratio = []
    median_feature = []
    maximum_feature = []
    std_feature = []
    spearman_correlation = []
    js_divergence = []
    similarity_U = np.zeros(len(project_window))
    for i in range(len(project_window)):
        if len(project_window[i]) == 0:
            similarity_U[i] = np.random.rand()
            defect_ratio.append(np.nan)
            nan_arr = np.zeros(12)
            for j in range(len(nan_arr)):
                nan_arr[j] = np.nan
            median_feature.append(nan_arr)
            maximum_feature.append(nan_arr)
            std_feature.append(nan_arr)
            spearman_correlation.append(np.nan)
            js_divergence.append(np.nan)
        else:
            temp_data = project_window[i]
            temp_length = len(project_window[i])
            temp_y, temp_X = [], []
            defect_count = 0
            for each in temp_data:
                temp_y.append(each[data_ind_reset.id_y])
                temp_X.append(each[data_ind_reset.id_X_np[1]])
                if each[data_ind_reset.id_y] == 1:
                    defect_count += 1
            defect_ratio.append(defect_count / temp_length)
            temp_median, temp_maximum, temp_std = [], [], []
            temp_X = np.array(temp_X)
            for j in range(len(temp_X[0])):
                temp_median.append(np.median(temp_X[:, j]))
                temp_maximum.append(np.max(temp_X[:, j]))
                temp_std.append(np.std(temp_X[:, j]))
            median_feature.append(temp_median)
            maximum_feature.append(temp_maximum)
            std_feature.append(temp_std)

            if i != target_id and len(project_window[target_id]) != 0:
                sp, js = calculate_spearman_correlation(project_window[target_id], project_window[i], data_ind_reset)
                spearman_correlation.append(sp)
                js_divergence.append(js)
            else:
                spearman_correlation.append(np.nan)
                js_divergence.append(np.nan)
            #     js_divergence.append(calculate_js_divergence(project_window[target_id], project_window[i]))

    defect_ratio = np.reshape(defect_ratio, [len(project_window), -1])
    commit = np.reshape(n_commits, [len(project_window), -1])
    metrics = np.concatenate([defect_ratio, median_feature, maximum_feature, std_feature, commit], axis=1)
    for j in range(metrics.shape[1]):
        temp_min = np.nanmin(metrics[:, j])
        temp_max = np.nanmax(metrics[:, j])
        metrics[:, j] = (metrics[:, j] - temp_min) / (temp_max - temp_min)
    # combine all metrics as a whole value
    # metrics_dis = np.zeros([len(metrics), len(metrics)])
    # for i in range(metrics_dis.shape[0]):
    #     for j in range(metrics_dis.shape[1]):
    #         temp = np.absolute(metrics[i] - metrics[j])
    #         metrics_dis[i][j] = np.nansum(temp)

    # use all metrics as a single value
    metrics_dis = np.zeros([len(metrics), len(metrics), 5])
    for i in range(metrics_dis.shape[0]):
        for j in range(metrics_dis.shape[1]):
            temp = np.absolute(metrics[i] - metrics[j])
            # metrics_dis[i][j] = np.nansum(temp)
            metrics_dis[i][j][0] = temp[0]
            metrics_dis[i][j][4] = temp[-1]
            metrics_dis[i][j][1] = np.nansum(temp[1:13]) / 12
            metrics_dis[i][j][2] = np.nansum(temp[13:25]) / 12
            metrics_dis[i][j][3] = np.nansum(temp[25:37]) / 12

    spearman_correlation = np.array(spearman_correlation)
    js_divergence = np.array(js_divergence)
    return -metrics_dis[target_id], -spearman_correlation, -js_divergence


def update_cp_weight(target_id, project_window, data_ind_reset, n_commits, combine_way, selected_features):
    """
    This method is related to similarity calcualtion.
    This method is used to calculat the simialrities between WP and CPs.

    Args:
        target_id (int): The index of WP.
        project_window (list): The list of sliding windows of all projects.
        data_ind_reset (object): An object to find the index of JIT-SDP features in data.
        n_commits (list): The list of the number of existing data of each project.
        combine_way (string): The type of method to calculate the sum of all distance on each metric.
        selected_features (list): The metrics used in similarity calculation.

    Returns:
        S_similarity (list): The similarities between WP and CPs.
    """
    # D_similarity shape: (10,16)-(features,cross-projects)
    D_similarity = load_D_similarity(target_id)
    # metrics_dis shape: (16,5)-(cp, features), feature order: defect, median, max, std, commit
    metrics_dis, sp_cor, js_dev = calculate_U_similarity(target_id, project_window, data_ind_reset, n_commits)
    # order of features: start, core, license, language, domain, company, user_interface, use_database, localized,
    #                       single_pl, defect, commit, median, max, std, sp, js
    S_similarity = np.zeros(23)

    if combine_way == "L1":
        for i in range(10):
            if i in selected_features:
                for j in range(23):
                    added_value = D_similarity[i][j]
                    if not np.isnan(added_value):
                        S_similarity[j] = S_similarity[j] + added_value
        for j in range(23):
            added = []
            if 10 in selected_features:
                added.append(metrics_dis[j][0])
            if 11 in selected_features:
                added.append(metrics_dis[j][4])
            if 12 in selected_features:
                added.append(metrics_dis[j][1])
            if 13 in selected_features:
                added.append(metrics_dis[j][2])
            if 14 in selected_features:
                added.append(metrics_dis[j][3])
            if 15 in selected_features:
                added.append(sp_cor[j])
            if 16 in selected_features:
                added.append(js_dev[j])
            added = np.array(added)
            added_value = np.nansum(added)
            S_similarity[j] = S_similarity[j] + added_value
        for j in range(len(S_similarity)):
            if S_similarity[j] < 0:
                S_similarity[j] = S_similarity[j] * -1
    # not fit feature selection
    elif combine_way == "L2":
        for i in range(10):
            for j in range(23):
                added_value = D_similarity[i][j]
                if not np.isnan(added_value):
                    S_similarity[j] = S_similarity[j] + added_value * added_value
        for j in range(23):
            added = []
            added.append(metrics_dis[j][0] * metrics_dis[j][0])
            added.append(metrics_dis[j][4] * metrics_dis[j][4])
            added.append(metrics_dis[j][1] * metrics_dis[j][1])
            added.append(metrics_dis[j][2] * metrics_dis[j][2])
            added.append(metrics_dis[j][3] * metrics_dis[j][3])
            added.append(sp_cor[j] * sp_cor[j])
            added.append(js_dev[j] * js_dev[j])
            added = np.array(added)
            added_value = np.nansum(added)
            S_similarity[j] = S_similarity[j] + added_value
        for j in range(len(S_similarity)):
            S_similarity[j] = math.pow(S_similarity[j], 0.5)
    else:
        raise Exception("There are no this combine way")
    S_similarity = S_similarity + 1
    for j in range(len(S_similarity)):
        S_similarity[j] = 1 / S_similarity[j]
    return S_similarity


def get_filtering_data(test_data_all, wait_days, norm_scaler, project_id, nb_para_tune, nb_test, data_ind_reset,
                       window_size, K, max_dist, discard_size, for_para):
    """
    This method is related to Filtering (A state-of-the-art online CP method).
    This method is used to get the filtered data on whole data stream.
    In this process, we guarantee the same results as when using the Filtering method in real online scenario.

    Args:
        test_data_all (list): The whole data stream.
        wait_days (int): The waiting time in online JIT-SDP.
        norm_scaler (object): An object used to normalize the data.
        project_id (int): The index of target project (WP).
        nb_para_tune (int): The number of WP data used to do parameter tuning, including the CP data before them.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        data_ind_reset (object): An object to find the index of JIT-SDP features in data.
        window_size (int): A parameter of Filtering, which is the size of the sliding window.
        K (int): A parameter of Filtering, which is the number of WP data used to calculate the distance of CP data.
        max_dist (int): A parameter of Filtering, which is a threshold to select CP data.
        discard_size (int): A parameter of Filtering, which is the size of the queue of discarded CP data.
        for_para (boolean): If True, this method is used for parameter tuning. Else, it is used for real running.

    Returns:
        filtering_test_data_all (list): The whole filtered data stream.
        filtering_target_idx (list): The list of index of WP data in whole filtered data stream.
        filtering_nb_pre (list): The filtered data stream used for pretrain.
        filtering_nb_test_WP (list): The filtered data stream before the last WP data.
        filtering_nb_para_tune (list): The filtered data stream used for parameter tuning.
    """
    if for_para == True:
        filtering_test_data_all = filtering_cross_data_for_para(test_data_all, wait_days, norm_scaler, project_id,
                                                                nb_para_tune,
                                                                window_size, K, max_dist, discard_size)
        # find the index that contains nb_test target data
        count = 0
        filtering_nb_pre = nb_para_tune
        for i in range(len(filtering_test_data_all)):
            if filtering_test_data_all[i][-1] == project_id:
                count = count + 1
            if count == filtering_nb_pre:
                filtering_nb_para_tune = i
                filtering_nb_test_WP = i

        filtering_target_idx = filtering_test_data_all[:, data_ind_reset.id_target] == project_id
        filtering_target_idx_para = filtering_target_idx[:nb_para_tune]
        if with_pretrain == 1:
            filtering_target_idx = filtering_target_idx[nb_para_tune + 1:filtering_nb_test_WP + 1]
        else:
            filtering_target_idx = filtering_target_idx[1:filtering_nb_test_WP + 1]
    elif for_para == False:
        filtering_test_data_all = filtering_cross_data(test_data_all, wait_days, norm_scaler, project_id, nb_para_tune,
                                                       window_size, K, max_dist, discard_size)
        # find the index that contains nb_test target data
        count = 0
        filtering_nb_pre = nb_para_tune
        for i in range(len(filtering_test_data_all)):
            if filtering_test_data_all[i][-1] == project_id:
                count = count + 1
                if nb_test == -1:
                    filtering_nb_test_WP = i
            if count == filtering_nb_pre:
                filtering_nb_para_tune = i
            if count == nb_test and nb_test != -1:
                filtering_nb_test_WP = i
                break

        filtering_target_idx = filtering_test_data_all[:, data_ind_reset.id_target] == project_id
        filtering_target_idx_para = filtering_target_idx[:nb_para_tune]
        if with_pretrain == 1:
            filtering_target_idx = filtering_target_idx[nb_para_tune + 1:filtering_nb_test_WP + 1]
        else:
            filtering_target_idx = filtering_target_idx[1:filtering_nb_test_WP + 1]
    return filtering_test_data_all, filtering_target_idx, filtering_nb_pre, filtering_nb_test_WP, filtering_nb_para_tune


def para_filtering_online_run(filtering_test_data_all, target_idx, clf_name, filtering_nb_para_tune,
                              data_ind_reset, norm_scaler, project_id, filtering_nb_pre,
                              filtering_nb_test_WP, seed_lst, wait_days, window_size, K, max_dist, discard_size,
                              actual_WP_para_tune):
    """
    This method is related to parameter tuning of Filtering (a state-of-the-art online CP method).
    This method is used to running the Filtering with the given parameters based on JIT-SDP models to get the performance.

    Args:
        filtering_test_data_all (list): The whole data stream.
        target_idx (int): The index of WP.
        clf_name (string): The name of base JIT-SDP model and the CP method.
        filtering_nb_para_tune (int): The number of WP data used to do parameter tuning, including the CP data before them.
        data_ind_reset (object): An object to find the index of JIT-SDP features in data.
        norm_scaler (object): An object used to normalize the data.
        project_id (int): The index of target project (WP).
        filtering_nb_pre (int): The number of WP data used to do parameter tuning, including the CP data before them.
        filtering_nb_test_WP (int): The index of the last WP data used for parameter tuning in cpps_test_data_all.
        seed_lst (list): The list of random seeds used when running.
        wait_days (int): The waiting time in online JIT-SDP.
        window_size (int): A parameter of Filtering.
        K (int): A parameter of Filtering.
        max_dist (float): A parameter of Filtering.
        discard_size (int): A parameter of Filtering.
        actual_WP_para_tune (int): The number of WP data used to do parameter tuning.

    Returns:
        average_gmean (float): The average gmean across time and seeds.
        average_recall1 (float): The average recall1 across time and seeds.
        average_recall0 (float): The average recall0 across time and seeds.
    """
    data_ptrn = filtering_test_data_all[:filtering_nb_para_tune]
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]

    """para-auto DenStream~(lambd, eps, beta, mu)"""
    our_clf_lst = (
        "odasc", "odasc_aio", "odasc_filtering", "odasc_addcp_adp", "odasc_sbp_l1",
        "odasc_sbp_l2")  # vip manually maintain
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        X_ptrn_norm = norm_scaler.my_transform(X_ptrn)
        auto_denStream = False
        if auto_denStream:
            eps, mu, beta, lambd = para_denStream(X_ptrn_norm, y_ptrn, nb_repeat=10)
        else:
            eps, mu, beta, lambd = 1.47, 1.57, 0.78, 0.26
            # eps, mu, beta, lambd = 2.09, 2.20, 0.74, 0.125

    """pre-train DenStream"""
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        if clf_name == "odasc_filtering":
            cluster = DenStream(theta_cl=None, lambd=lambd, eps=eps, beta=beta, mu=mu)
            cluster.partial_fit(X_ptrn_norm, y_ptrn)
    else:
        cluster = 0
    """para-auto classifiers~(n_tree, theta_imb, theta_cl)"""
    nb_run = 5  # 30 in systematic exp
    temp = str.split(clf_name, "_")
    dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/" + temp[0] + "/"
    auto_name = "%s-para-%dstep-%drun" % (temp[0], filtering_nb_pre, nb_run) + ".pkl"
    exist_clf_para = os.path.exists(dir_auto_para + auto_name)
    # if exist_clf_para and not just_run:
    if exist_clf_para:
        para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
        n_tree, theta_imb, theta_cl, p, m, th = \
            para_dict["n_trees"], para_dict["theta_imb"], para_dict["theta_cl"], para_dict["p"], \
                para_dict["m"], para_dict["th"]
    else:
        raise Exception("There are no parameters when clf_name=" + temp[0])

    # update DenStream para
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        cluster.theta_cl = theta_cl

    """main parts across seeds"""
    for ss, seed in enumerate(seed_lst):
        to_dir = uti_rslt_dir_filtering_para(clf_name, project_id, wait_days, window_size, K, max_dist, discard_size)
        os.makedirs(to_dir, exist_ok=True)
        # analyze filenames in this dir:
        # find T that is larger than nb_data to save computational cost and load the results.
        exist_result, to_dir = uti_rslt_dir_analyze(to_dir, clf_name, filtering_nb_test_WP, seed)
        if not exist_result:
            to_dir += "/T" + str(filtering_nb_test_WP) + "/"
            os.makedirs(to_dir, exist_ok=True)
        # file_name-s
        flnm_test = "%s%s.rslt_test.s%d" % (to_dir, clf_name, seed)

        """load or compute"""
        if exist_result:
            rslt_test = np.loadtxt(flnm_test)
            rslt_test = rslt_test[:actual_WP_para_tune]
            # cutting the results if nb_test_actual < len(rslt_test)
        else:
            """pre-train classifier"""
            if clf_name == "oob_filtering":
                classifier = OzaBaggingClassifier_OOB(HoeffdingTreeClassifier(), n_tree, seed, theta_imb)
                if with_pretrain == 1:
                    classifier.partial_fit(X_ptrn, y_ptrn, label_val)
            elif clf_name == "odasc_filtering":
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                if with_pretrain == 1:
                    cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)
                    classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn, label_val)
                    cluster_pre = 1
                else:
                    cluster = DenStream(theta_cl=theta_cl, lambd=lambd, eps=eps, beta=beta, mu=mu)
                    cluster_pre = 0
            elif clf_name == "pbsa_filtering":
                classifier = OzaBaggingClassifier_PBSA(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, p, m, th)
                if with_pretrain == 1:
                    y_ptrn_pre = classifier.predict(X_ptrn)
                    classifier.train_model(X_ptrn, y_ptrn, label_val)
            else:
                raise Exception("Undefined clf_name=%s." % clf_name)

            """[core] test-then-training process:
            at each test step, only one test data arrives, while maybe no or several training data become available
            """
            if with_pretrain == 1:
                nb_test_act = filtering_nb_test_WP - filtering_nb_para_tune
            else:
                nb_test_act = filtering_nb_test_WP
            test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
            cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
            cl_train_lst, use_cluster_lst = [], []

            if with_pretrain == 1:
                prev_test_time, data_buffer, nb_train_data = data_ptrn[-1, data_ind_reset.id_time], None, 0  # vip
            else:
                prev_test_time, data_buffer, nb_train_data = filtering_test_data_all[
                    0, data_ind_reset.id_time], None, 0  # vip
            for tt in range(nb_test_act):
                # get the test data
                if with_pretrain == 1:
                    test_step = tt + filtering_nb_para_tune
                else:
                    test_step = tt
                new_1data = filtering_test_data_all[test_step, :].reshape((1, -1))
                test_X = new_1data[data_ind_reset.id_X_np]
                test_time[tt] = new_1data[:, data_ind_reset.id_time]
                test_y_tru[tt] = new_1data[:, data_ind_reset.id_y]

                if new_1data[0, data_ind_reset.id_target] == project_id:
                    target_idx[tt] = True
                    """test: predict with classifiers"""
                    test_y_pre[tt] = classifier.predict(test_X)[0]

                    """get the new train data batch"""
                    data_buffer, new_train_defect, new_train_clean, new_train_unlabeled = \
                        set_train_stream(prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer,
                                         wait_days)

                    # note the order (clean, defect)
                    cmt_time_train = np.concatenate(
                        (new_train_clean[:, data_ind_reset.id_time], new_train_defect[:, data_ind_reset.id_time]))
                    use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                    X_train = np.concatenate(
                        (new_train_clean[data_ind_reset.id_X_np], new_train_defect[data_ind_reset.id_X_np]))
                    y_train_obv = np.concatenate(
                        (np.zeros(new_train_clean.shape[0]), np.ones(new_train_defect.shape[0])))
                    y_train_tru = np.concatenate(
                        (new_train_clean[:, data_ind_reset.id_y], new_train_defect[:, data_ind_reset.id_y]))
                    y_train_target = np.concatenate(
                        (new_train_clean[:, data_ind_reset.id_target],
                         new_train_defect[:, data_ind_reset.id_target]))
                    X_train_weight = np.ones(len(y_train_target))
                    nb_train_data += y_train_obv.shape[0]

                    # assign
                    cmt_time_train_lst.extend(cmt_time_train.tolist())
                    use_time_train_lst.extend(use_time_train.tolist())
                    y_train_obv_lst.extend(y_train_obv.tolist())
                    y_train_tru_lst.extend(y_train_tru.tolist())

                    """then train: update classifiers and DenStream given new labelled training data"""
                    if y_train_obv.shape[0] > 0:
                        if clf_name == "oob_filtering":
                            classifier.partial_fit(X_train, y_train_obv, label_val, X_train_weight)
                            # assign
                            cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                            use_cluster_lst = cl_train_lst
                        elif clf_name == "pbsa_filtering":
                            classifier.pbsa_flow(X_train, y_train_obv, tt, new_train_unlabeled[data_ind_reset.id_X_np],
                                                 new_train_defect, data_ind_reset, label_val, X_train_weight)
                            # assign
                            cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                            use_cluster_lst = cl_train_lst
                        elif clf_name == "odasc_filtering":
                            X_train_norm = norm_scaler.my_transform(X_train)
                            if cluster_pre == 0:
                                cluster.partial_fit(X_train_norm, y_train_obv, X_train_weight)
                                cluster_pre = 1
                            cl_train, cl_c1_refine, use_cluster_train = \
                                cluster.compute_CLs(X_train_norm, y_train_obv)
                            # update classifier
                            classifier.partial_fit(X_train, y_train_obv, cl_train, label_val, X_train_weight)
                            # update micro-cluster
                            cluster.partial_fit(X_train_norm, y_train_obv, X_train_weight)
                            cluster.revise_cluster_info(X_train_norm, y_train_obv, cl_train)
                            # assign
                            cl_train_lst.extend(cl_train.tolist())
                            use_cluster_lst.extend(use_cluster_train.tolist())
                        else:
                            raise Exception("Undefined classifier with clf_name=%s." % clf_name)

                    prev_test_time = test_time[tt]  # update VIP
                else:
                    target_idx[tt] = False
                    test_y_pre[tt] = -1
                    if new_1data.ndim == 1:  # debug
                        new_1data = new_1data.reshape((1, -1))
                    if new_1data[0, data_ind_reset.id_y] == 0:
                        new_1data[0, data_ind_reset.id_vl] = np.inf
                    # set data_buffer, (ts, XX, vl)
                    if data_buffer is None:  # initialize
                        data_buffer = new_1data
                    else:
                        data_buffer = np.vstack((data_buffer, new_1data))
            test_time = test_time[target_idx]
            test_y_tru = test_y_tru[target_idx]
            test_y_pre = test_y_pre[target_idx]
            # return 1: rslt_test ~ (test_time, y_true, y_pred)
            rslt_test = np.vstack((test_time, test_y_tru, test_y_pre)).T
            np.savetxt(flnm_test, rslt_test, fmt='%d\t %d\t %d',
                       header="%test_time, yy, y_pre) ")

        """performance evaluation"""
        # pf eval throughout test steps
        test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
        pfs_tt_dct = uti_eval_pfs(test_y_tru, test_y_pre)

        # assign
        if ss == 0:  # init
            n_row, n_col = pfs_tt_dct["gmean_tt"].shape[0], len(seed_lst)
            cl_rmse, gmean_tt_ss = np.empty(n_col), np.empty((n_row, n_col))
            r1_tt_ss, r0_tt_ss = np.copy(gmean_tt_ss), np.copy(gmean_tt_ss)
            mcc_tt_ss = np.copy(gmean_tt_ss)
        gmean_tt_ss[:, ss], r1_tt_ss[:, ss], r0_tt_ss[:, ss], mcc_tt_ss[:, ss] = \
            pfs_tt_dct["gmean_tt"], pfs_tt_dct["recall1_tt"], pfs_tt_dct["recall0_tt"], pfs_tt_dct["mcc_tt"]

    """ave pf across seeds"""
    gmean_tt_ave_ss = np.nanmean(gmean_tt_ss, axis=1)
    r1_tt_ave_ss = np.nanmean(r1_tt_ss, axis=1)
    r0_tt_ave_ss = np.nanmean(r0_tt_ss, axis=1)
    return np.nanmean(gmean_tt_ave_ss), r1_tt_ave_ss, r0_tt_ave_ss


def para_classifiers_online(clf_name, data_ptrn, nb_repeat, wait_days, nb_pre,
                            cluster_trained, project_id):
    """
    This method is used to do parameter tuning of the JIT-SDP models, including OOB, ODaSC and PBSA.

    Args:
        clf_name (string): The name of base JIT-SDP model and the CP method.
        data_ptrn (list): The data stream used for parameter tuning.
        nb_repeat (int): The number of runs for parameter tuning.
        wait_days (int): The waiting time in online JIT-SDP.
        nb_pre (int): The number of WP data used to do parameter tuning, including the CP data before them.
        cluster_trained (object): The pretrained Denstream cluster used for ODaSC.
        project_id (int): The index of target project (WP).

    Returns:
        best_n_tree (int): A parameter of OOB, ODaSC and PBSA.
        best_theta_imb (float): A parameter of OOB, ODaSC and PBSA.
        best_theta_cl (float): A parameter of ODaSC.
        best_p (float): A parameter of PBSA.
        best_m (float): A parameter of PBSA.
        best_th (float): A parameter of PBSA.
    """
    # para tuning
    n_tree_lst = [5, 10, 15, 20, 30]  # 5
    theta_imb_lst = [0.9, 0.95, 0.99, 0.999]  # 4
    theta_cl_lst = [0.8, 0.9]  # 2
    p_lst = [0.15, 0.25, 0.35]
    m_lst = [1.5, 2.0, 2.7182]
    th_lst = [0.2, 0.3, 0.4, 0.5, 0.6]
    print("\nauto para-%s ..." % clf_name + str(project_id))
    semi_para = True
    nb_test = len(data_ptrn)

    seed_list = range(nb_repeat)
    best_gmean = 0
    best_n_tree = 10
    best_theta_imb = 0.9
    best_theta_cl = 0.8
    best_p = 0.25
    best_m = 1.5
    best_th = 0.3
    if clf_name == "oob":
        para_list = itertools.product(n_tree_lst, theta_imb_lst)
        for n_tree, theta_imb in para_list:
            theta_cl, p, m, th = -1, -1, -1, -1
            gmean_tt_ave_ss, r1_tt_ave_ss, r0_tt_ave_ss = \
                para_classifiers_online_run(data_ptrn, clf_name, nb_pre, nb_test, wait_days, seed_list,
                                            n_tree, theta_imb, theta_cl, p, m, th, cluster_trained, project_id)
            if gmean_tt_ave_ss > best_gmean:
                best_gmean = gmean_tt_ave_ss
                best_n_tree = n_tree
                best_theta_cl = theta_cl
                best_theta_imb = theta_imb
                best_p = p
                best_m = m
                best_th = th
    elif clf_name == "odasc":
        para_list = itertools.product(n_tree_lst, theta_imb_lst, theta_cl_lst)
        for n_tree, theta_imb, theta_cl in para_list:
            p, m, th = -1, -1, -1
            gmean_tt_ave_ss, r1_tt_ave_ss, r0_tt_ave_ss = \
                para_classifiers_online_run(data_ptrn, clf_name, nb_pre, nb_test, wait_days, seed_list,
                                            n_tree, theta_imb, theta_cl, p, m, th, cluster_trained, project_id)
            if gmean_tt_ave_ss > best_gmean:
                best_gmean = gmean_tt_ave_ss
                best_n_tree = n_tree
                best_theta_cl = theta_cl
                best_theta_imb = theta_imb
                best_p = p
                best_m = m
                best_th = th
    elif clf_name == "pbsa":
        para_list = itertools.product(n_tree_lst, p_lst, m_lst, th_lst)
        for n_tree, p, m, th in para_list:
            theta_cl = -1
            # pbsa use 0.99 as theta_imb
            theta_imb = 0.99
            gmean_tt_ave_ss, r1_tt_ave_ss, r0_tt_ave_ss = \
                para_classifiers_online_run(data_ptrn, clf_name, nb_pre, nb_test, wait_days, seed_list,
                                            n_tree, theta_imb, theta_cl, p, m, th, cluster_trained, project_id)
            if gmean_tt_ave_ss > best_gmean:
                best_gmean = gmean_tt_ave_ss
                best_n_tree = n_tree
                best_theta_cl = theta_cl
                best_theta_imb = theta_imb
                best_p = p
                best_m = m
                best_th = th
    elif "oob" in clf_name and clf_name != "oob":
        # load the parameters of oob
        dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/oob/"
        os.makedirs(dir_auto_para, exist_ok=True)
        auto_name = "%s-para-%dstep-%drun" % ("oob", nb_pre, nb_repeat) + ".pkl"
        exist_clf_para = os.path.exists(dir_auto_para + auto_name)

        if exist_clf_para:
            para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
            n_tree, theta_imb = \
                para_dict["n_trees"], para_dict["theta_imb"]
            theta_cl, p, m, th = invalid_val, invalid_val, invalid_val, invalid_val
            return n_tree, theta_imb, theta_cl, p, m, th
        else:
            raise Exception("There are no parameters when clf_name=oob")
    elif "odasc" in clf_name and clf_name != "odasc":
        # load the parameters of odasc
        dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/odasc/"
        os.makedirs(dir_auto_para, exist_ok=True)
        auto_name = "%s-para-%dstep-%drun" % ("odasc", nb_pre, nb_repeat) + ".pkl"
        exist_clf_para = os.path.exists(dir_auto_para + auto_name)

        if exist_clf_para:
            para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
            n_tree, theta_imb, theta_cl = \
                para_dict["n_trees"], para_dict["theta_imb"], para_dict["theta_cl"]
            p, m, th = invalid_val, invalid_val, invalid_val
            return n_tree, theta_imb, theta_cl, p, m, th
        else:
            raise Exception("There are no parameters when clf_name=odasc")
    elif "pbsa" in clf_name and clf_name != "pbsa":
        # load the parameters of pbsa
        dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/pbsa/"
        os.makedirs(dir_auto_para, exist_ok=True)
        auto_name = "%s-para-%dstep-%drun" % ("pbsa", nb_pre, nb_repeat) + ".pkl"
        exist_clf_para = os.path.exists(dir_auto_para + auto_name)

        if exist_clf_para:
            para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
            n_tree, theta_imb, p, m, th = \
                para_dict["n_trees"], para_dict["theta_imb"], para_dict["p"], para_dict["m"], para_dict["th"]
            theta_cl = invalid_val
            return n_tree, theta_imb, theta_cl, p, m, th
        else:
            raise Exception("There are no parameters when clf_name=pbsa")
    return best_n_tree, best_theta_imb, best_theta_cl, best_p, best_m, best_th


def para_classifiers_online_run(test_data_para, clf_name, nb_pre, nb_test, wait_days, seed_list,
                                n_tree, theta_imb, theta_cl, p, m, th, cluster_trained, project_id):
    """
    This method is related to parameter tuning of the JIT-SDP models, including OOB, ODaSC and PBSA.
    This method is used to running the JIT-SDP models with the given parameters to get the performance.

    Args:
        test_data_para (list): The data stream used for parameter tuning.
        clf_name (string): The name of base JIT-SDP model and the CP method.
        nb_pre (int): The number of WP data used to do parameter tuning, including the CP data before them.
        nb_test (int): Here is the length of test_data_para.
        wait_days (int): The waiting time in online JIT-SDP.
        seed_list (list): The list of random seeds used when running.
        n_tree (int): A parameter for OOB, ODaSC and PBSA.
        theta_imb (float): A parameter of OOB, ODaSC and PBSA.
        theta_cl (float): A parameter of ODaSC.
        p (float): A parameter of PBSA.
        m (float): A parameter of PBSA.
        th (float): A parameter of PBSA.
        cluster_trained (object): The pretrained Denstream cluster used for ODaSC.
        project_id (int): The index of target project (WP).

    Returns:
        average_gmean (float): The average gmean across time and seeds.
        average_recall1 (float): The average recall1 across time and seeds.
        average_recall0 (float): The average recall0 across time and seeds.
    """
    # data pre-train
    n_fea = 12
    seed_auto_tune = 255126
    data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                          n_fea=n_fea)
    norm_scaler_para = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
    norm_scaler_para.my_fit(test_data_para[data_ind_reset.id_X_np])

    our_clf_lst = ("our", "our_new")  # vip manually maintain

    """main parts across seeds"""
    for ss, seed in enumerate(seed_list):
        to_dir = uti_rslt_dir_base_para(clf_name, project_id, wait_days, n_tree, theta_imb, theta_cl, p, m, th)
        os.makedirs(to_dir, exist_ok=True)
        # analyze filenames in this dir:
        # find T that is larger than nb_data to save computational cost and load the results.
        exist_result, to_dir = uti_rslt_dir_analyze(to_dir, clf_name, nb_test, seed)
        if not exist_result:
            to_dir += "/T" + str(nb_test) + "/"
            os.makedirs(to_dir, exist_ok=True)
        # file_name-s
        flnm_test = "%s%s.rslt_test.s%d" % (to_dir, clf_name, seed)

        """load or compute"""
        if exist_result:
            rslt_test = np.loadtxt(flnm_test)
            # cutting the results if nb_test_actual < len(rslt_test)
        else:
            """set model"""
            if "oob" in clf_name:
                classifier = OzaBaggingClassifier_OOB(HoeffdingTreeClassifier(), n_tree, seed, theta_imb)
            elif "odasc" in clf_name:
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                cluster_used = copy.deepcopy(cluster_trained)
                cluster_used.theta_cl = theta_cl
                cluster_pre = 0
            elif "pbsa" in clf_name:
                classifier = OzaBaggingClassifier_PBSA(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, p, m, th)
            else:
                raise Exception("Undefined clf_name=%s." % clf_name)

            """[core] test-then-training process:
            at each test step, only one test data arrives, while maybe no or several training data become available
            """
            nb_test_act = nb_test
            test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
            cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
            cl_train_lst, use_cluster_lst = [], []

            prev_test_time, data_buffer, nb_train_data = test_data_para[0, data_ind_reset.id_time], None, 0
            for tt in range(nb_test_act):
                # get the test data
                test_step = tt
                new_1data = test_data_para[test_step, :].reshape((1, -1))
                test_X = new_1data[data_ind_reset.id_X_np]
                test_time[tt] = new_1data[:, data_ind_reset.id_time]
                test_y_tru[tt] = new_1data[:, data_ind_reset.id_y]

                """test: predict with classifiers"""
                test_y_pre[tt] = classifier.predict(test_X)[0]
                """then train: update classifiers and DenStream given new labelled training data"""
                X_train = test_X * np.ones(1)
                y_train_obv = test_y_tru[tt] * np.ones(1)
                use_time_train = test_time[tt] * np.ones(1)
                if clf_name == "oza" or "oob" in clf_name:
                    classifier.partial_fit(X_train, y_train_obv, label_val)
                    # assign
                    cl_train_lst.extend(invalid_val * np.ones(y_train_obv.shape))
                    use_cluster_lst = cl_train_lst
                elif "pbsa" in clf_name:
                    new_train_clean = []
                    new_train_defect = []
                    if test_y_tru[tt] == 1:
                        new_train_defect = X_train
                    elif test_y_tru[tt] == 0:
                        new_train_clean = X_train
                    classifier.pbsa_flow(X_train, y_train_obv, tt, new_train_clean,
                                         new_train_defect, data_ind_reset, label_val)
                    # assign
                    cl_train_lst.extend(invalid_val * np.ones(y_train_obv.shape))
                    use_cluster_lst = cl_train_lst
                elif "odasc" in clf_name:
                    X_train_norm = norm_scaler_para.my_transform(X_train)
                    if cluster_pre == 0:
                        cluster_used.partial_fit(X_train_norm, y_train_obv)
                        cluster_pre = 1
                    cl_train, cl_c1_refine, use_cluster_train = \
                        cluster_used.compute_CLs(X_train_norm, y_train_obv)
                    # update classifier
                    classifier.partial_fit(X_train, y_train_obv, cl_train, label_val)
                    # update micro-cluster
                    cluster_used.partial_fit(X_train_norm, y_train_obv)
                    cluster_used.revise_cluster_info(X_train_norm, y_train_obv, cl_train)
                    # assign
                    cl_train_lst.extend(cl_train.tolist())
                    use_cluster_lst.extend(use_cluster_train.tolist())
                else:
                    raise Exception("Undefined classifier with clf_name=%s." % clf_name)

            # return 1: rslt_test ~ (test_time, y_true, y_pred)
            rslt_test = np.vstack((test_time, test_y_tru, test_y_pre)).T
            np.savetxt(flnm_test, rslt_test, fmt='%d\t %d\t %d',
                       header="%test_time, yy, y_pre) ")

        """performance evaluation"""
        # pf eval throughout test steps
        test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
        pfs_tt_dct = uti_eval_pfs(test_y_tru, test_y_pre)

        # assign
        if ss == 0:  # init
            n_row, n_col = pfs_tt_dct["gmean_tt"].shape[0], len(seed_list)
            cl_rmse, gmean_tt_ss = np.empty(n_col), np.empty((n_row, n_col))
            r1_tt_ss, r0_tt_ss = np.copy(gmean_tt_ss), np.copy(gmean_tt_ss)
            mcc_tt_ss = np.copy(gmean_tt_ss)
        gmean_tt_ss[:, ss], r1_tt_ss[:, ss], r0_tt_ss[:, ss], mcc_tt_ss[:, ss] = \
            pfs_tt_dct["gmean_tt"], pfs_tt_dct["recall1_tt"], pfs_tt_dct["recall0_tt"], pfs_tt_dct["mcc_tt"]

    """ave pf across seeds"""
    gmean_tt_ave_ss = np.nanmean(gmean_tt_ss, axis=1)
    r1_tt_ave_ss = np.nanmean(r1_tt_ss, axis=1)
    r0_tt_ave_ss = np.nanmean(r0_tt_ss, axis=1)
    return np.nanmean(gmean_tt_ave_ss), r1_tt_ave_ss, r0_tt_ave_ss


def para_filtering_online(test_data_all, wait_days, norm_scaler, project_id, nb_para_tune, nb_test, data_ind_reset,
                          nb_repeat, clf, actual_WP_para_tune):
    """
    This method is used to do parameter tuning of Filtering (a state-of-the-art online CP method).

    Args:
        test_data_all (list): The whole data stream.
        wait_days (int): The waiting time in online JIT-SDP.
        norm_scaler (object): An object used to normalize the data.
        project_id (int): The index of target project (WP).
        nb_para_tune (int): The number of WP data used to do parameter tuning, including the CP data before them.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        data_ind_reset (object): An object to find the index of JIT-SDP features in data.
        nb_repeat (int): The number of runs for parameter tuning.
        clf (string): The name of base JIT-SDP model and the CP method.
        actual_WP_para_tune (int): The number of WP data used to do parameter tuning.

    Returns:
        best_window_size (int): A parameter of Filtering.
        best_K (int): A parameter of Filtering.
        best_max_dist (float): A parameter of Filtering.
        best_discard_size (int): A parameter of Filtering.
    """
    # para tuning
    window_size = [500, 600, 700, 1000]
    K = [5, 50, 100, 200]
    max_dist = [0.6, 0.7, 0.8]
    discard_size = [500, 1000]

    seed_list = range(nb_repeat)
    best_window_size = -1
    best_K = -1
    best_max_dist = -1
    best_discard_size = -1
    best_gmean = 0
    para_list = itertools.product(window_size, K, max_dist, discard_size)

    for window_size, K, max_dist, discard_size in para_list:
        # todo 20221219
        filtering_test_data_all, filtering_target_idx, filtering_nb_pre, filtering_nb_test_WP, filtering_nb_para_tune = \
            get_filtering_data(test_data_all, wait_days, norm_scaler, project_id, nb_para_tune, nb_test, data_ind_reset,
                               window_size, K, max_dist, discard_size, True)
        gmean_tt_ave_ss, r1_tt_ave_ss, r0_tt_ave_ss = \
            para_filtering_online_run(filtering_test_data_all, filtering_target_idx, clf, filtering_nb_para_tune,
                                      data_ind_reset, norm_scaler, project_id, filtering_nb_pre,
                                      filtering_nb_test_WP, range(nb_repeat), wait_days, window_size, K, max_dist,
                                      discard_size,
                                      actual_WP_para_tune)
        if gmean_tt_ave_ss >= best_gmean:
            best_gmean = gmean_tt_ave_ss
            best_window_size = window_size
            best_K = K
            best_max_dist = max_dist
            best_discard_size = discard_size
        elif np.isnan(gmean_tt_ave_ss):
            best_window_size = 500
            best_K = 5
            best_max_dist = 0.8
            best_discard_size = 500
            break
    return best_window_size, best_K, best_max_dist, best_discard_size


def para_crops_online_run(cpps_test_data_all, target_idx, clf_name, cpps_nb_para_tune,
                          data_ind_reset, norm_scaler, project_id, cpps_nb_pre,
                          cpps_nb_test_WP, seed_lst, wait_days, window_size, update_period, select_threshold,
                          actual_WP_para_tune, cluster):
    """
    This method is related to parameter tuning of the proposed CroPS.
    This method is used to running the CroPS with the given parameters based on JIT-SDP models to get the performance.

    Args:
        cpps_test_data_all (list): The whole data stream.
        target_idx (int): The index of WP.
        clf_name (string): The name of base JIT-SDP model and the CP method.
        cpps_nb_para_tune (int): The number of WP data used to do parameter tuning, including the CP data before them.
        data_ind_reset (object): An object to find the index of JIT-SDP features in data.
        norm_scaler (object): An object used to normalize the data.
        project_id (int): The index of target project (WP).
        cpps_nb_pre (int): The number of WP data used to do parameter tuning, including the CP data before them.
        cpps_nb_test_WP (int): The index of the last WP data used for parameter tuning in cpps_test_data_all.
        seed_lst (list): The list of random seeds used when running.
        wait_days (int): The waiting time in online JIT-SDP.
        window_size (int): A parameter of CroPS.
        update_period (int): A parameter of CroPS.
        select_threshold (float): A parameter of CroPS.
        actual_WP_para_tune (int): The number of WP data used to do parameter tuning.
        cluster (object): The pretrained Denstream cluster used for ODaSC.

    Returns:
        average_gmean (float): The average gmean across time and seeds.
        average_recall1 (float): The average recall1 across time and seeds.
        average_recall0 (float): The average recall0 across time and seeds.
    """
    data_ptrn = cpps_test_data_all[:cpps_nb_para_tune]
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]

    """para-auto DenStream~(lambd, eps, beta, mu)"""
    if "odasc" in clf_name:
        X_ptrn_norm = norm_scaler.my_transform(X_ptrn)
        auto_denStream = False
        if auto_denStream:
            eps, mu, beta, lambd = para_denStream(X_ptrn_norm, y_ptrn, nb_repeat=10)
        else:
            eps, mu, beta, lambd = 1.47, 1.57, 0.78, 0.26
            # eps, mu, beta, lambd = 2.09, 2.20, 0.74, 0.125

    """para-auto classifiers~(n_tree, theta_imb, theta_cl)"""
    nb_run = 5  # 30 in systematic exp
    temp = str.split(clf_name, "_")
    dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/" + temp[0] + "/"
    auto_name = "%s-para-%dstep-%drun" % (temp[0], cpps_nb_pre, nb_run) + ".pkl"
    exist_clf_para = os.path.exists(dir_auto_para + auto_name)
    # if exist_clf_para and not just_run:
    if exist_clf_para:
        para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
        n_tree, theta_imb, theta_cl, p, m, th = \
            para_dict["n_trees"], para_dict["theta_imb"], para_dict["theta_cl"], para_dict["p"], \
                para_dict["m"], para_dict["th"]
    else:
        raise Exception("There are no parameters when clf_name=" + temp[0])

    # update DenStream para
    if "odasc" in clf_name:
        cluster.theta_cl = theta_cl

    """main parts across seeds"""
    for ss, seed in enumerate(seed_lst):
        to_dir = uti_rslt_dir_cpps_para(clf_name, project_id, wait_days, window_size, update_period, select_threshold)
        os.makedirs(to_dir, exist_ok=True)
        # analyze filenames in this dir:
        # find T that is larger than nb_data to save computational cost and load the results.
        exist_result, to_dir = uti_rslt_dir_analyze(to_dir, clf_name, cpps_nb_test_WP, seed)
        if not exist_result:
            to_dir += "/T" + str(cpps_nb_test_WP) + "/"
            os.makedirs(to_dir, exist_ok=True)
        # file_name-s
        flnm_test = "%s%s.rslt_test.s%d" % (to_dir, clf_name, seed)

        """load or compute"""
        if exist_result:
            rslt_test = np.loadtxt(flnm_test)
            rslt_test = rslt_test[:actual_WP_para_tune]
            # cutting the results if nb_test_actual < len(rslt_test)
        else:
            """pre-train classifier"""
            if "oob" in clf_name:
                classifier = OzaBaggingClassifier_OOB(HoeffdingTreeClassifier(), n_tree, seed, theta_imb)
                if with_pretrain == 1:
                    classifier.partial_fit(X_ptrn, y_ptrn, label_val)
            elif "odasc" in clf_name:
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                if with_pretrain == 1:
                    cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)
                    classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn, label_val)
                    cluster_pre = 1
                else:
                    cluster = DenStream(theta_cl=theta_cl, lambd=lambd, eps=eps, beta=beta, mu=mu)
                    cluster_pre = 0
            elif "pbsa" in clf_name:
                classifier = OzaBaggingClassifier_PBSA(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, p, m, th)
                if with_pretrain == 1:
                    y_ptrn_pre = classifier.predict(X_ptrn)
                    classifier.train_model(X_ptrn, y_ptrn, label_val)
            else:
                raise Exception("Undefined clf_name=%s." % clf_name)

            """[core] test-then-training process:
            at each test step, only one test data arrives, while maybe no or several training data become available
            """
            if with_pretrain == 1:
                nb_test_act = cpps_nb_test_WP - cpps_nb_para_tune
            else:
                nb_test_act = cpps_nb_test_WP
            test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
            cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
            cl_train_lst, use_cluster_lst = [], []

            if with_pretrain == 1:
                prev_test_time, data_buffer, nb_train_data = data_ptrn[-1, data_ind_reset.id_time], None, 0  # vip
            else:
                prev_test_time, data_buffer, nb_train_data = cpps_test_data_all[
                    0, data_ind_reset.id_time], None, 0  # vip
            if "cpps" in clf_name:
                temp_clf = clf_name.split("_")
                # cpps_w_size = int(temp_clf[2])
                project_window, selected_project = sbp_initial(range(23), window_size)
                use_sbp = 1
                selected_features = []
                if len(temp_clf) > 3 and temp_clf[3] == "fs":
                    print("use selected features")
                    selected_features = load_selected_features(temp_clf[0], project_id)
                else:
                    for feature_id in range(17):
                        selected_features.append(feature_id)
                n_commit = np.zeros(len(project_window))
                combine_way = "L1"
            else:
                use_sbp = 0
            cp_weight = np.ones(23)
            selected_project = [project_id]
            selected_project = np.array(selected_project)

            for tt in range(nb_test_act):
                # get the test data
                if with_pretrain == 1:
                    test_step = tt + cpps_nb_para_tune
                else:
                    test_step = tt
                new_1data = cpps_test_data_all[test_step, :].reshape((1, -1))
                test_X = new_1data[data_ind_reset.id_X_np]
                test_time[tt] = new_1data[:, data_ind_reset.id_time]
                test_y_tru[tt] = new_1data[:, data_ind_reset.id_y]

                if use_sbp == 1:
                    if n_commit[project_id] % update_period == 0 and new_1data[
                        0, data_ind_reset.id_target] == project_id:
                        cp_weight = update_cp_weight(project_id, project_window, data_ind_reset, n_commit, combine_way,
                                                     selected_features)
                        selected_project = []
                        for index_project in range(23):
                            if cp_weight[index_project] >= select_threshold:
                                selected_project.append(index_project)
                        selected_project = np.array(selected_project)
                        # print(clf_name + " " + project_name + " ts:" + str(
                        #     n_commit[project_id]) + " cp_weight:" + str(cp_weight))
                        # print(selected_project)

                if new_1data[0, data_ind_reset.id_target] == project_id:
                    target_idx[tt] = True
                    n_commit[project_id] = n_commit[project_id] + 1
                    """test: predict with classifiers"""
                    test_y_pre[tt] = classifier.predict(test_X)[0]

                    """get the new train data batch"""
                    data_buffer, new_train_defect, new_train_clean, new_train_unlabeled = \
                        set_train_stream(prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer,
                                         wait_days)

                    if use_sbp == 0:
                        # note the order (clean, defect)
                        cmt_time_train = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_time], new_train_defect[:, data_ind_reset.id_time]))
                        use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                        X_train = np.concatenate(
                            (new_train_clean[data_ind_reset.id_X_np], new_train_defect[data_ind_reset.id_X_np]))
                        y_train_obv = np.concatenate(
                            (np.zeros(new_train_clean.shape[0]), np.ones(new_train_defect.shape[0])))
                        y_train_tru = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_y], new_train_defect[:, data_ind_reset.id_y]))
                        y_train_target = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_target],
                             new_train_defect[:, data_ind_reset.id_target]))
                        X_train_weight = np.ones(len(y_train_target))
                        nb_train_data += y_train_obv.shape[0]
                    elif use_sbp == 1:
                        for each in new_train_clean:
                            each[data_ind_reset.id_y] = 0
                            project_window[int(each[data_ind_reset.id_target])].append(each)
                        for each in new_train_defect:
                            each[data_ind_reset.id_y] = 1
                            project_window[int(each[data_ind_reset.id_target])].append(each)

                        stay_clean = np.in1d(new_train_clean[:, data_ind_reset.id_target], selected_project)
                        stay_defect = np.in1d(new_train_defect[:, data_ind_reset.id_target], selected_project)
                        new_train_clean = new_train_clean[stay_clean]
                        new_train_defect = new_train_defect[stay_defect]

                        # note the order (clean, defect)
                        cmt_time_train = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_time], new_train_defect[:, data_ind_reset.id_time]))
                        use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                        X_train = np.concatenate(
                            (new_train_clean[data_ind_reset.id_X_np], new_train_defect[data_ind_reset.id_X_np]))
                        y_train_obv = np.concatenate(
                            (np.zeros(new_train_clean.shape[0]), np.ones(new_train_defect.shape[0])))
                        y_train_tru = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_y], new_train_defect[:, data_ind_reset.id_y]))
                        y_train_target = np.concatenate(
                            (new_train_clean[:, data_ind_reset.id_target],
                             new_train_defect[:, data_ind_reset.id_target]))
                        X_train_weight = np.ones(len(y_train_target))
                        # X_train_weight = y_train_target.copy()
                        # for X_i in range(len(X_train_weight)):
                        #     X_i_target = X_train_weight[X_i]
                        #     X_train_weight[X_i] = cp_weight[int(X_i_target)]
                        nb_train_data += y_train_obv.shape[0]

                    # assign
                    cmt_time_train_lst.extend(cmt_time_train.tolist())
                    use_time_train_lst.extend(use_time_train.tolist())
                    y_train_obv_lst.extend(y_train_obv.tolist())
                    y_train_tru_lst.extend(y_train_tru.tolist())

                    """then train: update classifiers and DenStream given new labelled training data"""
                    if y_train_obv.shape[0] > 0:
                        if "oob" in clf_name:
                            classifier.partial_fit(X_train, y_train_obv, label_val, X_train_weight)
                            # assign
                            cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                            use_cluster_lst = cl_train_lst
                        elif "pbsa" in clf_name:
                            classifier.pbsa_flow(X_train, y_train_obv, tt, new_train_unlabeled[data_ind_reset.id_X_np],
                                                 new_train_defect, data_ind_reset, label_val, X_train_weight)
                            # assign
                            cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                            use_cluster_lst = cl_train_lst
                        elif "odasc" in clf_name:
                            X_train_norm = norm_scaler.my_transform(X_train)
                            if cluster_pre == 0:
                                cluster.partial_fit(X_train_norm, y_train_obv, X_train_weight)
                                cluster_pre = 1
                            cl_train, cl_c1_refine, use_cluster_train = \
                                cluster.compute_CLs(X_train_norm, y_train_obv)
                            # update classifier
                            classifier.partial_fit(X_train, y_train_obv, cl_train, label_val, X_train_weight)
                            # update micro-cluster
                            cluster.partial_fit(X_train_norm, y_train_obv, X_train_weight)
                            cluster.revise_cluster_info(X_train_norm, y_train_obv, cl_train)
                            # assign
                            cl_train_lst.extend(cl_train.tolist())
                            use_cluster_lst.extend(use_cluster_train.tolist())
                        else:
                            raise Exception("Undefined classifier with clf_name=%s." % clf_name)

                    prev_test_time = test_time[tt]  # update VIP
                else:
                    target_idx[tt] = False
                    test_y_pre[tt] = -1
                    if use_sbp == 1:
                        this_project = int(new_1data[0, data_ind_reset.id_target])
                        n_commit[this_project] = n_commit[this_project] + 1
                    if new_1data.ndim == 1:  # debug
                        new_1data = new_1data.reshape((1, -1))
                    if new_1data[0, data_ind_reset.id_y] == 0:
                        new_1data[0, data_ind_reset.id_vl] = np.inf
                    # set data_buffer, (ts, XX, vl)
                    if data_buffer is None:  # initialize
                        data_buffer = new_1data
                    else:
                        data_buffer = np.vstack((data_buffer, new_1data))
            test_time = test_time[target_idx]
            test_y_tru = test_y_tru[target_idx]
            test_y_pre = test_y_pre[target_idx]
            # return 1: rslt_test ~ (test_time, y_true, y_pred)
            rslt_test = np.vstack((test_time, test_y_tru, test_y_pre)).T
            np.savetxt(flnm_test, rslt_test, fmt='%d\t %d\t %d',
                       header="%test_time, yy, y_pre) ")

        """performance evaluation"""
        # pf eval throughout test steps
        test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
        pfs_tt_dct = uti_eval_pfs(test_y_tru, test_y_pre)

        # assign
        if ss == 0:  # init
            n_row, n_col = pfs_tt_dct["gmean_tt"].shape[0], len(seed_lst)
            cl_rmse, gmean_tt_ss = np.empty(n_col), np.empty((n_row, n_col))
            r1_tt_ss, r0_tt_ss = np.copy(gmean_tt_ss), np.copy(gmean_tt_ss)
            mcc_tt_ss = np.copy(gmean_tt_ss)
        gmean_tt_ss[:, ss], r1_tt_ss[:, ss], r0_tt_ss[:, ss], mcc_tt_ss[:, ss] = \
            pfs_tt_dct["gmean_tt"], pfs_tt_dct["recall1_tt"], pfs_tt_dct["recall0_tt"], pfs_tt_dct["mcc_tt"]

    """ave pf across seeds"""
    gmean_tt_ave_ss = np.nanmean(gmean_tt_ss, axis=1)
    r1_tt_ave_ss = np.nanmean(r1_tt_ss, axis=1)
    r0_tt_ave_ss = np.nanmean(r0_tt_ss, axis=1)
    return np.nanmean(gmean_tt_ave_ss), r1_tt_ave_ss, r0_tt_ave_ss


def para_crops_online(test_data_all, wait_days, norm_scaler, project_id, nb_para_tune, nb_test, data_ind_reset,
                      nb_repeat, clf, actual_WP_para_tune):
    """
    This method is used to do parameter tuning of the proposed CroPS.

    Args:
        test_data_all (list): The whole data stream.
        wait_days (int): The waiting time in online JIT-SDP.
        norm_scaler (object): An object used to normalize the data.
        project_id (int): The index of target project (WP).
        nb_para_tune (int): The number of WP data used to do parameter tuning, including the CP data before them.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        data_ind_reset (object): An object to find the index of JIT-SDP features in data.
        nb_repeat (int): The number of runs for parameter tuning.
        clf (string): The name of base JIT-SDP model and the CP method.
        actual_WP_para_tune (int): The number of WP data used to do parameter tuning.

    Returns:
        best_window_size (int): A parameter of CroPS.
        best_update_period (int): A parameter of CroPS.
        best_select_threshold (float): A parameter of CroPS.
    """
    # para tuning
    window_size = [500, 700, 1000]
    update_period = [100, 200, 500]
    select_threshold = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]

    seed_list = range(nb_repeat)
    best_window_size = -1
    best_update_period = -1
    best_select_threshold = -1
    best_gmean = 0
    para_list = itertools.product(window_size, update_period, select_threshold)

    # find the index that contains nb_test target data
    count = 0
    cpps_nb_pre = nb_para_tune
    for i in range(len(test_data_all)):
        if test_data_all[i][-1] == project_id:
            count = count + 1
        if count == cpps_nb_pre:
            cpps_nb_para_tune = i
            cpps_nb_test_WP = i

    cpps_target_idx = test_data_all[:, data_ind_reset.id_target] == project_id
    cpps_target_idx_para = cpps_target_idx[:nb_para_tune]
    if with_pretrain == 1:
        cpps_target_idx = cpps_target_idx[nb_para_tune + 1:cpps_nb_test_WP + 1]
    else:
        cpps_target_idx = cpps_target_idx[1:cpps_nb_test_WP + 1]
    cpps_data_ptrn = test_data_all[:cpps_nb_para_tune]
    cpps_X_ptrn, cpps_y_ptrn = cpps_data_ptrn[data_ind_reset.id_X_np], cpps_data_ptrn[:, data_ind_reset.id_y]

    """para-auto DenStream~(lambd, eps, beta, mu)"""
    if "odasc" in clf:
        cpps_X_ptrn_norm = norm_scaler.my_transform(cpps_X_ptrn)
        auto_denStream = False
        if auto_denStream:
            eps, mu, beta, lambd = para_denStream(cpps_X_ptrn_norm, cpps_y_ptrn, nb_repeat=10)
        else:
            eps, mu, beta, lambd = 1.47, 1.57, 0.78, 0.26
            # eps, mu, beta, lambd = 2.09, 2.20, 0.74, 0.125

    """pre-train DenStream"""
    if "odasc" in clf:
        cluster = DenStream(theta_cl=None, lambd=lambd, eps=eps, beta=beta, mu=mu)
        cluster.partial_fit(cpps_X_ptrn_norm, cpps_y_ptrn)
    else:
        cluster = 0
    if "ensemble" in clf:
        # load the parameters of cpps
        temp_clf_name = clf[:-9]
        dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/" + temp_clf_name + "/"
        os.makedirs(dir_auto_para, exist_ok=True)
        auto_name = "%s-cpps-para-%dstep-%drun" % (temp_clf_name, nb_para_tune, nb_repeat) + ".pkl"
        exist_clf_para = os.path.exists(dir_auto_para + auto_name)

        if exist_clf_para:
            para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
            window_size, update_period, select_threshold = \
                para_dict["window_size"], para_dict["update_period"], para_dict["select_threshold"]
            return window_size, update_period, select_threshold
    if "random" in clf:
        # load the parameters of cpps
        temp_clf_name = clf[:-7]
        dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/" + temp_clf_name + "/"
        os.makedirs(dir_auto_para, exist_ok=True)
        auto_name = "%s-cpps-para-%dstep-%drun" % (temp_clf_name, nb_para_tune, nb_repeat) + ".pkl"
        exist_clf_para = os.path.exists(dir_auto_para + auto_name)

        if exist_clf_para:
            para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
            window_size, update_period, select_threshold = \
                para_dict["window_size"], para_dict["update_period"], para_dict["select_threshold"]
            return window_size, update_period, select_threshold

    for window_size, update_period, select_threshold in para_list:
        gmean_tt_ave_ss, r1_tt_ave_ss, r0_tt_ave_ss = \
            para_crops_online_run(test_data_all, cpps_target_idx, clf, cpps_nb_para_tune,
                                  data_ind_reset, norm_scaler, project_id, cpps_nb_pre,
                                  cpps_nb_test_WP, range(nb_repeat), wait_days, window_size, update_period,
                                  select_threshold,
                                  actual_WP_para_tune, cluster)
        if gmean_tt_ave_ss >= best_gmean:
            best_gmean = gmean_tt_ave_ss
            best_window_size = window_size
            best_update_period = update_period
            best_select_threshold = select_threshold
        elif np.isnan(gmean_tt_ave_ss):
            best_window_size = 500
            best_update_period = 100
            best_select_threshold = 0.1
            break
    return best_window_size, best_update_period, best_select_threshold


def sdp_runs(clf_name="odasc", project_id=6, nb_para_tune=500, nb_test=5000, wait_days=15,
             seed_lst=range(20), verbose_int=0, pca_plot=True, just_run=False, load_result=False):
    """
    This method is the core part to running JIT-SDP models and CP method.
    This method is used to do the experiments of RQ2 and RQ3.

    Args:
        clf_name (string): The name of base JIT-SDP model and the CP method.
        project_id (int): The index of target project (WP).
        nb_para_tune (int): The number of WP data used to do parameter tuning.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        wait_days (int): The waiting time in online JIT-SDP.
        seed_lst (list): The list of random seeds used when running.
        verbose_int (int): A number to control the print of running information. "-1" means no print; a larger value
            means deeper and more detailed "print".
        pca_plot (boolean): A parameter to control whether plot the result.
        just_run (boolean): If True, this method will not load or save results, for safety reason.
        load_result (boolean): If True, this method will skip the Filtering part and will not save results to save time.
    """
    clf_name = clf_name.lower()
    if pca_plot:
        x_lim, y_lim = None, None
    project_name = data_id_2name(project_id)
    info_run = "%s: %s, wtt=%d, #seed=%d" % (clf_name, project_name, wait_days, len(seed_lst))
    # if just_run:  # revise the print level to the most detailed level
    #     verbose_int = 3

    """prepare test data stream"""
    report_nb_test = nb_test
    if clf_name == "oza" or clf_name == "oob" or clf_name == "odasc" or clf_name == "orb" or clf_name == "pbsa":
        test_stream = set_test_stream(project_name)
        test_stream.X = np.hstack(
            (test_stream.X, (np.ones(len(test_stream.X)) * project_id).reshape(len(test_stream.X), 1)))
        X_org = test_stream.X[class_data_ind_org().id_X_np]
        # convert fea14 to fea13 and the test data stream
        XX, use_data = real_data_preprocess(X_org)
        yy = test_stream.y[use_data]
        data_time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
        vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
        target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

        # handle negative nb_test
        n_data_all, n_fea = XX.shape[0], XX.shape[1]  # after fea conversion for jit-sdp
        assert n_fea == 12, "# transformed fea should be 13. Sth. is wrong."
        if nb_test < 0:
            nb_test += n_data_all
            if verbose_int >= 2:
                print("actual nb_test=%d" % nb_test)
        assert nb_para_tune < nb_test, "nb_pre=%d should be smaller than nb_data=%d" % (nb_para_tune, nb_test)

        # fea normalizer based on all test data used for DenStream
        norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
        norm_scaler.my_fit(XX)
        # print('std:', np.std(norm_scaler.my_transform(XX), axis=0))

        # prepare all test samples
        test_data_all = np.hstack((data_time, XX, vl, yy, target))  # col=4+12 ~ (time, fea12, vl, yy, target)
        data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                              n_fea=n_fea)

        ## use 10% to para
        # nb_para_tune = int(len(test_data_all) * 0.1)
        actual_WP_para_tune = nb_para_tune

        if nb_test == -1:
            nb_test_WP = len(test_data_all)
        else:
            nb_test_WP = nb_test
        nb_pre = nb_para_tune
        target_idx_para = 0
        target_idx = test_data_all[:, data_ind_reset.id_target] == project_id
    elif "aio" in clf_name or "cpps" in clf_name:
        test_stream = set_test_stream(project_name)
        test_stream.X = np.hstack(
            (test_stream.X, (np.ones(len(test_stream.X)) * project_id).reshape(len(test_stream.X), 1)))
        X_org = test_stream.X[class_data_ind_org().id_X_np]
        # convert fea14 to fea13 and the test data stream
        XX, use_data = real_data_preprocess(X_org)
        yy = test_stream.y[use_data]
        data_time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
        vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
        target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

        # handle negative nb_test
        n_data_all, n_fea = XX.shape[0], XX.shape[1]  # after fea conversion for jit-sdp
        assert n_fea == 12, "# transformed fea should be 13. Sth. is wrong."
        if nb_test < 0:
            nb_test += n_data_all
            if verbose_int >= 2:
                print("actual nb_test=%d" % nb_test)
        assert nb_para_tune < nb_test, "nb_pre=%d should be smaller than nb_data=%d" % (nb_para_tune, nb_test)

        norm_data = XX

        # prepare all test samples
        test_data_all = np.hstack((data_time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
        data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                              n_fea=n_fea)
        ## use 10% to para
        # nb_para_tune = int(len(test_data_all) * 0.1)
        actual_WP_para_tune = nb_para_tune

        """add cross project data"""
        for i in range(23):
            if i != project_id:
                project_name_cp = data_id_2name(i)
                test_stream = set_test_stream(project_name_cp)
                test_stream.X = np.hstack(
                    (test_stream.X, (np.ones(len(test_stream.X)) * i).reshape(len(test_stream.X), 1)))
                X_org = test_stream.X[class_data_ind_org().id_X_np]
                # convert fea14 to fea13 and the test data stream
                XX, use_data = real_data_preprocess(X_org)
                yy = test_stream.y[use_data]
                data_time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
                vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
                target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

                test_data_temp = np.hstack((data_time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
                test_data_all = np.vstack([test_data_all, test_data_temp])

                norm_data = np.vstack([norm_data, XX])

        idx = test_data_all[:, 0].argsort()
        test_data_all = test_data_all[idx]

        # fea normalizer based on all test data used for DenStream
        norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
        norm_scaler.my_fit(norm_data)

        # find the index that contains nb_test target data
        count = 0
        nb_pre = nb_para_tune
        for i in range(len(test_data_all)):
            if test_data_all[i][-1] == project_id:
                count = count + 1
                if nb_test == -1:
                    nb_test_WP = i
            if count == nb_pre:
                nb_para_tune = i
            if count == nb_test and nb_test != -1:
                nb_test_WP = i
                break

        target_idx = test_data_all[:, data_ind_reset.id_target] == project_id
        target_idx_para = target_idx[:nb_para_tune]
        if with_pretrain == 1:
            target_idx = target_idx[nb_para_tune + 1:nb_test_WP + 1]
        else:
            target_idx = target_idx[1:nb_test_WP + 1]

        # para tuning for cpps
        if "cpps" in clf_name:
            nb_run = 5
            dir_cpps_para = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/"
            os.makedirs(dir_cpps_para, exist_ok=True)
            auto_name = "%s-cpps-para-%dstep-%drun" % (clf_name, nb_pre, nb_run) + ".pkl"
            exist_clf_para = os.path.exists(dir_cpps_para + auto_name)
            if exist_clf_para:
                para_dict = pkl.load(open(dir_cpps_para + auto_name, 'rb'))
                cpps_window_size, update_period, select_threshold = \
                    para_dict["window_size"], para_dict["update_period"], para_dict["select_threshold"]
            else:
                cpps_window_size, update_period, select_threshold = para_crops_online(test_data_all, wait_days,
                                                                                      norm_scaler,
                                                                                      project_id, nb_pre, nb_test,
                                                                                      data_ind_reset, nb_run, clf_name,
                                                                                      actual_WP_para_tune)
                # save para_bst
                if not just_run:
                    para_dict = {"window_size": cpps_window_size, "update_period": update_period,
                                 "select_threshold": select_threshold}
                    with open(dir_cpps_para + auto_name, 'wb') as save_file:
                        pkl.dump(para_dict, save_file)
    elif "filtering" in clf_name:
        test_stream = set_test_stream(project_name)
        test_stream.X = np.hstack(
            (test_stream.X, (np.ones(len(test_stream.X)) * project_id).reshape(len(test_stream.X), 1)))
        X_org = test_stream.X[class_data_ind_org().id_X_np]
        # convert fea14 to fea13 and the test data stream
        XX, use_data = real_data_preprocess(X_org)
        yy = test_stream.y[use_data]
        data_time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
        vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
        target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

        # handle negative nb_test
        n_data_all, n_fea = XX.shape[0], XX.shape[1]  # after fea conversion for jit-sdp
        assert n_fea == 12, "# transformed fea should be 13. Sth. is wrong."
        if nb_test < 0:
            nb_test += n_data_all
            if verbose_int >= 2:
                print("actual nb_test=%d" % nb_test)
        assert nb_para_tune < nb_test, "nb_pre=%d should be smaller than nb_data=%d" % (nb_para_tune, nb_test)

        norm_data = XX

        # prepare all test samples
        test_data_all = np.hstack((data_time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
        data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                              n_fea=n_fea)
        ## use 10% to para
        # nb_para_tune = int(len(test_data_all) * 0.1)
        actual_WP_para_tune = nb_para_tune

        """add cross project data"""
        for i in range(23):
            if i != project_id:
                project_name_cp = data_id_2name(i)
                test_stream = set_test_stream(project_name_cp)
                test_stream.X = np.hstack(
                    (test_stream.X, (np.ones(len(test_stream.X)) * i).reshape(len(test_stream.X), 1)))
                X_org = test_stream.X[class_data_ind_org().id_X_np]
                # convert fea14 to fea13 and the test data stream
                XX, use_data = real_data_preprocess(X_org)
                yy = test_stream.y[use_data]
                data_time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
                vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
                target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

                test_data_temp = np.hstack((data_time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
                test_data_all = np.vstack([test_data_all, test_data_temp])

                norm_data = np.vstack([norm_data, XX])

        idx = test_data_all[:, 0].argsort()
        test_data_all = test_data_all[idx]

        # fea normalizer based on all test data used for DenStream
        norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
        norm_scaler.my_fit(norm_data)

        # para filtering
        nb_run = 5
        dir_filtering_para = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/"
        os.makedirs(dir_filtering_para, exist_ok=True)
        auto_name = "%s-filtering-para-%dstep-%drun" % (clf_name, nb_para_tune, nb_run) + ".pkl"
        exist_clf_para = os.path.exists(dir_filtering_para + auto_name)
        if exist_clf_para:
            para_dict = pkl.load(open(dir_filtering_para + auto_name, 'rb'))
            filtering_window_size, K, max_dist, discard_size = \
                para_dict["window_size"], para_dict["K"], para_dict["max_dist"], para_dict["discard_size"]
        else:
            filtering_window_size, K, max_dist, discard_size = para_filtering_online(test_data_all, wait_days,
                                                                                     norm_scaler,
                                                                                     project_id, nb_para_tune, nb_test,
                                                                                     data_ind_reset, nb_run, clf_name,
                                                                                     actual_WP_para_tune)
            # save para_bst
            if not just_run:
                para_dict = {"window_size": filtering_window_size, "K": K, "max_dist": max_dist,
                             "discard_size": discard_size}
                with open(dir_filtering_para + auto_name, 'wb') as save_file:
                    pkl.dump(para_dict, save_file)
        filtering_start_time = time.time()
        if not load_result:
            test_data_all, target_idx, nb_pre, nb_test_WP, nb_para_tune = \
                get_filtering_data(test_data_all, wait_days, norm_scaler, project_id, nb_para_tune, nb_test,
                                   data_ind_reset,
                                   filtering_window_size, K, max_dist, discard_size, False)
        else:
            nb_pre = nb_para_tune
        filtering_end_time = time.time()
        filtering_time = filtering_end_time - filtering_start_time

    if "ensemble" in clf_name or "random" in clf_name:
        ensemble_threshold = []
        ensemble_threshold.append(select_threshold)
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
        for each in thresholds:
            if each not in ensemble_threshold:
                ensemble_threshold.append(each)
    # if "group" in clf_name:
    #     ensemble_threshold = [select_threshold, 0, 0.1, 0.2, 0.3, 0.4, 0.5]

    data_ptrn = test_data_all[:nb_para_tune]
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]

    """para-auto DenStream~(lambd, eps, beta, mu)"""
    if "odasc" in clf_name:
        X_ptrn_norm = norm_scaler.my_transform(X_ptrn)
        auto_denStream = False
        if auto_denStream:
            eps, mu, beta, lambd = para_denStream(X_ptrn_norm, y_ptrn, nb_repeat=10)
        else:
            eps, mu, beta, lambd = 1.47, 1.57, 0.78, 0.26
            # eps, mu, beta, lambd = 2.09, 2.20, 0.74, 0.125

    """pre-train DenStream"""
    if "odasc" in clf_name:
        cluster = DenStream(theta_cl=None, lambd=lambd, eps=eps, beta=beta, mu=mu)
        cluster.partial_fit(X_ptrn_norm, y_ptrn)
    else:
        cluster = 0
    """para-auto classifiers~(n_tree, theta_imb, theta_cl)"""
    nb_run = 5  # 30 in systematic exp
    dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/"
    os.makedirs(dir_auto_para, exist_ok=True)
    auto_name = "%s-para-%dstep-%drun" % (clf_name, nb_pre, nb_run) + ".pkl"
    exist_clf_para = os.path.exists(dir_auto_para + auto_name)

    # if exist_clf_para and not just_run:
    if exist_clf_para:
        para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
        n_tree, theta_imb, theta_cl, p, m, th = \
            para_dict["n_trees"], para_dict["theta_imb"], para_dict["theta_cl"], para_dict["p"], \
                para_dict["m"], para_dict["th"]
    else:
        n_tree, theta_imb, theta_cl, p, m, th = para_classifiers_online(clf_name, data_ptrn, nb_run, wait_days, nb_pre,
                                                                        cluster, project_id)
        # save para_bst
        if not just_run:
            para_dict = {"n_trees": n_tree, "theta_imb": theta_imb, "theta_cl": theta_cl, "p": p,
                         "m": m, "th": th}
            with open(dir_auto_para + auto_name, 'wb') as save_file:
                pkl.dump(para_dict, save_file)

    if verbose_int >= 1:
        print("\n%s, best para:\nn_test=%d, n_tree=%d, theta_imb=%.3f, theta_cl=%.3f" % (
            info_run, nb_test, n_tree, theta_imb, theta_cl))

    # update DenStream para
    if "odasc" in clf_name:
        cluster.theta_cl = theta_cl

    """main parts across seeds"""
    rslt_test_ac_seed = []
    for ss, seed in enumerate(seed_lst):
        if "cpps" not in clf_name:
            cpps_window_size, update_period, select_threshold = 0, 0, 0
        if "filtering" not in clf_name:
            filtering_window_size, K, max_dist, discard_size = 0, 0, 0, 0
        to_dir = uti_rslt_dir(clf_name, project_id, wait_days, n_tree, theta_imb, theta_cl,
                              p, m, th, filtering_window_size, K, max_dist, discard_size,
                              cpps_window_size, update_period, select_threshold)
        os.makedirs(to_dir, exist_ok=True)
        # analyze filenames in this dir:
        # find T that is larger than nb_data to save computational cost and load the results.
        exist_result, to_dir = uti_rslt_dir_analyze(to_dir, clf_name, nb_test, seed)
        if not exist_result:
            to_dir += "/T" + str(nb_test) + "/"
            os.makedirs(to_dir, exist_ok=True)
        # file_name-s
        flnm_test = "%s%s.rslt_test.s%d" % (to_dir, clf_name, seed)
        flnm_train = "%s%s.rslt_train.s%d" % (to_dir, clf_name, seed)
        flnm_info = "%s%s.rslt_info.s%d" % (to_dir, clf_name, seed)
        flnm_ensemble = "%s%s.rslt_ensemble.s%d.txt" % (to_dir, clf_name, seed)

        """load or compute"""
        if exist_result:
            rslt_test = np.loadtxt(flnm_test)
            rslt_train = np.loadtxt(flnm_train)
            rslt_info = np.loadtxt(flnm_info)
            if nb_test != -1:
                rslt_test = rslt_test[:nb_test]
            # cutting the results if nb_test_actual < len(rslt_test)
            nb_test_act = nb_test - nb_para_tune
            rslt_test_ac_seed.append(rslt_test)
            # if len(rslt_test) > nb_test_act:
            #     rslt_test = rslt_test[:nb_test_act, :]
            #     rslt_train = rslt_train[:nb_test_act, :]
            #     rslt_on_imb = rslt_on_imb[:nb_test_act, :]
            # return 1: rslt_test~(test_time, y_true, y_pred), note: test_time is the commit_time
            # return 2: rslt_train~(commit_time, use_time, yy, y_obv, cl, use_cluster)
            # return 3: rslt_on_imb~(test_time, on_imb_c0, on_imb_c1)
        else:
            """pre-train classifier"""
            cluster_pre = 0
            if clf_name == "oza":
                classifier = OzaBaggingClassifier(HoeffdingTreeClassifier(), n_tree, seed)
                if with_pretrain == 1:
                    classifier.partial_fit(X_ptrn, y_ptrn, label_val)
            elif "oob" in clf_name:
                classifier = OzaBaggingClassifier_OOB(HoeffdingTreeClassifier(), n_tree, seed, theta_imb)
                if with_pretrain == 1:
                    classifier.partial_fit(X_ptrn, y_ptrn, label_val)
            elif "odasc" in clf_name:
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                if with_pretrain == 1:
                    cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)
                    classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn, label_val)
                    cluster_pre = 1
                else:
                    cluster = DenStream(theta_cl=theta_cl, lambd=lambd, eps=eps, beta=beta, mu=mu)
                    cluster_pre = 0
            elif "pbsa" in clf_name:
                classifier = OzaBaggingClassifier_PBSA(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, p, m, th)
                if with_pretrain == 1:
                    y_ptrn_pre = classifier.predict(X_ptrn)
                    classifier.train_model(X_ptrn, y_ptrn, label_val)
            else:
                raise Exception("Undefined clf_name=%s." % clf_name)

            if "ensemble" in clf_name or "random" in clf_name:
                ensemble_len = len(ensemble_threshold)
                classifiers = []
                clusters = []
                cluster_pres = []
                for nb_clf in range(ensemble_len):
                    classifiers.append(copy.deepcopy(classifier))
                    clusters.append(copy.deepcopy(cluster))
                    cluster_pres.append(copy.deepcopy(cluster_pre))
                ensemble_save = []
            if "group" in clf_name:
                ensemble_len = len(ensemble_threshold)
                classifiers = []
                clusters = []
                cluster_pres = []
                for nb_clf in range(ensemble_len):
                    classifiers.append(copy.deepcopy(classifier))
                    clusters.append(copy.deepcopy(cluster))
                    cluster_pres.append(copy.deepcopy(cluster_pre))
                ensemble_save = []

            """[core] test-then-training process:
            at each test step, only one test data arrives, while maybe no or several training data become available
            """
            running_start_time = time.time()
            if with_pretrain == 1:
                nb_test_act = nb_test_WP - nb_para_tune
            else:
                nb_test_act = nb_test_WP
            test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
            cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
            cl_train_lst, use_cluster_lst = [], []
            CP_positive_data = []
            CP_positive_size = 50
            pre_r1, pre_r2, pre_gmean = 0, 0, 0

            if with_pretrain == 1:
                prev_test_time, data_buffer, nb_train_data = data_ptrn[-1, data_ind_reset.id_time], None, 0  # vip
            else:
                prev_test_time, data_buffer, nb_train_data = test_data_all[0, data_ind_reset.id_time], None, 0  # vip
            notadd = 0
            # project_window, selected_project = sbp_initial(range(14), 500)
            # if clf_name == "oob_sbp_l1" or clf_name == "orb_sbp_l1" or clf_name == "odasc_sbp_l1" or clf_name == "pbsa_sbp_l1":
            #     project_window, selected_project = sbp_initial(range(23), 500)
            #     use_sbp = 1
            #     n_commit = np.zeros(len(project_window))
            #     combine_way = "L1"
            # elif clf_name == "oob_sbp_l2" or clf_name == "orb_sbp_l2" or clf_name == "odasc_sbp_l2" or clf_name == "pbsa_sbp_l2":
            #     project_window, selected_project = sbp_initial(range(23), 500)
            #     use_sbp = 1
            #     n_commit = np.zeros(len(project_window))
            #     combine_way = "L2"
            # else:
            #     use_sbp = 0
            if "cpps" in clf_name:
                temp_clf = clf_name.split("_")
                # cpps_w_size = int(temp_clf[2])
                project_window, selected_project = sbp_initial(range(23), cpps_window_size)
                use_sbp = 1
                selected_features = []
                if len(temp_clf) > 3 and temp_clf[3] == "fs":
                    print("use selected features")
                    selected_features = load_selected_features(temp_clf[0], project_id)
                else:
                    for feature_id in range(17):
                        selected_features.append(feature_id)
                n_commit = np.zeros(len(project_window))
                combine_way = "L1"
                if "random" in clf_name:
                    ensemble_selected_project = []
                    for nb_ensemble in range(len(ensemble_threshold)):
                        selected_project = []
                        for index_project in range(23):
                            if np.random.rand() >= 0.5:
                                selected_project.append(index_project)
                        selected_project = np.array(selected_project)
                        ensemble_selected_project.append(selected_project)
            else:
                use_sbp = 0
            cp_weight = np.ones(23)
            selected_project = [project_id]
            selected_project = np.array(selected_project)
            for tt in range(nb_test_act):
                # get the test data
                if with_pretrain == 1:
                    test_step = tt + nb_para_tune
                else:
                    test_step = tt
                new_1data = test_data_all[test_step, :].reshape((1, -1))
                test_X = new_1data[data_ind_reset.id_X_np]
                test_time[tt] = new_1data[:, data_ind_reset.id_time]
                test_y_tru[tt] = new_1data[:, data_ind_reset.id_y]

                if use_sbp == 1 and "random" not in clf_name:
                    # update selected project
                    if n_commit[project_id] % update_period == 0 and new_1data[
                        0, data_ind_reset.id_target] == project_id:
                        cp_weight = update_cp_weight(project_id, project_window, data_ind_reset, n_commit, combine_way,
                                                     selected_features)
                        if "ensemble" in clf_name:
                            ensemble_selected_project = []
                            for nb_ensemble in range(len(ensemble_threshold)):
                                selected_project = []
                                for index_project in range(23):
                                    if cp_weight[index_project] >= ensemble_threshold[nb_ensemble]:
                                        selected_project.append(index_project)
                                selected_project = np.array(selected_project)
                                ensemble_selected_project.append(selected_project)
                        elif "group" in clf_name:
                            ensemble_selected_project = []
                            for nb_ensemble in range(len(ensemble_threshold)):
                                if nb_ensemble == 0:
                                    selected_project = []
                                    for index_project in range(23):
                                        if cp_weight[index_project] >= ensemble_threshold[nb_ensemble]:
                                            selected_project.append(index_project)
                                else:
                                    selected_project = [project_id]
                                    if nb_ensemble != len(ensemble_threshold) - 1:
                                        for index_project in range(23):
                                            if ensemble_threshold[nb_ensemble] <= cp_weight[index_project] <= ensemble_threshold[nb_ensemble + 1]:
                                                selected_project.append(index_project)
                                    else:
                                        for index_project in range(23):
                                            if ensemble_threshold[nb_ensemble] <= cp_weight[index_project] <= 1:
                                                selected_project.append(index_project)
                                selected_project = np.array(selected_project)
                                ensemble_selected_project.append(selected_project)
                        else:
                            selected_project = []
                            for index_project in range(23):
                                if cp_weight[index_project] >= select_threshold:
                                    selected_project.append(index_project)
                            selected_project = np.array(selected_project)
                        # print(clf_name + " " + project_name + " ts:" + str(
                        #     n_commit[project_id]) + " cp_weight:" + str(cp_weight))
                        # print(selected_project)

                if new_1data[0, data_ind_reset.id_target] == project_id:
                    target_idx[tt] = True
                    if use_sbp == 1:
                        n_commit[project_id] = n_commit[project_id] + 1
                    if "ensemble" in clf_name or "random" in clf_name:
                        ensemble_temp_save = []
                        if len(ensemble_threshold) >= 1 and tt > nb_para_tune:
                            now_pre = test_y_pre[:tt]
                            if with_pretrain == 0:
                                start = 0
                            else:
                                start = nb_para_tune
                            now_data = test_data_all[start:test_step]
                            for each in now_data:
                                if each[data_ind_reset.id_y] == 0:
                                    each[data_ind_reset.id_vl] = np.inf
                            # 1) set train_data_defect and update data_buffer
                            is_defect = test_time[tt] > (now_data[:, data_ind_reset.id_time] + cvt_day2timestamp(
                                now_data[:, data_ind_reset.id_vl]))
                            now_pre_defect = now_pre[is_defect]
                            now_pre = now_pre[~is_defect]
                            now_obv_defect = np.ones(len(now_pre_defect))

                            # update data_buffer: pop out defect-inducing data
                            now_data = now_data[~is_defect, :]  # (time, 13-fea, vl, y)

                            # 2) set train_data_clean and update data_buffer
                            wait_days_clean = test_time[tt] > now_data[:, data_ind_reset.id_time] + cvt_day2timestamp(
                                wait_days)
                            now_pre_clean = now_pre[wait_days_clean]
                            now_obv_clean = np.zeros(len(now_pre_clean))
                            now_pre = np.concatenate((now_pre_clean, now_pre_defect))
                            now_obv = np.concatenate((now_obv_clean, now_obv_defect))
                            now_obv_real = []
                            now_pre_real = []
                            for nb_label in range(len(now_pre)):
                                if now_pre[nb_label] != -1:
                                    now_obv_real.append(now_obv[nb_label])
                                    now_pre_real.append(now_pre[nb_label])
                            now_r1 = recall_score(now_obv_real, now_pre_real, pos_label=1)
                            now_r0 = recall_score(now_obv_real, now_pre_real, pos_label=0)
                            if now_r1 < now_r0:
                                ensemble_temp_save.append(1)
                                ensemble_temp_save.append(test_y_tru[tt])
                                final_y = 0
                                for clfs in classifiers:
                                    temp_y = clfs.predict(test_X)[0]
                                    ensemble_temp_save.append(temp_y)
                                    if temp_y == 1:
                                        final_y = 1
                                test_y_pre[tt] = final_y
                                # if classifiers[0].predict(test_X)[0] == 0 and temp_y == 1:
                                #     print("a cor")
                            else:
                                test_y_pre[tt] = classifiers[0].predict(test_X)[0]
                                ensemble_temp_save.append(0)
                                ensemble_temp_save.append(test_y_tru[tt])
                                ensemble_temp_save.append(test_y_pre[tt])
                                for i in range(len(classifiers) - 1):
                                    ensemble_temp_save.append(-1)
                        else:
                            test_y_pre[tt] = classifiers[0].predict(test_X)[0]
                            ensemble_temp_save.append(-1)
                            ensemble_temp_save.append(test_y_tru[tt])
                            ensemble_temp_save.append(test_y_pre[tt])
                            for i in range(len(classifiers) - 1):
                                ensemble_temp_save.append(-1)

                        ensemble_save.append(ensemble_temp_save)
                        """get the new train data batch"""
                        data_buffer, new_train_defect, new_train_clean, new_train_unlabeled = \
                            set_train_stream(prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer,
                                             wait_days)

                        for nb_ensemble in range(len(ensemble_threshold)):
                            ensemble_new_train_defect = copy.deepcopy(new_train_defect)
                            ensemble_new_train_clean = copy.deepcopy(new_train_clean)
                            for each in ensemble_new_train_clean:
                                each[data_ind_reset.id_y] = 0
                                if nb_ensemble == 0:
                                    project_window[int(each[data_ind_reset.id_target])].append(each)
                            for each in ensemble_new_train_defect:
                                each[data_ind_reset.id_y] = 1
                                if nb_ensemble == 0:
                                    project_window[int(each[data_ind_reset.id_target])].append(each)

                            stay_clean = np.in1d(ensemble_new_train_clean[:, data_ind_reset.id_target],
                                                 ensemble_selected_project[nb_ensemble])
                            stay_defect = np.in1d(ensemble_new_train_defect[:, data_ind_reset.id_target],
                                                  ensemble_selected_project[nb_ensemble])
                            ensemble_new_train_clean = ensemble_new_train_clean[stay_clean]
                            ensemble_new_train_defect = ensemble_new_train_defect[stay_defect]

                            # note the order (clean, defect)
                            cmt_time_train = np.concatenate(
                                (ensemble_new_train_clean[:, data_ind_reset.id_time],
                                 ensemble_new_train_defect[:, data_ind_reset.id_time]))
                            use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                            X_train = np.concatenate(
                                (ensemble_new_train_clean[data_ind_reset.id_X_np],
                                 ensemble_new_train_defect[data_ind_reset.id_X_np]))
                            y_train_obv = np.concatenate(
                                (np.zeros(ensemble_new_train_clean.shape[0]),
                                 np.ones(ensemble_new_train_defect.shape[0])))
                            y_train_tru = np.concatenate(
                                (ensemble_new_train_clean[:, data_ind_reset.id_y],
                                 ensemble_new_train_defect[:, data_ind_reset.id_y]))
                            y_train_target = np.concatenate(
                                (ensemble_new_train_clean[:, data_ind_reset.id_target],
                                 ensemble_new_train_defect[:, data_ind_reset.id_target]))
                            X_train_weight = np.ones(len(y_train_target))
                            # X_train_weight = y_train_target.copy()
                            # for X_i in range(len(X_train_weight)):
                            #     X_i_target = X_train_weight[X_i]
                            #     X_train_weight[X_i] = cp_weight[int(X_i_target)]
                            nb_train_data += y_train_obv.shape[0]

                            # assign
                            cmt_time_train_lst.extend(cmt_time_train.tolist())
                            use_time_train_lst.extend(use_time_train.tolist())
                            y_train_obv_lst.extend(y_train_obv.tolist())
                            y_train_tru_lst.extend(y_train_tru.tolist())
                            if verbose_int >= 2:
                                print("\ttest_step=%d, y_true=%d, y_pre=%d: %s" % (
                                    test_step, test_y_tru[tt], test_y_pre[tt], clf_name))
                                print("\tnew_train: (tru, obv, target)=", y_train_tru, y_train_obv, y_train_target)
                                print("\t\t#acc_train_data = %d" % nb_train_data)

                            """then train: update classifiers and DenStream given new labelled training data"""
                            if y_train_obv.shape[0] > 0:
                                if clf_name == "oza" or "oob" in clf_name:
                                    classifiers[nb_ensemble].partial_fit(X_train, y_train_obv, label_val,
                                                                         X_train_weight)
                                    # assign
                                    cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                                    use_cluster_lst = cl_train_lst
                                elif "pbsa" in clf_name:
                                    classifiers[nb_ensemble].pbsa_flow(X_train, y_train_obv, tt,
                                                                       new_train_unlabeled[data_ind_reset.id_X_np],
                                                                       new_train_defect, data_ind_reset, label_val,
                                                                       X_train_weight)
                                    # assign
                                    cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                                    use_cluster_lst = cl_train_lst
                                elif "orb" in clf_name:
                                    y_train_pre = classifiers[nb_ensemble].predict(X_train)
                                    classifiers[nb_ensemble].partial_fit(X_train, y_train_obv, y_train_pre, label_val,
                                                                         X_train_weight)
                                    # assign
                                    cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                                    use_cluster_lst = cl_train_lst
                                elif "odasc" in clf_name:
                                    X_train_norm = norm_scaler.my_transform(X_train)
                                    if cluster_pres[nb_ensemble] == 0:
                                        clusters[nb_ensemble].partial_fit(X_train_norm, y_train_obv, X_train_weight)
                                        cluster_pres[nb_ensemble] = 1
                                    cl_train, cl_c1_refine, use_cluster_train = \
                                        clusters[nb_ensemble].compute_CLs(X_train_norm, y_train_obv)
                                    # update classifier
                                    classifiers[nb_ensemble].partial_fit(X_train, y_train_obv, cl_train, label_val,
                                                                         X_train_weight)
                                    # update micro-cluster
                                    clusters[nb_ensemble].partial_fit(X_train_norm, y_train_obv, X_train_weight)
                                    clusters[nb_ensemble].revise_cluster_info(X_train_norm, y_train_obv, cl_train)
                                    # assign
                                    cl_train_lst.extend(cl_train.tolist())
                                    use_cluster_lst.extend(use_cluster_train.tolist())
                                    # print
                                    if verbose_int >= 2:
                                        for y_tru_, y_obv_, cl_ in zip(y_train_tru, y_train_obv, cl_train):
                                            print(
                                                "\t\t\ty_trn_tru=%d, y_trn_obv=%d, cl_est=%.2f" % (y_tru_, y_obv_, cl_))
                                    # if pca_plot and False:  # manual control
                                    #     info = "test-step=%d, train Fea14_org with y_true" % test_step
                                    #     cluster.plot_cluster(X_train_norm, y_train_tru, pca_hd, info, x_lim, y_lim, True)
                                else:
                                    raise Exception("Undefined classifier with clf_name=%s." % clf_name)

                        prev_test_time = test_time[tt]  # update VIP
                    else:
                        """test: predict with classifiers"""
                        test_y_pre[tt] = classifier.predict(test_X)[0]

                        """get the new train data batch"""
                        data_buffer, new_train_defect, new_train_clean, new_train_unlabeled = \
                            set_train_stream(prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer,
                                             wait_days)

                        if use_sbp == 0:
                            # note the order (clean, defect)
                            cmt_time_train = np.concatenate(
                                (new_train_clean[:, data_ind_reset.id_time], new_train_defect[:, data_ind_reset.id_time]))
                            use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                            X_train = np.concatenate(
                                (new_train_clean[data_ind_reset.id_X_np], new_train_defect[data_ind_reset.id_X_np]))
                            y_train_obv = np.concatenate(
                                (np.zeros(new_train_clean.shape[0]), np.ones(new_train_defect.shape[0])))
                            y_train_tru = np.concatenate(
                                (new_train_clean[:, data_ind_reset.id_y], new_train_defect[:, data_ind_reset.id_y]))
                            y_train_target = np.concatenate(
                                (new_train_clean[:, data_ind_reset.id_target],
                                 new_train_defect[:, data_ind_reset.id_target]))
                            X_train_weight = np.ones(len(y_train_target))
                            nb_train_data += y_train_obv.shape[0]
                        elif use_sbp == 1:
                            for each in new_train_clean:
                                each[data_ind_reset.id_y] = 0
                                project_window[int(each[data_ind_reset.id_target])].append(each)
                            for each in new_train_defect:
                                each[data_ind_reset.id_y] = 1
                                project_window[int(each[data_ind_reset.id_target])].append(each)

                            stay_clean = np.in1d(new_train_clean[:, data_ind_reset.id_target], selected_project)
                            stay_defect = np.in1d(new_train_defect[:, data_ind_reset.id_target], selected_project)
                            new_train_clean = new_train_clean[stay_clean]
                            new_train_defect = new_train_defect[stay_defect]

                            # note the order (clean, defect)
                            cmt_time_train = np.concatenate(
                                (new_train_clean[:, data_ind_reset.id_time], new_train_defect[:, data_ind_reset.id_time]))
                            use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                            X_train = np.concatenate(
                                (new_train_clean[data_ind_reset.id_X_np], new_train_defect[data_ind_reset.id_X_np]))
                            y_train_obv = np.concatenate(
                                (np.zeros(new_train_clean.shape[0]), np.ones(new_train_defect.shape[0])))
                            y_train_tru = np.concatenate(
                                (new_train_clean[:, data_ind_reset.id_y], new_train_defect[:, data_ind_reset.id_y]))
                            y_train_target = np.concatenate(
                                (new_train_clean[:, data_ind_reset.id_target],
                                 new_train_defect[:, data_ind_reset.id_target]))
                            X_train_weight = np.ones(len(y_train_target))
                            # X_train_weight = y_train_target.copy()
                            # for X_i in range(len(X_train_weight)):
                            #     X_i_target = X_train_weight[X_i]
                            #     X_train_weight[X_i] = cp_weight[int(X_i_target)]
                            nb_train_data += y_train_obv.shape[0]

                        # assign
                        cmt_time_train_lst.extend(cmt_time_train.tolist())
                        use_time_train_lst.extend(use_time_train.tolist())
                        y_train_obv_lst.extend(y_train_obv.tolist())
                        y_train_tru_lst.extend(y_train_tru.tolist())
                        if verbose_int >= 2:
                            print("\ttest_step=%d, y_true=%d, y_pre=%d: %s" % (
                                test_step, test_y_tru[tt], test_y_pre[tt], clf_name))
                            print("\tnew_train: (tru, obv, target)=", y_train_tru, y_train_obv, y_train_target)
                            print("\t\t#acc_train_data = %d" % nb_train_data)

                        """then train: update classifiers and DenStream given new labelled training data"""
                        if y_train_obv.shape[0] > 0:
                            if clf_name == "oza" or "oob" in clf_name:
                                classifier.partial_fit(X_train, y_train_obv, label_val, X_train_weight)
                                # assign
                                cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                                use_cluster_lst = cl_train_lst
                            elif "pbsa" in clf_name:
                                classifier.pbsa_flow(X_train, y_train_obv, tt, new_train_unlabeled[data_ind_reset.id_X_np],
                                                     new_train_defect, data_ind_reset, label_val, X_train_weight)
                                # assign
                                cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                                use_cluster_lst = cl_train_lst
                            elif "orb" in clf_name:
                                y_train_pre = classifier.predict(X_train)
                                classifier.partial_fit(X_train, y_train_obv, y_train_pre, label_val, X_train_weight)
                                # assign
                                cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                                use_cluster_lst = cl_train_lst
                            elif "odasc" in clf_name:
                                X_train_norm = norm_scaler.my_transform(X_train)
                                if cluster_pre == 0:
                                    cluster.partial_fit(X_train_norm, y_train_obv, X_train_weight)
                                    cluster_pre = 1
                                cl_train, cl_c1_refine, use_cluster_train = \
                                    cluster.compute_CLs(X_train_norm, y_train_obv)
                                # update classifier
                                classifier.partial_fit(X_train, y_train_obv, cl_train, label_val, X_train_weight)
                                # update micro-cluster
                                cluster.partial_fit(X_train_norm, y_train_obv, X_train_weight)
                                cluster.revise_cluster_info(X_train_norm, y_train_obv, cl_train)
                                # assign
                                cl_train_lst.extend(cl_train.tolist())
                                use_cluster_lst.extend(use_cluster_train.tolist())
                                # print
                                if verbose_int >= 2:
                                    for y_tru_, y_obv_, cl_ in zip(y_train_tru, y_train_obv, cl_train):
                                        print("\t\t\ty_trn_tru=%d, y_trn_obv=%d, cl_est=%.2f" % (y_tru_, y_obv_, cl_))
                                # if pca_plot and False:  # manual control
                                #     info = "test-step=%d, train Fea14_org with y_true" % test_step
                                #     cluster.plot_cluster(X_train_norm, y_train_tru, pca_hd, info, x_lim, y_lim, True)
                            else:
                                raise Exception("Undefined classifier with clf_name=%s." % clf_name)

                        prev_test_time = test_time[tt]  # update VIP
                else:
                    target_idx[tt] = False
                    if use_sbp == 1:
                        this_project = int(new_1data[0, data_ind_reset.id_target])
                        n_commit[this_project] = n_commit[this_project] + 1
                    test_y_pre[tt] = -1
                    if new_1data.ndim == 1:  # debug
                        new_1data = new_1data.reshape((1, -1))
                    if new_1data[0, data_ind_reset.id_y] == 0:
                        new_1data[0, data_ind_reset.id_vl] = np.inf
                    # set data_buffer, (ts, XX, vl)
                    if data_buffer is None:  # initialize
                        data_buffer = new_1data
                    else:
                        data_buffer = np.vstack((data_buffer, new_1data))
            running_end_time = time.time()
            # only use WP results
            """save returns"""
            if clf_name != "oob" and clf_name != "odasc" and clf_name != "pbsa":
                test_time = test_time[target_idx]
                test_y_tru = test_y_tru[target_idx]
                test_y_pre = test_y_pre[target_idx]
            # return 1: rslt_test ~ (test_time, y_true, y_pred)
            rslt_test = np.vstack((test_time, test_y_tru, test_y_pre)).T
            # return 2: rslt_train ~ (commit_time, use_time, yy, y_obv, cl, use_cluster)
            train_y_tru, train_y_obv = np.array(y_train_tru_lst), np.array(y_train_obv_lst)
            cmt_time_train, use_time_train = np.array(cmt_time_train_lst), np.array(use_time_train_lst)
            # cl_pre, use_cluster = np.array(cl_train_lst), np.array(use_cluster_lst)
            rslt_train = np.vstack((cmt_time_train, use_time_train, train_y_tru, train_y_obv)).T
            running_time = running_end_time - running_start_time
            if "filtering" in clf_name:
                running_time = running_time + filtering_time
            running_info = []
            running_info.append(running_time)

            # save
            if not just_run:
                info_str = " Note: '%d' means invalidity" % invalid_val
                np.savetxt(flnm_test, rslt_test, fmt='%d\t %d\t %d',
                           header="%test_time, yy, y_pre) " + info_str)
                np.savetxt(flnm_train, rslt_train, fmt='%d %d\t %d\t %d',
                           header="%commit_time, use_time, yy, y_obv) " + info_str)
                np.savetxt(flnm_info, running_info, fmt='%d',
                           header="%running_time) " + info_str)
                if "ensemble" in clf_name or "random" in clf_name:
                    ensemble_save = np.array(ensemble_save)
                    np.savetxt(flnm_ensemble, ensemble_save, fmt="%d", header="r1<r0?,true label,main label,others")
            rslt_test_ac_seed.append(rslt_test)

        # demonstration
        if verbose_int >= 1:
            print("\n" + "--" * 20)
            print("%s -- seed=%d: " % (info_run, seed))
            uti_print_pf(rslt_test, rslt_train, with_pretrain)

        """performance evaluation"""
        # cl pf: rmse
        if with_pretrain == 1:
            train_y_tru, train_y_obv, CLs_pre = rslt_train[:, 2], rslt_train[:, 3], rslt_train[:, 4]
            CLs_tru = comp_cl_upper(train_y_tru, train_y_obv)
            cl_rmse_this = uti_eval_cl(CLs_tru, CLs_pre, False)

        # pf eval throughout test steps
        test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
        pfs_tt_dct = uti_eval_pfs(test_y_tru, test_y_pre)

        # assign
        if ss == 0:  # init
            n_row, n_col = pfs_tt_dct["gmean_tt"].shape[0], len(seed_lst)
            cl_rmse, gmean_tt_ss = np.empty(n_col), np.empty((n_row, n_col))
            r1_tt_ss, r0_tt_ss = np.copy(gmean_tt_ss), np.copy(gmean_tt_ss)
            mcc_tt_ss = np.copy(gmean_tt_ss)
            running_time_ss = np.zeros(len(seed_lst))
        if with_pretrain == 1:
            cl_rmse[ss] = cl_rmse_this
        if exist_result:
            running_time_ss[ss] = rslt_info
        else:
            running_time_ss[ss] = running_time
        gmean_tt_ss[:, ss], r1_tt_ss[:, ss], r0_tt_ss[:, ss], mcc_tt_ss[:, ss] = \
            pfs_tt_dct["gmean_tt"], pfs_tt_dct["recall1_tt"], pfs_tt_dct["recall0_tt"], pfs_tt_dct["mcc_tt"]

    """ave pf across seeds"""
    gmean_tt_ave_ss = np.nanmean(gmean_tt_ss, axis=1)
    r1_tt_ave_ss = np.nanmean(r1_tt_ss, axis=1)
    r0_tt_ave_ss = np.nanmean(r0_tt_ss, axis=1)
    mcc_tt_ave_ss = np.nanmean(mcc_tt_ss, axis=1)
    running_time_ave_ss = np.nanmean(running_time_ss)

    get_periods_pf = False
    if get_periods_pf:
        """periods division"""
        if "odasc" in clf_name:
            base_name = "odasc"
        elif "oob" in clf_name:
            base_name = "oob"
        elif "pbsa" in clf_name:
            base_name = "pbsa"
        division_dir = "%s%s.rslt_test.s" % (to_dir, clf_name)
        real_division_dir = "%s%s.division.s" % (to_dir, base_name)
        initial, peaks = period_division(division_dir, real_division_dir, base_name, clf_name, len(seed_lst), info_run)
        initial_pf, drop_pf, stable_pf = evaluate_periods(initial, peaks, gmean_tt_ave_ss)
    else:
        initial_pf, drop_pf, stable_pf = np.nan, np.nan, np.nan

    """save result to csv file"""
    if not just_run and not load_result:
        to_dir_csv = "../results/rslt.report/"
        os.makedirs(to_dir_csv, exist_ok=True)
        to_flnm_csv = to_dir_csv + "pf_bst_ave%d_p%d_n%d.csv" % (len(seed_lst), 1000, report_nb_test)
        with open(to_flnm_csv, "a+") as fh2:
            if not os.path.getsize(to_flnm_csv):  # header
                print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
                    "project", "wait_days", "method", "gmean_bst", "r1_bst", "r0_bst", "mcc_bst",
                    "initial_pf", "drop_pf", "stable_pf", "running_time"),
                      file=fh2)
            print("%s,%d,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d" % (
                project_name, wait_days, clf_name,
                np.nanmean(gmean_tt_ave_ss), np.nanmean(r1_tt_ave_ss), np.nanmean(r0_tt_ave_ss),
                np.nanmean(mcc_tt_ave_ss),
                initial_pf, drop_pf, stable_pf, running_time_ave_ss), file=fh2)

    # demonstrate
    if verbose_int >= 0:
        print("\n" + "==" * 20)
        print("%s -- ave seeds:" % info_run)
        print("\tgmean=%.4f, r1=%.4f, r0=%.4f, mcc=%.4f" % (
            np.nanmean(gmean_tt_ave_ss), np.nanmean(r1_tt_ave_ss), np.nanmean(r0_tt_ave_ss), np.nanmean(mcc_tt_ave_ss)))
    if pca_plot:
        pfs_tt = np.column_stack((gmean_tt_ave_ss, r1_tt_ave_ss, r0_tt_ave_ss))
        uti_plot_pfs_online(pfs_tt, ["gmean", "r1", "r0"], info_run, save_plot=True)

    # return 1~3: gmean_tt_ave_ss ~(test_step, )
    # return 4: cl_rmse_ave_ss~float
    rslt_return = {"gmean": gmean_tt_ss, "mcc": mcc_tt_ss, "r1": r1_tt_ss, "r0": r0_tt_ss, "gmean_ave": gmean_tt_ave_ss,
                   "mcc_ave": mcc_tt_ave_ss, "r1_ave": r1_tt_ave_ss, "r0_ave": r0_tt_ave_ss}
    return rslt_return, rslt_test_ac_seed


def cpps_record(project_id=6, clf_name="oob", nb_para_tune=1000, nb_test=5000, wait_days=15, verbose_int=0,
                use_fs=False, is_RQ1=False):
    """
    This method is related to the similarity calculation and RQ1.
    This method is used to record the similarities between WP and CPs across time into csv file.

    Args:
        project_id (int): The index of target project (WP).
        clf_name (string): The name of base JIT-SDP model and the CP method.
        nb_para_tune (int): The number of WP data used to do parameter tuning.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        wait_days (int): The waiting time in online JIT-SDP.
        verbose_int (int): A number to control the print of running information. "-1" means no print; a larger value
            means deeper and more detailed "print".
        use_fs (boolean): If True, it will only use the selected metrics to calculate similarities.
        is_RQ1 (boolean): If True, this method will calculate the similarities used in RQ1 (using the data before the
            5000th WP data).
    """
    project_name = data_id_2name(project_id)
    print("start: " + project_name + " " + clf_name)
    """prepare test data stream"""
    report_nb_test = nb_test
    test_stream = set_test_stream(project_name)
    test_stream.X = np.hstack(
        (test_stream.X, (np.ones(len(test_stream.X)) * project_id).reshape(len(test_stream.X), 1)))
    X_org = test_stream.X[class_data_ind_org().id_X_np]
    # convert fea14 to fea13 and the test data stream
    XX, use_data = real_data_preprocess(X_org)
    yy = test_stream.y[use_data]
    time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
    vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
    target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

    # handle negative nb_test
    n_data_all, n_fea = XX.shape[0], XX.shape[1]  # after fea conversion for jit-sdp
    assert n_fea == 12, "# transformed fea should be 12. Sth. is wrong."
    if nb_test < 0:
        nb_test += n_data_all
        if verbose_int >= 2:
            print("actual nb_test=%d" % nb_test)
    assert nb_para_tune <= nb_test, "nb_pre=%d should be smaller than nb_data=%d" % (nb_para_tune, nb_test)

    norm_data = XX

    # prepare all test samples
    test_data_all = np.hstack((time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
    data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                          n_fea=n_fea)
    """add cross project data"""
    for i in range(23):
        if i != project_id:
            project_name_cp = data_id_2name(i)
            test_stream = set_test_stream(project_name_cp)
            test_stream.X = np.hstack(
                (test_stream.X, (np.ones(len(test_stream.X)) * i).reshape(len(test_stream.X), 1)))
            X_org = test_stream.X[class_data_ind_org().id_X_np]
            # convert fea14 to fea13 and the test data stream
            XX, use_data = real_data_preprocess(X_org)
            yy = test_stream.y[use_data]
            time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
            vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
            target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

            test_data_temp = np.hstack((time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
            test_data_all = np.vstack([test_data_all, test_data_temp])

            norm_data = np.vstack([norm_data, XX])

    idx = test_data_all[:, 0].argsort()
    test_data_all = test_data_all[idx]

    # fea normalizer based on all test data used for DenStream
    norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
    norm_scaler.my_fit(norm_data)

    # find the index that contains nb_test target data
    count = 0
    nb_pre = nb_para_tune
    for i in range(len(test_data_all)):
        if test_data_all[i][-1] == project_id:
            count = count + 1
            if nb_test == -1:
                nb_test_WP = i
        if count == nb_pre:
            nb_para_tune = i
        if count == nb_test and nb_test != -1:
            nb_test_WP = i
            break

    target_idx = test_data_all[:, data_ind_reset.id_target] == project_id
    target_idx_para = target_idx[:nb_para_tune]
    if with_pretrain == 1:
        target_idx = target_idx[nb_para_tune + 1:nb_test_WP + 1]
    else:
        target_idx = target_idx[1:nb_test_WP + 1]

    nb_run = 5
    dir_cpps_para = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/"
    os.makedirs(dir_cpps_para, exist_ok=True)
    auto_name = "%s-cpps-para-%dstep-%drun" % (clf_name, nb_pre, nb_run) + ".pkl"
    exist_clf_para = os.path.exists(dir_cpps_para + auto_name)
    if exist_clf_para:
        para_dict = pkl.load(open(dir_cpps_para + auto_name, 'rb'))
        cpps_window_size, update_period, select_threshold = \
            para_dict["window_size"], para_dict["update_period"], para_dict["select_threshold"]
    else:
        raise "para not found"
    if is_RQ1 == True:
        cpps_window_size = nb_test
        update_period = nb_test
    # data pre-train
    # if clf_name == "odasc_addcp":
    #     data_ptrn = test_data_all[:nb_para_tune]
    #     data_ptrn = data_ptrn[target_idx_para]
    # else:
    #     data_ptrn = test_data_all[:nb_para_tune]
    data_ptrn = test_data_all[:nb_para_tune]
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]

    """[core] test-then-training process:
                at each test step, only one test data arrives, while maybe no or several training data become available
                """
    if with_pretrain == 1:
        nb_test_act = nb_test_WP - nb_para_tune
    else:
        nb_test_act = nb_test_WP
    test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
    cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
    cl_train_lst, use_cluster_lst = [], []
    CP_positive_data = []
    CP_positive_size = 50
    pre_r1, pre_r2, pre_gmean = 0, 0, 0

    if with_pretrain == 1:
        prev_test_time, data_buffer, nb_train_data = data_ptrn[-1, data_ind_reset.id_time], None, 0  # vip
    else:
        prev_test_time, data_buffer, nb_train_data = test_data_all[0, data_ind_reset.id_time], None, 0  # vip
    notadd = 0
    # project_window, selected_project = sbp_initial(range(14), 500)
    project_window, selected_project = sbp_initial(range(23), cpps_window_size)
    use_sbp = 1
    n_commit = np.zeros(len(project_window))
    combine_way = "L1"
    cp_weight = np.ones(23)
    selected_project = [project_id]
    selected_project = np.array(selected_project)
    selected_features = []
    if use_fs == True:
        print("use selected features")
        selected_features = load_selected_features(clf_name, project_id)
    else:
        for feature_id in range(17):
            selected_features.append(feature_id)
    for tt in range(nb_test_act):
        # get the test data
        if with_pretrain == 1:
            test_step = tt + nb_para_tune
        else:
            test_step = tt
        new_1data = test_data_all[test_step, :].reshape((1, -1))
        test_X = new_1data[data_ind_reset.id_X_np]
        test_time[tt] = new_1data[:, data_ind_reset.id_time]
        test_y_tru[tt] = new_1data[:, data_ind_reset.id_y]

        if use_sbp == 1:
            if n_commit[project_id] % update_period == 0 and new_1data[0, data_ind_reset.id_target] == project_id:
                cp_weight = update_cp_weight(project_id, project_window, data_ind_reset, n_commit, combine_way,
                                             selected_features)
                cp_length = []
                for index_cp in range(len(n_commit)):
                    cp_length.append(n_commit[index_cp])
                ###
                to_dir_csv = "../results/rslt.report/"
                os.makedirs(to_dir_csv, exist_ok=True)
                if use_fs == False:
                    to_flnm_csv = to_dir_csv + "cpps_similarity.csv"
                else:
                    to_flnm_csv = to_dir_csv + clf_name + "_cpps_similarity_fs.csv"
                with open(to_flnm_csv, "a+") as fh2:
                    if not os.path.getsize(to_flnm_csv):  # header
                        print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"
                              "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
                                  "project", "timestep", "sim0", "sim1", "sim2", "sim3", "sim4", "sim5", "sim6", "sim7",
                                  "sim8", "sim9", "sim10", "sim11", "sim12", "sim13", "sim14", "sim15", "sim16",
                                  "sim17",
                                  "sim18", "sim19", "sim20", "sim21", "sim22",
                                  "len0", "len1", "len2", "len3", "len4", "len5", "len6", "len7", "len8",
                                  "len9", "len10", "len11", "len12", "len13", "len14", "len15", "len16", "len17",
                                  "len18", "len19", "len20", "len21", "len22",
                                  "update_period", "window_size", "threshold", "clf_name"
                              ), file=fh2)
                    print(
                        "%s,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
                        "%.5f,%.5f,%.5f,%.5f,%.5f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
                        "%d,%d,%d,%.2f,%s" % (
                            project_name, cp_length[project_id], cp_weight[0], cp_weight[1], cp_weight[2], cp_weight[3],
                            cp_weight[4], cp_weight[5], cp_weight[6], cp_weight[7], cp_weight[8], cp_weight[9],
                            cp_weight[10],
                            cp_weight[11], cp_weight[12], cp_weight[13], cp_weight[14], cp_weight[15], cp_weight[16],
                            cp_weight[17],
                            cp_weight[18], cp_weight[19], cp_weight[20], cp_weight[21], cp_weight[22],
                            cp_length[0], cp_length[1], cp_length[2], cp_length[3],
                            cp_length[4], cp_length[5], cp_length[6], cp_length[7], cp_length[8], cp_length[9],
                            cp_length[10],
                            cp_length[11], cp_length[12], cp_length[13], cp_length[14], cp_length[15], cp_length[16],
                            cp_length[17],
                            cp_length[18], cp_length[19], cp_length[20], cp_length[21], cp_length[22],
                            update_period, cpps_window_size, select_threshold, clf_name
                        ), file=fh2)
                ###
                for index_weight in range(len(cp_weight)):
                    if cp_weight[index_weight] < 0:
                        cp_weight[index_weight] = 0
                selected_project = []
                for index_project in range(23):
                    if cp_weight[index_project] >= 1:
                        selected_project.append(index_project)
                selected_project = np.array(selected_project)
                # print(clf_name + " " + project_name + " ts:" + str(
                #     n_commit[project_id]) + " cp_weight:" + str(cp_weight))
                # print(selected_project)

        if new_1data[0, data_ind_reset.id_target] == project_id:
            target_idx[tt] = True
            if use_sbp == 1:
                n_commit[project_id] = n_commit[project_id] + 1
            data_buffer, new_train_defect, new_train_clean, new_train_unlabeled = \
                set_train_stream(prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer,
                                 wait_days)
            for each in new_train_clean:
                each[data_ind_reset.id_y] = 0
                project_window[int(each[data_ind_reset.id_target])].append(each)
            for each in new_train_defect:
                each[data_ind_reset.id_y] = 1
                project_window[int(each[data_ind_reset.id_target])].append(each)

        else:
            target_idx[tt] = False
            if use_sbp == 1:
                this_project = int(new_1data[0, data_ind_reset.id_target])
                n_commit[this_project] = n_commit[this_project] + 1
            test_y_pre[tt] = -1
            if new_1data.ndim == 1:  # debug
                new_1data = new_1data.reshape((1, -1))
            if new_1data[0, data_ind_reset.id_y] == 0:
                new_1data[0, data_ind_reset.id_vl] = np.inf
            # set data_buffer, (ts, XX, vl)
            if data_buffer is None:  # initialize
                data_buffer = new_1data
            else:
                data_buffer = np.vstack((data_buffer, new_1data))
    return 0


def para_denStream(X_norm, y_true, nb_repeat=10):
    """
    This method is deprecated.
    This method is used to do parameter tuning of Denstream cluster used in ODaSC.

    Args:
        X_norm (list): The normalized feature values of data used for parameter tuning.
        y_true (list): The class label of data used for parameter tuning.

    Returns:
        eps_opt (float): A parameter for Denstream.
        mu_opt (float): A parameter for Denstream.
        beta_opt (float): A parameter for Denstream.
        lambd_opt (float): A parameter for Denstream.
    """
    # define evaluation
    seed_auto_tune = 2380546
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=nb_repeat, random_state=seed_auto_tune)

    # define search space
    space = dict()
    space['eps'] = loguniform(0.1, 20)
    space['mu'] = loguniform(0.1, 15)
    space['beta'] = uniform(0.2, 0.6)
    space['lambd'] = loguniform(0.125, 1.5)

    # define model
    class our_cluster(BaseEstimator, ClusterMixin):
        def __init__(self, lambd=None, eps=None, beta=None, mu=None):
            self.cluster = None
            self.eps = eps
            self.lambd = lambd
            self.beta = beta
            self.mu = mu

        # y: ignored
        def fit(self, X_norm, y=y_true):
            theta_cl = 0.8  # by default, no impact

            # Zixin modified, 2022/08/09, raise an exception when beta * mu <= 1
            if self.beta * self.mu <= 1:
                raise AssertionError("[Parameters Error]beta * mu <= 1, skipped\n")
            self.cluster = DenStream(theta_cl=theta_cl, lambd=self.lambd, eps=self.eps, beta=self.beta, mu=self.mu)
            self.cluster.partial_fit(X_norm, y)

        def predict(self, testX):
            y_pred = self.cluster.predict(testX)
            return y_pred

    model = our_cluster()

    # define search
    # y ignored
    def silhouette_scorer(estimator, X, y=y_true):
        try:
            estimator.fit(X)
        except AssertionError as ae:
            print(ae)
            return -1

        labels = estimator.predict(X)
        try:
            score = silhouette_score(X, labels, random_state=seed_auto_tune)
        except ValueError:
            print("[Value Error] All labels are the same: %d" % np.unique(labels)[0])
            return -1
        return score

    search = RandomizedSearchCV(
        model, space, n_iter=200, scoring=silhouette_scorer,
        n_jobs=-1, cv=cv, random_state=seed_auto_tune, verbose=False)

    # execute search
    result = search.fit(X_norm, y_true)

    # prepare returns
    eps_opt = result.best_params_['eps']
    mu_opt = result.best_params_['mu']
    beta_opt = result.best_params_['beta']
    lambd_opt = result.best_params_['lambd']
    return eps_opt, mu_opt, beta_opt, lambd_opt


def comp_cl_upper(y_true, y_obv):
    """
    This method is related to ODaSC (a state-of-the-art JIT-SDP model).
    This method is used to calculate the upper bound of confidence level.

    Args:
        y_true (list): The list of true labels.
        y_obv (list): The list of observed labels.

    Returns:
        upper_conf_levels (list): The upper bound of confidence level of each data.
    """
    assert y_true.shape == y_obv.shape, "the shape of y_obv should equal to that of y_true"
    upper_conf_levels = np.ones(np.size(y_true))
    upper_conf_levels[np.where(y_obv != y_true)] = 0
    # upper_conf_levels = np.where(y_true == y_true, 1, 0)
    return upper_conf_levels


class my_norm_scaler:
    """
    This class is used for normalization.
    """
    def __init__(self, n_fea, norm_name="z_score"):
        """
        This method is used to initialize this class.

        Args:
            n_fea (int): The number of features of a commit data.
            norm_name (string): The type of normalization
        """
        self.n_fea = n_fea
        self.norm_name = norm_name  # by default z-score
        if self.norm_name.lower() == "min_max".lower():
            self.norm_scaler = preprocessing.MinMaxScaler()
        elif self.norm_name.lower() == "z_score".lower():
            self.norm_scaler = preprocessing.StandardScaler()

    def check_feature(self, XX):
        """
        This method is used to check whether the data meets the requirements.

        Args:
            XX (list): The list of feature values of commit data.
        """
        assert XX.shape[1] == self.n_fea, "wrong fea number. It should be 12 for transformed jit-sdp."

    def my_fit(self, XX):
        """
        This method is used to train this class.

        Args:
            XX (list): The list of feature values of commit data.

        Returns:
            my_norm (object): An object used for normalization.
        """
        self.check_feature(XX)
        """see comments in my_transform() below"""
        if self.n_fea == 12:  # for jit-sdp: the 1st fea "fix_bug" should NOT be normalised.
            my_norm = self.norm_scaler.fit(XX[:, 1:])
        else:  # for synthetic
            my_norm = self.norm_scaler.fit(XX)
        return my_norm

    def my_transform(self, xx):
        """
        This method is used to normalize the input commit data.

        Args:
            XX (list): The list of feature values of commit data.

        Returns:
            xx_trans (list): The list of feature values of commit data after normalization.
        """
        if xx.ndim == 1:  # if xx contains only 1 data sample
            xx = xx.reshape((-1, self.n_fea))
        """the real jit-sdp vs synthetic. 
        This is roughly decided based on # fea: for jit-sdp: n_fea=13; for syn: probably NOT 13.
        """
        if self.n_fea == 12:  # for jit-sdp: the 1st fea "fix_bug" should remain unchanged.
            xx_trans = np.hstack((xx[:, 0].reshape(-1, 1), self.norm_scaler.transform(xx[:, 1:])))
        else:  # for synthetic
            xx_trans = self.norm_scaler.transform(xx)
        return xx_trans


def uti_rslt_dir(clf_name="odasc", project_id=1, wait_days=15,
                 n_trees=5, theta_imb=0.9, theta_cl=0.8, p=0.25, m=1.5, th=0.3,
                 filtering_window_size=500, K=50, max_dist=0.7, discard_size=500,
                 cpps_window_size=500, update_period=100, selected_threshold=0.3):
    """
    This method is used to find the directory of the folder which save the prediction results.
    If the prediction results already exist, it will just load the results to save time.

    Args:
        clf_name (string): The name of base JIT-SDP model and the CP method.
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        n_trees (int): A parameter for OOB, ODaSC and PBSA.
        theta_imb (float): A parameter of OOB, ODaSC and PBSA.
        theta_cl (float): A parameter of ODaSC.
        p (float): A parameter of PBSA.
        m (float): A parameter of PBSA.
        th (float): A parameter of PBSA.
        filtering_window_size (int): A parameter of Filtering.
        K (int): A parameter of Filtering.
        max_dist (float): A parameter of Filtering.
        discard_size (int): A parameter of Filtering.
        cpps_window_size (int): A parameter of CroPS.
        update_period (int): A parameter of CroPS.
        selected_threshold (float): A parameter of CroPS.

    Returns:
        to_dir (string): The directory of the folder which save the prediction results.
    """
    # 2022-7-30
    clf_name = clf_name.lower()
    pre_to_dir = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/" + str(wait_days) + "d"
    to_dir = pre_to_dir + "/n_trees" + str(n_trees) + "-theta_imb" + str(theta_imb)  # para info, classifier
    if "odasc" in clf_name:
        to_dir += "-theta_cl" + str(theta_cl)
    if "pbsa" in clf_name:
        to_dir += "-p" + str(p) + "-m" + str(m) + "-th" + str(th)
    if "filtering" in clf_name:
        to_dir += "-filtering_window_size" + str(filtering_window_size) + "-K" + str(K) + "-max_dist" + str(max_dist) \
                  + "-discard_size" + str(discard_size)
    if "cpps" in clf_name:
        to_dir += "-cpps_window_size" + str(cpps_window_size) + "-update_period" + str(update_period) + \
                  "-threshold" + str(selected_threshold)

    # if clf_name != "oza" and clf_name != "oob" and clf_name != "our":
    #     to_dir += "k_refine%d" % k_power_refine  # modified
    return to_dir


def uti_rslt_dir_base_para(clf_name="odasc", project_id=1, wait_days=15,
                           n_trees=5, theta_imb=0.9, theta_cl=0.8, p=0.25, m=1.5, th=0.3):
    """
    This method is used to find the directory of the folder which save the prediction results used for parameter tuning
        of base JIT-SDP models, including OOB, ODaSC and PBSA.

    Args:
        clf_name (string): The name of base JIT-SDP model.
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        n_trees (int): A parameter for OOB, ODaSC and PBSA.
        theta_imb (float): A parameter of OOB, ODaSC and PBSA.
        theta_cl (float): A parameter of ODaSC.
        p (float): A parameter of PBSA.
        m (float): A parameter of PBSA.
        th (float): A parameter of PBSA.

    Returns:
        to_dir (string): The directory of the folder which save the prediction results.
    """
    clf_name = clf_name.lower()
    pre_to_dir = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/para/" + str(wait_days) + "d"
    to_dir = pre_to_dir + "/n_trees" + str(n_trees) + "-theta_imb" + str(theta_imb) + "-theta_cl" + str(theta_cl) \
             + "-p" + str(p) + "-m" + str(m) + "-th" + str(th)  # para info, classifier
    return to_dir


def uti_rslt_dir_filtering_para(clf_name="odasc", project_id=1, wait_days=15,
                                window_size=500, k=5, max_dist=0.8, discard_size=500):
    """
    This method is used to find the directory of the folder which save the prediction results used for parameter tuning
        of Filtering (a state-of-the-art online CP method).

    Args:
        clf_name (string): The name of base JIT-SDP model.
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        window_size (int): A parameter of Filtering.
        K (int): A parameter of Filtering.
        max_dist (float): A parameter of Filtering.
        discard_size (int): A parameter of Filtering.

    Returns:
        to_dir (string): The directory of the folder which save the prediction results.
    """
    clf_name = clf_name.lower()
    pre_to_dir = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/para/" + str(wait_days) + "d"
    to_dir = pre_to_dir + "/window_size" + str(window_size) + "-k" + str(k) + "-max_dist" + str(max_dist) \
             + "-discard_size" + str(discard_size)  # para info, classifier
    return to_dir


def uti_rslt_dir_cpps_para(clf_name="odasc", project_id=1, wait_days=15,
                           window_size=500, update_period=100, selected_threshold=0.3):
    """
    This method is used to find the directory of the folder which save the prediction results used for parameter tuning
        of proposed CroPS.

    Args:
        clf_name (string): The name of base JIT-SDP model.
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        window_size (int): A parameter of CroPS.
        update_period (int): A parameter of CroPS.
        selected_threshold (float): A parameter of CroPS.

    Returns:
        to_dir (string): The directory of the folder which save the prediction results.
    """
    clf_name = clf_name.lower()
    pre_to_dir = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/para/" + str(wait_days) + "d"
    to_dir = pre_to_dir + "/window_size" + str(window_size) + "-update_period" + str(update_period) + \
             "-threshold" + str(selected_threshold)  # para info, classifier
    return to_dir


def uti_rslt_dir_ground_truth(clf_name="odasc", project_id=1, wait_days=15,
                 n_trees=5, theta_imb=0.9, theta_cl=0.8, p=0.25, m=1.5, th=0.3,
                 used_project=[0]):
    """
    This method is used to find the directory of the folder which save the prediction results about the "WP+1CP"
        experiments in RQ1.

    Args:
        clf_name (string): The name of base JIT-SDP model.
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        n_trees (int): A parameter for OOB, ODaSC and PBSA.
        theta_imb (float): A parameter of OOB, ODaSC and PBSA.
        theta_cl (float): A parameter of ODaSC.
        p (float): A parameter of PBSA.
        m (float): A parameter of PBSA.
        th (float): A parameter of PBSA.
        used_project (list): The list of the index of projects which will be used in the JIT-SDP model.

    Returns:
        to_dir (string): The directory of the folder which save the prediction results.
    """
    # 2022-7-30
    clf_name = clf_name.lower()
    pre_to_dir = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/" + str(wait_days) + "d"
    to_dir = pre_to_dir + "/n_trees" + str(n_trees) + "-theta_imb" + str(theta_imb)  # para info, classifier
    if "odasc" in clf_name:
        to_dir += "-theta_cl" + str(theta_cl)
    if "pbsa" in clf_name:
        to_dir += "-p" + str(p) + "-m" + str(m) + "-th" + str(th)
    to_dir += "-used_project" + str(used_project)
    return to_dir


def uti_rslt_dir_analyze(to_dir, clf_name, nb_test, seed):
    """
    This method is used to find the final directory of the files which save the prediction results.

    Args:
        to_dir (string): The directory of the folder which save the prediction results.
        clf_name (string): The name of base JIT-SDP model and the CP method.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        seed (int): The random seed set in the current running.

    Returns:
        exist_result (boolean): If True, the results exist.
        to_dir (string): If exist_result is True, this is the final directory of the files which save the prediction
            results. Else, this is the directory of the folder which save the prediction results.
    """
    exist_result = False
    fold_names = next(os.walk(to_dir))[1]
    if len(fold_names) > 0:
        for _, fold_name in enumerate(fold_names):
            nb_test_saved = int(fold_name[fold_name.find("T") + 1:])
            # use new
            # 20230419 change == as >=
            if nb_test_saved == nb_test:
                to_dir_4save = to_dir
                to_dir += "/T" + str(nb_test_saved) + "/"
                flnm_test = to_dir + clf_name + ".rslt_test.s" + str(seed)
                exist_result = os.path.exists(flnm_test)
                if exist_result:
                    break
                else:
                    """handle empty (e.g.) T5000 folder"""
                    to_dir = to_dir_4save
    return exist_result, to_dir


def uti_plot_pfs_online(pf_tt, clf_lst, title_info, save_plot=False):
    """
    This method is used to plot the performance across time.

    Args:
        pf_tt (list): The performance across all time steps.
        clf_lst (list): The evaluation metrics needed to be plotted.
        title_info (string): The title of the plotted figure.
        save_plot (boolean): If True, save the plotted figure.
    """
    if not (isinstance(clf_lst, list) or isinstance(clf_lst, tuple)):
        raise Exception("Error: clf_lst should be a list")
    if np.ndim(pf_tt) == 1:
        pf_tt = pf_tt[:, np.newaxis]
    if pf_tt.shape[1] != len(clf_lst):
        raise Exception("Error: # classifier NOT matches column size of pf_tt")

    # plot
    xx = np.array(range(pf_tt.shape[0]))  # shape (nb_test, )
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cc, clf_name in enumerate(clf_lst):
        ax.plot(xx, pf_tt[:, cc], label=clf_name)

    # setup
    ax.set_title(title_info)
    plt.ylim((-0.1, 1.1))
    ax.grid(True)
    ax.legend(loc="best")

    # show/plot
    if not save_plot:
        plt.show()
    else:
        to_dir_png = "../results/rslt.plot/"
        os.makedirs(to_dir_png, exist_ok=True)
        to_flnm = to_dir_png + title_info
        to_flnm = to_flnm.replace(":", "_")
        to_flnm = to_flnm.replace(",", "_")
        to_flnm = to_flnm.replace(" ", "")
        plt.savefig(to_flnm + ".png")
        # print("plot results are saved into %s" % to_dir_png)


def uti_print_pf(rslt_test, rslt_train, with_pretrain):
    """
    This method is used to print some detailed information of running.

    Args:
        rslt_test (list): The information of the test data on whole time step.
        rslt_train (lsit): The information of the training data on whole time step.
        with_pretrain (boolean): If True, the running process contains pretrain.
    """
    # extract data info
    test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
    if with_pretrain == 1:
        y_train_tru_all, y_train_obv_all, cl_pre = rslt_train[:, 2], rslt_train[:, 3], rslt_train[:, 4]

        # training label noise
        nb_train, nb_train_noise = y_train_obv_all.shape[0], len(np.where(y_train_tru_all != y_train_obv_all)[0])
        trn_label_noise = nb_train_noise / nb_train  # vip
        print("\t training: label_noise=%f" % trn_label_noise)

        # training 1-sided label noise
        nb_defect = np.sum(y_train_obv_all == 1)
        trn_1side_noise = nb_train_noise / nb_defect
        print("\t training: 1sided_noise=%f" % trn_1side_noise)

    # c1% of test data, i.e., true c1%
    nb_test_c1, nb_test = len(np.where(test_y_tru == 1)[0]), test_y_tru.shape[0]
    tst_c1_percent = nb_test_c1 / nb_test
    print("\t testing: class imbalance c1%%=%f" % tst_c1_percent)

    # pf: rmse
    if with_pretrain == 1:
        cl_tru = comp_cl_upper(y_train_tru_all, y_train_obv_all)
        uti_eval_cl(cl_tru, cl_pre, True)
    # pf: online prediction
    uti_eval_pfs(test_y_tru, test_y_pre, True)


def uti_eval_cl(CL_tru, CL_est, verbose=False):
    """
    This method is related to ODaSC (a state-of-the-art JIT-SDP model).
    This method is used to evaluate the estimated label confidence.

    Args:
        CL_tru (list): The real label confidence level.
        CL_est (list): The estimated label confidence level.

    Returns:
        rmse (float): The mean squared error between real label confidence level and estimated label confidence level.
    """
    rmse = mean_squared_error(CL_tru, CL_est, squared=False)
    if verbose:
        print("\t rmse of cl_est=%f." % rmse)
    return rmse


def uti_eval_pfs(test_y_tru, test_y_pre, verbose=False):
    """
    This method is used to evaluate the performance of JIT-SDP model and CP method.

    Args:
        test_y_tru (list): The ture label of test data.
        test_y_pre (list): The prediction label of test data.

    Returns:
        pfs_tt_dct (dict): The average performance across time, including gmean, recall1, recall0 and mcc.
    """
    # ave PFs across test steps
    theta_eval = 0.99
    pfs_tt_dct = compute_online_PF(test_y_tru, test_y_pre, theta_eval)
    gmean_ave_tt = np.nanmean(pfs_tt_dct["gmean_tt"])
    mcc_ave_tt = np.nanmean(pfs_tt_dct["mcc_tt"])
    r1_ave_tt, r0_ave_tt = np.nanmean(pfs_tt_dct["recall1_tt"]), np.nanmean(pfs_tt_dct["recall0_tt"])
    if verbose:
        print("\t ave online gmean=%.4f, r1=%.4f, r0=%.4f, mcc=%.4f" % (gmean_ave_tt, r1_ave_tt, r0_ave_tt, mcc_ave_tt))
    return pfs_tt_dct


def period_division(dir, real_dir, base_name, target_name, times, info_run):
    """
    This method is deprecated.
    This method is used to divide the periods across time, including initial phase, sudden drop phase and stable phase.

    Args:
        dir (string): The directory to save the information about period division.
        real_dir (string): The directory to save the information about period division.
        base_name (string): The name of the base JIT-SDP model.
        target_name (string): The name of base JIT-SDP model and the CP method.
        times (int): The runing times.
        info_run (string): The information of running.

    Returns:
        initial (int): The time step of the end of initial phase.
        peaks (list): The time steps of the start and end of each sudden drop phase.
    """
    exist_division = os.path.exists(real_dir)
    if exist_division:
        div_rslt = np.loadtxt(real_dir)
        initial, peaks = int(div_rslt[0]), []
        for each_peak in div_rslt[1:]:
            peaks.append(int(each_peak))
    else:
        dir = dir.replace(target_name, base_name)
        info_run = info_run.replace(target_name, base_name)
        for ss in range(times):
            rslt_test = np.loadtxt(dir + str(ss))
            # pf eval throughout test steps
            test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
            pfs_tt = uti_eval_pfs(test_y_tru, test_y_pre)

            # assign
            if ss == 0:  # init
                n_row, n_col = pfs_tt["gmean_tt"].shape[0], times
                gmean_tt_ss = np.empty((n_row, n_col))

            gmean_tt_ss[:, ss] = pfs_tt["gmean_tt"]

        gmean_tt_ave_ss = np.nanmean(gmean_tt_ss, axis=1)

        gmean_avg = np.nanmean(gmean_tt_ave_ss)
        threshold = 0.8 * gmean_avg
        initial = -1
        drop = []
        for i in range(len(gmean_tt_ave_ss)):
            if initial == -1 and gmean_tt_ave_ss[i] >= gmean_avg:
                initial = i
            if initial != -1 and i >= 1:
                if gmean_tt_ave_ss[i - 1] >= threshold and gmean_tt_ave_ss[i] <= threshold:
                    drop.append(i)
                elif gmean_tt_ave_ss[i - 1] <= threshold and gmean_tt_ave_ss[i] >= threshold:
                    drop.append(i)
        peaks = detecta.detect_onset(gmean_tt_ave_ss, threshold=threshold, n_below=100)
        larger = peaks[:] > initial
        peaks = peaks[larger]
        if len(peaks) % 2 != 0:
            peaks = peaks[:-1]
        print(initial)
        print(peaks)
        # demonstrate
        division_plot = 0
        print("\n" + "==" * 20)
        print("%s -- ave seeds:" % info_run)
        print("\tgmean=%.4f" % (gmean_avg))

        if division_plot:
            uti_plot(gmean_avg, threshold, initial, peaks, gmean_tt_ave_ss, ["gmean"], info_run, save_plot=True)

        if base_name == target_name:
            dir = dir.replace("rslt_test", "division")
            save_result = np.concatenate((np.array([initial]), peaks))
            np.savetxt(dir, save_result, fmt='%d',
                       header="%inital and peaks")
    return initial, peaks


def uti_plot(gmean_avg, threshold, initial, peaks, pf_tt, clf_lst, title_info, save_plot=False):
    """
    This method is deprecated.
    This method is used to plot the performance across time of periods division of base method.

    Args:
        gmean_avg (float): The average gmean across time.
        threshold (float): The threshold to divide sudden drop period.
        initial (int): The time step of the end of initial phase.
        peaks (list): The time steps of the start and end of each sudden drop phase.
        pf_tt (list): The gmean across all time steps.
        clf_lst (list): The evaluation metrics needed to be plotted.
        title_info (string): The title of the plotted figure.
        save_plot (boolean): If True, save the plotted figure.
    """
    if not (isinstance(clf_lst, list) or isinstance(clf_lst, tuple)):
        raise Exception("Error: clf_lst should be a list")
    if np.ndim(pf_tt) == 1:
        pf_tt = pf_tt[:, np.newaxis]
    if pf_tt.shape[1] != len(clf_lst):
        raise Exception("Error: # classifier NOT matches column size of pf_tt")

    # plot
    xx = np.array(range(pf_tt.shape[0]))  # shape (nb_test, )
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cc, clf_name in enumerate(clf_lst):
        ax.plot(xx, pf_tt[:, cc], label=clf_name)
    plt.axhline(y=gmean_avg, ls="--", color='black', label="gmean_avg")
    plt.axhline(y=threshold, ls="--", color='red', label="gmean_threshold")
    plt.axvline(x=initial, color='black')
    for i in range(len(peaks)):
        if i % 2 == 0:
            plt.axvline(x=peaks[i], color='red')
            plt.axvline(x=peaks[i + 1], color='red')
            plt.axvspan(peaks[i], peaks[i + 1], alpha=0.2)

    # setup
    ax.set_title(title_info)
    plt.ylim((-0.1, 1.1))
    ax.grid(True)
    ax.legend(loc="best")

    # show/plot
    if not save_plot:
        plt.show()
    else:
        to_dir_png = "../results/rslt.plot/"
        os.makedirs(to_dir_png, exist_ok=True)
        to_flnm = to_dir_png + title_info
        to_flnm = to_flnm.replace(" ", "")
        to_flnm = to_flnm.replace("#", "")
        to_flnm = to_flnm.replace(":", "_")
        to_flnm = to_flnm.replace(",", "_")
        plt.savefig(to_flnm + ".png")
        # print("plot results are saved into %s" % to_dir_png)


def evaluate_periods(initial, peaks, result):
    """
    This method is deprecated.
    This method is used to evaluate the performance on each period, including initial phase, sudden drop phase
        and stable phase.
    The period division step is following the work "TSE2022 Cross-Project Online Just-In-Time Software Defect Prediction".

    Args:
        initial (int): The time step of the end of initial phase.
        peaks (list): The time steps of the start and end of each sudden drop phase.
        result (list): The performance across time steps.

    Returns:
        result_initial (float): The average performance on initial phase.
        result_drop (float): The average performance on sudden drop phase.
        result_stable (float): The average performance on stable phase.
    """
    result_initial = result[0:initial]
    result_drop = []
    result_stable = []
    index = initial
    for i in range(len(peaks)):
        if i % 2 == 0:
            result_stable = np.concatenate((result_stable, result[index:peaks[i]]))
            result_drop = np.concatenate((result_drop, result[peaks[i]:peaks[i + 1]]))
            index = peaks[i + 1]
    result_stable = np.concatenate((result_stable, result[index:]))
    return np.nanmean(result_initial), np.nanmean(result_drop), np.nanmean(result_stable)


def sdp_runs_ground_truth(clf_name="odasc", project_id=0, used_project=[0, 1], nb_para_tune=500, nb_test=5000,
                          wait_days=15,
                          seed_lst=range(20), verbose_int=0, pca_plot=True, just_run=False):
    """
    This method is related to RQ1.
    This method is used to do the "WP+1CP" experiments to get the ground truth that which CPs are instructive to WP.

    Args:
        clf_name (string): The name of base JIT-SDP model and the CP method.
        project_id (int): The index of target project (WP).
        used_project (list): The list of the index of projects which will be used in the JIT-SDP model.
        nb_para_tune (int): The number of WP data used to do parameter tuning.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        wait_days (int): The waiting time in online JIT-SDP.
        seed_lst (list): The list of random seeds used when running.
        verbose_int (int): A number to control the print of running information. "-1" means no print; a larger value
            means deeper and more detailed "print".
        pca_plot (boolean): A parameter to control whether plot the result.
        just_run (boolean): If True, this method will not load or save results, for safety reason.
    """
    clf_name = clf_name.lower()
    if pca_plot:
        x_lim, y_lim = None, None
    project_name = data_id_2name(project_id)
    info_run = "%s: %s, wtt=%d, #seed=%d" % (clf_name, project_name, wait_days, len(seed_lst))
    if just_run:  # revise the print level to the most detailed level
        verbose_int = 3

    """prepare test data stream"""
    report_nb_test = nb_test
    if clf_name == "oza" or clf_name == "oob" or clf_name == "odasc" or clf_name == "orb" or clf_name == "pbsa":
        test_stream = set_test_stream(project_name)
        test_stream.X = np.hstack(
            (test_stream.X, (np.ones(len(test_stream.X)) * project_id).reshape(len(test_stream.X), 1)))
        X_org = test_stream.X[class_data_ind_org().id_X_np]
        # convert fea14 to fea13 and the test data stream
        XX, use_data = real_data_preprocess(X_org)
        yy = test_stream.y[use_data]
        time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
        vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
        target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

        # handle negative nb_test
        n_data_all, n_fea = XX.shape[0], XX.shape[1]  # after fea conversion for jit-sdp
        assert n_fea == 12, "# transformed fea should be 13. Sth. is wrong."
        if nb_test < 0:
            nb_test += n_data_all
            if verbose_int >= 2:
                print("actual nb_test=%d" % nb_test)
        assert nb_para_tune <= nb_test, "nb_pre=%d should be smaller than nb_data=%d" % (nb_para_tune, nb_test)

        norm_data = XX

        # prepare all test samples
        test_data_all = np.hstack((time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
        data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                              n_fea=n_fea)
        """add cross project data"""
        for i in range(23):
            if i in used_project:
                if i != project_id:
                    project_name_cp = data_id_2name(i)
                    test_stream = set_test_stream(project_name_cp)
                    test_stream.X = np.hstack(
                        (test_stream.X, (np.ones(len(test_stream.X)) * i).reshape(len(test_stream.X), 1)))
                    X_org = test_stream.X[class_data_ind_org().id_X_np]
                    # convert fea14 to fea13 and the test data stream
                    XX, use_data = real_data_preprocess(X_org)
                    yy = test_stream.y[use_data]
                    time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
                    vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
                    target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

                    test_data_temp = np.hstack((time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
                    test_data_all = np.vstack([test_data_all, test_data_temp])

                    norm_data = np.vstack([norm_data, XX])
            else:
                if i != project_id:
                    project_name_cp = data_id_2name(i)
                    test_stream = set_test_stream(project_name_cp)
                    test_stream.X = np.hstack(
                        (test_stream.X, (np.ones(len(test_stream.X)) * i).reshape(len(test_stream.X), 1)))
                    X_org = test_stream.X[class_data_ind_org().id_X_np]
                    # convert fea14 to fea13 and the test data stream
                    XX, use_data = real_data_preprocess(X_org)

                    norm_data = np.vstack([norm_data, XX])

        idx = test_data_all[:, 0].argsort()
        test_data_all = test_data_all[idx]

        # fea normalizer based on all test data used for DenStream
        norm_scaler = my_norm_scaler(n_fea=n_fea, norm_name="z_score")
        norm_scaler.my_fit(norm_data)

        # find the index that contains nb_test target data
        count = 0
        nb_pre = nb_para_tune
        last_WP = -1
        for i in range(len(test_data_all)):
            if test_data_all[i][-1] == project_id:
                count = count + 1
                if nb_test == -1:
                    nb_test_WP = i
                last_WP = i
            if count == nb_pre:
                nb_para_tune = i
            if count == nb_test and nb_test != -1:
                nb_test_WP = i
                break
        if nb_test != -1 and count < nb_test:
            nb_test_WP = last_WP

        target_idx = test_data_all[:, data_ind_reset.id_target] == project_id
        target_idx_para = target_idx[:nb_para_tune]
        if with_pretrain == 1:
            target_idx = target_idx[nb_para_tune + 1:nb_test_WP + 1]
        else:
            target_idx = target_idx[1:nb_test_WP + 1]

    data_ptrn = test_data_all[:nb_para_tune]
    X_ptrn, y_ptrn = data_ptrn[data_ind_reset.id_X_np], data_ptrn[:, data_ind_reset.id_y]

    """para-auto DenStream~(lambd, eps, beta, mu)"""
    our_clf_lst = ("odasc", "odasc_aio", "odasc_filtering", "odasc_addcp_adp", "odasc_sbp")  # vip manually maintain
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        X_ptrn_norm = norm_scaler.my_transform(X_ptrn)
        auto_denStream = False
        if auto_denStream:
            eps, mu, beta, lambd = para_denStream(X_ptrn_norm, y_ptrn, nb_repeat=10)
        else:
            eps, mu, beta, lambd = 1.47, 1.57, 0.78, 0.26
            # eps, mu, beta, lambd = 2.09, 2.20, 0.74, 0.125

    """pre-train DenStream"""
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        if "odasc" in clf_name:
            cluster = DenStream(theta_cl=None, lambd=lambd, eps=eps, beta=beta, mu=mu)
            cluster.partial_fit(X_ptrn_norm, y_ptrn)
    else:
        cluster = 0
    """para-auto classifiers~(n_tree, theta_imb, theta_cl)"""
    nb_run = 5  # 30 in systematic exp
    dir_auto_para = dir_rslt_save + data_id_2name(project_id) + "/" + clf_name + "/"
    os.makedirs(dir_auto_para, exist_ok=True)
    auto_name = "%s-para-%dstep-%drun" % (clf_name, nb_pre, nb_run) + ".pkl"
    exist_clf_para = os.path.exists(dir_auto_para + auto_name)

    # if exist_clf_para and not just_run:
    if exist_clf_para:
        para_dict = pkl.load(open(dir_auto_para + auto_name, 'rb'))
        n_tree, theta_imb, theta_cl, p, m, th = \
            para_dict["n_trees"], para_dict["theta_imb"], para_dict["theta_cl"], para_dict["p"], \
                para_dict["m"], para_dict["th"]
    else:
        n_tree, theta_imb, theta_cl, p, m, th = para_classifiers_online(clf_name, data_ptrn, nb_run, wait_days, nb_pre,
                                                                        cluster, project_id)
        # save para_bst
        if not just_run:
            para_dict = {"n_trees": n_tree, "theta_imb": theta_imb, "theta_cl": theta_cl, "p": p,
                         "m": m, "th": th}
            with open(dir_auto_para + auto_name, 'wb') as save_file:
                pkl.dump(para_dict, save_file)
    if verbose_int >= 1:
        print("\n%s, best para:\nn_test=%d, n_tree=%d, theta_imb=%.3f, theta_cl=%.3f" % (
            info_run, nb_test, n_tree, theta_imb, theta_cl))

    # update DenStream para
    if any(clf_name == clf_ for _, clf_ in enumerate(our_clf_lst)):
        cluster.theta_cl = theta_cl

    """main parts across seeds"""
    for ss, seed in enumerate(seed_lst):
        to_dir = uti_rslt_dir_ground_truth(clf_name, project_id, wait_days,
                 n_tree, theta_imb, theta_cl, p, m, th, used_project)
        os.makedirs(to_dir, exist_ok=True)
        # analyze filenames in this dir:
        # find T that is larger than nb_data to save computational cost and load the results.
        exist_result, to_dir = uti_rslt_dir_analyze(to_dir, clf_name, nb_test, seed)
        if not exist_result:
            to_dir += "/T" + str(nb_test) + "/"
            os.makedirs(to_dir, exist_ok=True)
        # file_name-s
        flnm_test = "%s%s.rslt_test.s%d" % (to_dir, clf_name, seed)
        flnm_train = "%s%s.rslt_train.s%d" % (to_dir, clf_name, seed)

        """load or compute"""
        if exist_result and not just_run:
            rslt_test = np.loadtxt(flnm_test)
            rslt_train = np.loadtxt(flnm_train)
            # cutting the results if nb_test_actual < len(rslt_test)
            nb_test_act = nb_test - nb_para_tune
            # if len(rslt_test) > nb_test_act:
            #     rslt_test = rslt_test[:nb_test_act, :]
            #     rslt_train = rslt_train[:nb_test_act, :]
            #     rslt_on_imb = rslt_on_imb[:nb_test_act, :]
            # return 1: rslt_test~(test_time, y_true, y_pred), note: test_time is the commit_time
            # return 2: rslt_train~(commit_time, use_time, yy, y_obv, cl, use_cluster)
            # return 3: rslt_on_imb~(test_time, on_imb_c0, on_imb_c1)
        else:
            """pre-train classifier"""
            cluster_pre = 0
            if clf_name == "oza":
                classifier = OzaBaggingClassifier(HoeffdingTreeClassifier(), n_tree, seed)
                if with_pretrain == 1:
                    classifier.partial_fit(X_ptrn, y_ptrn, label_val)
            elif "oob" in clf_name:
                classifier = OzaBaggingClassifier_OOB(HoeffdingTreeClassifier(), n_tree, seed, theta_imb)
                if with_pretrain == 1:
                    classifier.partial_fit(X_ptrn, y_ptrn, label_val)
            elif "odasc" in clf_name:
                classifier = OzaBaggingClassifier_OOC(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, theta_cl)
                if with_pretrain == 1:
                    cl_ptrn = comp_cl_upper(y_ptrn, y_ptrn)
                    classifier.partial_fit(X_ptrn, y_ptrn, cl_ptrn, label_val)
                    cluster_pre = 1
                else:
                    cluster = DenStream(theta_cl=theta_cl, lambd=lambd, eps=eps, beta=beta, mu=mu)
                    cluster_pre = 0
            elif "pbsa" in clf_name:
                classifier = OzaBaggingClassifier_PBSA(HoeffdingTreeClassifier(), n_tree, seed, theta_imb, p, m, th)
                if with_pretrain == 1:
                    y_ptrn_pre = classifier.predict(X_ptrn)
                    classifier.train_model(X_ptrn, y_ptrn, label_val)
            else:
                raise Exception("Undefined clf_name=%s." % clf_name)

            """[core] test-then-training process:
            at each test step, only one test data arrives, while maybe no or several training data become available
            """
            if with_pretrain == 1:
                nb_test_act = nb_test_WP - nb_para_tune
            else:
                nb_test_act = nb_test_WP
            test_time, test_y_tru, test_y_pre = np.empty(nb_test_act), np.empty(nb_test_act), np.empty(nb_test_act)
            cmt_time_train_lst, use_time_train_lst, y_train_tru_lst, y_train_obv_lst = [], [], [], []
            cl_train_lst, use_cluster_lst = [], []

            if with_pretrain == 1:
                prev_test_time, data_buffer, nb_train_data = data_ptrn[-1, data_ind_reset.id_time], None, 0  # vip
            else:
                prev_test_time, data_buffer, nb_train_data = test_data_all[0, data_ind_reset.id_time], None, 0  # vip
            notadd = 0
            # project_window, selected_project = sbp_initial(range(14), 500)
            for tt in range(nb_test_act):
                # get the test data
                if with_pretrain == 1:
                    test_step = tt + nb_para_tune
                else:
                    test_step = tt
                new_1data = test_data_all[test_step, :].reshape((1, -1))
                test_X = new_1data[data_ind_reset.id_X_np]
                test_time[tt] = new_1data[:, data_ind_reset.id_time]
                test_y_tru[tt] = new_1data[:, data_ind_reset.id_y]

                if new_1data[0, data_ind_reset.id_target] == project_id:
                    target_idx[tt] = True
                    """test: predict with classifiers"""
                    test_y_pre[tt] = classifier.predict(test_X)[0]

                    """get the new train data batch"""
                    data_buffer, new_train_defect, new_train_clean, new_train_unlabeled = \
                        set_train_stream(prev_test_time, test_time[tt], new_1data, data_ind_reset, data_buffer,
                                         wait_days)

                    cmt_time_train = np.concatenate(
                        (new_train_clean[:, data_ind_reset.id_time], new_train_defect[:, data_ind_reset.id_time]))
                    use_time_train = test_time[tt] * np.ones(cmt_time_train.shape)
                    X_train = np.concatenate(
                        (new_train_clean[data_ind_reset.id_X_np], new_train_defect[data_ind_reset.id_X_np]))
                    y_train_obv = np.concatenate(
                        (np.zeros(new_train_clean.shape[0]), np.ones(new_train_defect.shape[0])))
                    y_train_tru = np.concatenate(
                        (new_train_clean[:, data_ind_reset.id_y], new_train_defect[:, data_ind_reset.id_y]))
                    y_train_target = np.concatenate(
                        (new_train_clean[:, data_ind_reset.id_target],
                         new_train_defect[:, data_ind_reset.id_target]))
                    nb_train_data += y_train_obv.shape[0]
                    X_train_weight = np.ones(len(y_train_target))

                    # assign
                    cmt_time_train_lst.extend(cmt_time_train.tolist())
                    use_time_train_lst.extend(use_time_train.tolist())
                    y_train_obv_lst.extend(y_train_obv.tolist())
                    y_train_tru_lst.extend(y_train_tru.tolist())
                    if verbose_int >= 2:
                        print("\ttest_step=%d, y_true=%d, y_pre=%d: %s" % (
                            test_step, test_y_tru[tt], test_y_pre[tt], clf_name))
                        print("\tnew_train: (tru, obv, target)=", y_train_tru, y_train_obv, y_train_target)
                        print("\t\t#acc_train_data = %d" % nb_train_data)

                    """then train: update classifiers and DenStream given new labelled training data"""
                    if y_train_obv.shape[0] > 0:
                        if "oob" in clf_name:
                            classifier.partial_fit(X_train, y_train_obv, label_val, X_train_weight)
                            # assign
                            cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                            use_cluster_lst = cl_train_lst
                        elif "pbsa" in clf_name:
                            classifier.pbsa_flow(X_train, y_train_obv, tt, new_train_unlabeled[data_ind_reset.id_X_np],
                                                 new_train_defect, data_ind_reset, label_val, X_train_weight)
                            # assign
                            cl_train_lst.extend(invalid_val * np.ones(y_train_tru.shape))
                            use_cluster_lst = cl_train_lst
                        elif "odasc" in clf_name:
                            X_train_norm = norm_scaler.my_transform(X_train)
                            if cluster_pre == 0:
                                cluster.partial_fit(X_train_norm, y_train_obv)
                                cluster_pre = 1
                            cl_train, cl_c1_refine, use_cluster_train = \
                                cluster.compute_CLs(X_train_norm, y_train_obv)
                            # update classifier
                            classifier.partial_fit(X_train, y_train_obv, cl_train, label_val, X_train_weight)
                            # update micro-cluster
                            cluster.partial_fit(X_train_norm, y_train_obv, X_train_weight)
                            cluster.revise_cluster_info(X_train_norm, y_train_obv, cl_train)
                            # assign
                            cl_train_lst.extend(cl_train.tolist())
                            use_cluster_lst.extend(use_cluster_train.tolist())
                            # print
                            if verbose_int >= 2:
                                for y_tru_, y_obv_, cl_ in zip(y_train_tru, y_train_obv, cl_train):
                                    print("\t\t\ty_trn_tru=%d, y_trn_obv=%d, cl_est=%.2f" % (y_tru_, y_obv_, cl_))
                            if pca_plot and False:  # manual control
                                info = "test-step=%d, train Fea14_org with y_true" % test_step
                                cluster.plot_cluster(X_train_norm, y_train_tru, pca_hd, info, x_lim, y_lim, True)
                        else:
                            raise Exception("Undefined classifier with clf_name=%s." % clf_name)

                    prev_test_time = test_time[tt]  # update VIP
                else:
                    target_idx[tt] = False
                    test_y_pre[tt] = -1
                    if new_1data.ndim == 1:  # debug
                        new_1data = new_1data.reshape((1, -1))
                    if new_1data[0, data_ind_reset.id_y] == 0:
                        new_1data[0, data_ind_reset.id_vl] = np.inf
                    # set data_buffer, (ts, XX, vl)
                    if data_buffer is None:  # initialize
                        data_buffer = new_1data
                    else:
                        data_buffer = np.vstack((data_buffer, new_1data))
            # only use WP results
            """save returns"""
            test_time = test_time[target_idx]
            test_y_tru = test_y_tru[target_idx]
            test_y_pre = test_y_pre[target_idx]
            # return 1: rslt_test ~ (test_time, y_true, y_pred)
            rslt_test = np.vstack((test_time, test_y_tru, test_y_pre)).T
            # return 2: rslt_train ~ (commit_time, use_time, yy, y_obv, cl, use_cluster)
            train_y_tru, train_y_obv = np.array(y_train_tru_lst), np.array(y_train_obv_lst)
            cmt_time_train, use_time_train = np.array(cmt_time_train_lst), np.array(use_time_train_lst)
            cl_pre, use_cluster = np.array(cl_train_lst), np.array(use_cluster_lst)
            rslt_train = np.vstack((cmt_time_train, use_time_train, train_y_tru, train_y_obv, cl_pre, use_cluster)).T
            # save
            if not just_run:
                info_str = " Note: '%d' means invalidity" % invalid_val
                np.savetxt(flnm_test, rslt_test, fmt='%d\t %d\t %d',
                           header="%test_time, yy, y_pre) " + info_str)
                np.savetxt(flnm_train, rslt_train, fmt='%d %d\t %d\t %d\t %f\t %d',
                           header="%commit_time, use_time, yy, y_obv, CL, use_cluster) " + info_str)

        # demonstration
        if verbose_int >= 1:
            print("\n" + "--" * 20)
            print("%s -- seed=%d: " % (info_run, seed))
            uti_print_pf(rslt_test, rslt_train, with_pretrain)

        """performance evaluation"""
        # cl pf: rmse
        if with_pretrain == 1:
            train_y_tru, train_y_obv, CLs_pre = rslt_train[:, 2], rslt_train[:, 3], rslt_train[:, 4]
            CLs_tru = comp_cl_upper(train_y_tru, train_y_obv)
            cl_rmse_this = uti_eval_cl(CLs_tru, CLs_pre, False)

        # pf eval throughout test steps
        test_y_tru, test_y_pre = rslt_test[:, 1], rslt_test[:, 2]
        pfs_tt_dct = uti_eval_pfs(test_y_tru, test_y_pre)

        # assign
        if ss == 0:  # init
            n_row, n_col = pfs_tt_dct["gmean_tt"].shape[0], len(seed_lst)
            cl_rmse, gmean_tt_ss = np.empty(n_col), np.empty((n_row, n_col))
            r1_tt_ss, r0_tt_ss = np.copy(gmean_tt_ss), np.copy(gmean_tt_ss)
            mcc_tt_ss = np.copy(gmean_tt_ss)
        if with_pretrain == 1:
            cl_rmse[ss] = cl_rmse_this
        gmean_tt_ss[:, ss], r1_tt_ss[:, ss], r0_tt_ss[:, ss], mcc_tt_ss[:, ss] = \
            pfs_tt_dct["gmean_tt"], pfs_tt_dct["recall1_tt"], pfs_tt_dct["recall0_tt"], pfs_tt_dct["mcc_tt"]

    """ave pf across seeds"""
    gmean_tt_ave_ss = np.nanmean(gmean_tt_ss, axis=1)
    r1_tt_ave_ss = np.nanmean(r1_tt_ss, axis=1)
    r0_tt_ave_ss = np.nanmean(r0_tt_ss, axis=1)
    mcc_tt_ave_ss = np.nanmean(mcc_tt_ss, axis=1)
    cl_rmse_ave_ss = np.nanmean(cl_rmse)

    used_project_name = []
    for each in used_project:
        used_project_name.append(data_id_2name(int(each)))
    """save result to csv file"""
    if not just_run:
        to_dir_csv = "../results/rslt.report/"
        os.makedirs(to_dir_csv, exist_ok=True)
        to_flnm_csv = to_dir_csv + "pf_bst_ave%d_p%d_n%d_ground_truth.csv" % (len(seed_lst), nb_pre, report_nb_test)
        with open(to_flnm_csv, "a+") as fh2:
            if not os.path.getsize(to_flnm_csv):  # header
                print("%s,%s,%s,%s,%s,%s,%s,%s" % (
                    "target_project", "used_project", "wait_days", "method",
                    "gmean_bst", "r1_bst", "r0_bst", "mcc_bst"), file=fh2)
            print("%s,%s,%d,%s,%.5f,%.5f,%.5f,%.5f" % (
                project_name, used_project_name, wait_days, clf_name,
                np.nanmean(gmean_tt_ave_ss), np.nanmean(r1_tt_ave_ss), np.nanmean(r0_tt_ave_ss),
                np.nanmean(mcc_tt_ave_ss)), file=fh2)

    # demonstrate
    if verbose_int >= 0:
        print("\n" + "==" * 20)
        print("%s -- ave seeds:" % info_run)
        print("\trmse=%f" % cl_rmse_ave_ss)
        print("\tgmean=%.4f, r1=%.4f, r0=%.4f" % (
            np.nanmean(gmean_tt_ave_ss), np.nanmean(r1_tt_ave_ss), np.nanmean(r0_tt_ave_ss)))
    if pca_plot:
        pfs_tt = np.column_stack((gmean_tt_ave_ss, r1_tt_ave_ss, r0_tt_ave_ss))
        uti_plot_pfs_online(pfs_tt, ["gmean", "r1", "r0"], info_run, save_plot=False)

    # return 1~3: gmean_tt_ave_ss ~(test_step, )
    # return 4: cl_rmse_ave_ss~float
    rslt_return = {"gmean": gmean_tt_ss, "mcc": mcc_tt_ss, "r1": r1_tt_ss, "r0": r0_tt_ss, "gmean_ave": gmean_tt_ave_ss,
                   "mcc_ave": mcc_tt_ave_ss, "r1_ave": r1_tt_ave_ss, "r0_ave": r0_tt_ave_ss}
    return rslt_return


def run_ground_truth(clf, target_id):
    """
    This method is related to RQ1.
    This method is used to run the "WP+1CP" experiments.

    Args:
        clf (string): The name of base JIT-SDP model and the CP method.
        target_id (int): The index of WP.
    """
    sdp_runs_ground_truth(clf_name=clf, project_id=target_id, used_project=[target_id], nb_test=5000, nb_para_tune=1000,
                          seed_lst=range(20), pca_plot=False)
    for i in range(23):
        if i != target_id:
            used_project = [i]
            sdp_runs_ground_truth(clf_name=clf, project_id=target_id, used_project=used_project, nb_test=5000,
                                  nb_para_tune=1000, seed_lst=range(20), pca_plot=False)


def run_ground_truth_reverse(clf, without_target_id):
    """
    This method is deprecated.
    This method is used to run the "AIO-1CP" experiments.

    Args:
        clf (string): The name of base JIT-SDP model and the CP method.
        without_target_id (int): The index of CP which will not be used in AIO.
    """
    used_project = []
    target_id = 15
    for i in range(23):
        if i != without_target_id and i != target_id:
            used_project.append(i)
    sdp_runs_ground_truth(clf_name=clf + "_aio", project_id=target_id, used_project=used_project, nb_test=5000,
                          nb_para_tune=1000, seed_lst=range(20), pca_plot=False)


def prepare_cat_metrics(arr):
    """
    This method is related to similarity calculation.
    This method is used to convert categorical metric into dummy variables.

    Args:
        arr (list): The values of all projects in a metric.

    Returns:
        metrics (list): The dummy variables.
    """
    all_cat = []
    for each in arr:
        if isinstance(each, str):
            temp = each.split(';')
            for te in temp:
                if te not in all_cat:
                    all_cat.append(te)
    metrics = np.zeros([len(arr), len(all_cat)])
    for i in range(metrics.shape[0]):
        if isinstance(arr[i], str):
            temp = arr[i].split(';')
            for j in range(metrics.shape[1]):
                if all_cat[j] in temp:
                    metrics[i][j] = 1
    return metrics


def calculate_D_similarity(file_path):
    """
    This method is related to the similarity calculation.
    This method is used to calculate the distance on each domain-aware metric between projects.

    Args:
        file_path (string): The path of the csv file which save the domain-aware metrics of all projects.

    Returns:
        S (list): A matrix about the distance on each domain-aware metric between projects. The first dimension related
            to the index of projects, each project occupies 23 places, indicating the distance from the other projects.
            The second dimension related to each domain-aware metric.
    """
    df = pd.read_csv(file_path)
    # normalize numeric metrics: starting_time, core_dev
    # starting_time
    start_min = np.nanmin(df["starting_time"])
    start_max = np.nanmax(df["starting_time"])
    start = (np.array(df["starting_time"]) - start_min) / (start_max - start_min)
    start = np.reshape(start, [len(start), -1])
    # core_dev
    core_min = np.nanmin(df["core_dev"])
    core_max = np.nanmax(df["core_dev"])
    core_mean = np.nanmean(df["core_dev"])
    core = (np.array(df["core_dev"]) - core_min) / (core_max - core_min)
    for i in range(len(core)):
        if np.isnan(core[i]):
            core[i] = (core_mean - core_min) / (core_max - core_min)
    core = np.reshape(core, [len(core), -1])

    # prepare categorical metrics: license, language, domain, company, user_interface, owner_type
    license = prepare_cat_metrics(np.array(df["license"]))
    language = prepare_cat_metrics(np.array(df["language"]))
    domain = prepare_cat_metrics(np.array(df["domain"]))
    company = prepare_cat_metrics(np.array(df["company"]))
    user_interface = prepare_cat_metrics(np.array(df["user_interface"]))
    owner_type = prepare_cat_metrics(np.array(df["owner_type"]))

    # prepare binary metrics: use_database, localized, single_pl
    use_database = np.array(df["use_database"])
    localized = np.array(df["localized"])
    single_pl = np.array(df["single_pl"])
    use_database = np.reshape(use_database, [len(use_database), -1])
    localized = np.reshape(localized, [len(localized), -1])
    single_pl = np.reshape(single_pl, [len(single_pl), -1])

    # as a group
    # # combine all metrics
    # all_metrics = np.concatenate([start, core, license, language, domain, company, user_interface,
    #                               owner_type, use_database, localized, single_pl], axis=1)
    #
    # # calculate similarity
    # S = np.zeros([len(all_metrics), len(all_metrics)])
    # for i in range(S.shape[0]):
    #     for j in range(S.shape[1]):
    #         temp = np.absolute(all_metrics[i]-all_metrics[j])
    #         S[i][j] = np.sum(temp)

    # as single
    # combine all metrics
    all_metrics = np.concatenate([start, core, license, language, domain, company, user_interface,
                                  use_database, localized, single_pl], axis=1)

    # calculate similarity
    S = np.zeros([len(all_metrics) * len(all_metrics), 10])
    for i in range(len(all_metrics)):
        for j in range(len(all_metrics)):
            temp = np.absolute(all_metrics[i] - all_metrics[j])
            for k in range(10):
                S[i * len(all_metrics) + j][k] = temp[k]
    df = pd.DataFrame(-S)

    df.to_csv("../results/20230103_D_similarity.csv")

    return -S


def load_D_similarity(target_id):
    """
    This method is related to similarity calculation.
    This method is used to load the calculated distance of domain-aware metrics between projects to save time.

    Args:
        target_id (int): The index of WP.

    Returns:
        all_metrics (list): The distance of domain-aware metrics between WP and CPs.
    """
    wd_df = pd.read_csv("../results/20230103_D_similarity.csv")
    start, core, license, language, domain, company, user_interface, use_database, localized, single_pl = \
        [], [], [], [], [], [], [], [], [], []
    for i in range(23):
        start_across_cp, core_across_cp, license_across_cp, language_across_cp, domain_across_cp, company_across_cp, \
            user_interface_across_cp, use_database_across_cp, localized_across_cp, single_pl_across_cp = \
            [], [], [], [], [], [], [], [], [], []
        for j in range(23):
            index = j + i * 23
            start_across_cp.append(wd_df["0"][index])
            core_across_cp.append(wd_df["1"][index])
            license_across_cp.append(wd_df["2"][index])
            language_across_cp.append(wd_df["3"][index])
            domain_across_cp.append(wd_df["4"][index])
            company_across_cp.append(wd_df["5"][index])
            user_interface_across_cp.append(wd_df["6"][index])
            use_database_across_cp.append(wd_df["7"][index])
            localized_across_cp.append(wd_df["8"][index])
            single_pl_across_cp.append(wd_df["9"][index])
        start.append(start_across_cp)
        core.append(core_across_cp)
        license.append(license_across_cp)
        language.append(language_across_cp)
        domain.append(domain_across_cp)
        company.append(company_across_cp)
        user_interface.append(user_interface_across_cp)
        use_database.append(use_database_across_cp)
        localized.append(localized_across_cp)
        single_pl.append(single_pl_across_cp)

    start = np.array(start)
    core = np.array(core)
    license = np.array(license)
    language = np.array(language)
    domain = np.array(domain)
    company = np.array(company)
    user_interface = np.array(user_interface)
    use_database = np.array(use_database)
    localized = np.array(localized)
    single_pl = np.array(single_pl)
    all_metrics = np.vstack([start[target_id], core[target_id], license[target_id], language[target_id],
                             domain[target_id], company[target_id], user_interface[target_id],
                             use_database[target_id], localized[target_id], single_pl[target_id]])
    return all_metrics


if __name__ == "__main__":
    # para_DenStream()
    # cpps_record(project_id=11, clf_name="oob_cpps", nb_para_tune=1000, nb_test=-1, wait_days=15, verbose_int=0,
    #             use_fs=False)
    # cpps_record(project_id=11, clf_name="pbsa_cpps", nb_para_tune=1000, nb_test=-1, wait_days=15, verbose_int=0,
    #             use_fs=False)
    # cpps_record(project_id=11, clf_name="odasc_cpps", nb_para_tune=1000, nb_test=-1, wait_days=15, verbose_int=0,
    #             use_fs=False)
    # cpps_record(project_id=18, clf_name="odasc_cpps", nb_para_tune=1000, nb_test=-1, wait_days=15, verbose_int=0,
    #             use_fs=False)
    # cpps_record(project_id=18, clf_name="oob_cpps", nb_para_tune=1000, nb_test=-1, wait_days=15, verbose_int=0,
    #             use_fs=False)
    # cpps_record(project_id=18, clf_name="pbsa_cpps", nb_para_tune=1000, nb_test=-1, wait_days=15, verbose_int=0,
    #             use_fs=False)
    # sdp_runs_temp_4_rust("odasc_cpps-0.15", project_id=18, nb_para_tune=1000, nb_test=-1, wait_days=15,
    #                      seed_lst=range(5), verbose_int=1, pca_plot=True, real_threshold=0.15)
    sdp_runs_ground_truth(clf_name="oob", project_id=14, used_project=[13], nb_test=5000, nb_para_tune=1000,
                          seed_lst=range(20), pca_plot=False)

    # for pid in range(23):
    #     cpps_record(project_id=pid, clf_name="oob_cpps", nb_para_tune=1000, nb_test=5000, wait_days=15, verbose_int=0,
    #                 use_fs=False, is_RQ1=True)
    # for pid in range(23):
    #     cpps_record(project_id=pid, clf_name="oob_cpps", nb_para_tune=1000, nb_test=1000, wait_days=15, verbose_int=0,
    #                 use_fs=False, is_RQ1=True)
    # multi_run_more(11, 15, "oob_aio")
    # calculate_D_similarity("../data/metrics_24.csv")
    # sdp_runs_window_similarity('odasc_aio', project_id=6, nb_para_tune=1000, nb_test=5010, wait_days=15,
    #                            seed_lst=range(1), verbose_int=1, pca_plot=False)
    # sdp_study()
    # for i in range(23):
    #     for j in [300, 600, 700, 1000]:
    #         cpps_record(project_id=i, nb_para_tune=1000, nb_test=-1, cpps_size=j, use_fs=True)
    #
    # pass
    # for i in range(23):
    #     run_window_similarity("odasc_aio", i)
    # sdp_runs_ground_truth(clf_name="pbsa", project_id=9, used_project=[9], nb_test=1000, nb_para_tune=1000,
    #                       seed_lst=range(20), pca_plot=False)
