"""
This file is used to load the exist result to save running time.
"""
from core_code import sdp_runs, uti_eval_pfs, sdp_runs_ground_truth
import os
from data.real_data_stream import data_id_2name
import numpy as np


def load_across_seed(clf_name, project_id, wait_days, to_csv=False, seed_lst=range(20), nb_test=-1):
    """
    This method is used to load the exist result of RQ2 and RQ3 on each seed and time step.

    Args:
        clf_name (string): The name of base JIT-SDP model and the CP method.
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        to_csv (boolean): If True, save the results into csv file.
        seed_lst (list): The random seeds need to be loaded.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.

    Returns:
        gmean_tt_ss (list): The online gmean on each time step and seed.
        mcc_tt_ss (list): The online mcc on each time step and seed.
        r1_tt_ss (list): The online recall1 on each time step and seed.
        r0_tt_ss (list): The online recall0 on each time step and seed.
        rslt_test_ac_seed (list): The result of test data on each time and seed.
    """
    rslt_return, rslt_test_ac_seed = sdp_runs(clf_name=clf_name, project_id=project_id, nb_para_tune=1000, nb_test=nb_test, wait_days=wait_days,
                           seed_lst=seed_lst, verbose_int=-1, pca_plot=False, just_run=False, load_result=True)

    gmean_tt_ss, mcc_tt_ss, r1_tt_ss, r0_tt_ss = \
        rslt_return["gmean"], rslt_return["mcc"], rslt_return["r1"], rslt_return["r0"]
    if to_csv:
        to_dir_csv = "../results/rslt.report/"
        os.makedirs(to_dir_csv, exist_ok=True)
        to_flnm_csv = to_dir_csv + "rslt_across_seed_" + ".csv"
        project_name = data_id_2name(project_id)
        with open(to_flnm_csv, "a+") as fh2:
            if not os.path.getsize(to_flnm_csv):  # header
                print("%s,%s,%s,%s,%s,%s,%s" % (
                    "project", "clf", "seed", "gmean",
                    "mcc", "r1", "r0"), file=fh2)
            for ss, seed in enumerate(seed_lst):
                print("%s,%s,%d,%.5f,%.5f,%.5f,%.5f" % (
                    project_name, clf_name, seed, np.nanmean(gmean_tt_ss[:, ss]),
                    np.nanmean(mcc_tt_ss[:, ss]), np.nanmean(r1_tt_ss[:, ss]), np.nanmean(r0_tt_ss[:, ss]),
                ), file=fh2)
    return gmean_tt_ss, mcc_tt_ss, r1_tt_ss, r0_tt_ss, rslt_test_ac_seed


def load_across_seed_gt(clf_name, project_id, used_project, wait_days, to_csv=False, seed_lst=range(20), nb_test=5000):
    """
    This method is used to load the exist result of ground truth used in RQ1 on each seed and time step.

    Args:
        clf_name (string): The name of base JIT-SDP model and the CP method.
        project_id (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        to_csv (boolean): If True, save the results into csv file.
        seed_lst (list): The random seeds need to be loaded.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.

    Returns:
        gmean_tt_ss (list): The online gmean on each time step and seed.
        mcc_tt_ss (list): The online mcc on each time step and seed.
        r1_tt_ss (list): The online recall1 on each time step and seed.
        r0_tt_ss (list): The online recall0 on each time step and seed.
    """
    rslt_return = sdp_runs_ground_truth(clf_name=clf_name, project_id=project_id, used_project=used_project,
                                                           nb_para_tune=1000, nb_test=nb_test, wait_days=wait_days,
                           seed_lst=seed_lst, verbose_int=-1, pca_plot=False, just_run=False)
    gmean_tt_ss, mcc_tt_ss, r1_tt_ss, r0_tt_ss = \
        rslt_return["gmean"], rslt_return["mcc"], rslt_return["r1"], rslt_return["r0"]
    return gmean_tt_ss, mcc_tt_ss, r1_tt_ss, r0_tt_ss


def get_online_pf(clf, project, wait_days, seed_lst):
    """
    This method is used to load the average online performance across seed on each time step.

    Args:
        clf (string): The name of base JIT-SDP model and the CP method.
        project (int): The index of target project (WP).
        wait_days (int): The waiting time in online JIT-SDP.
        seed_lst (list): The random seeds need to be loaded.

    Returns:
        gmean (list): The average online gmean across seed on each time step.
        mcc (list): The average online mcc across seed on each time step.
        r1 (list): The average online recall1 across seed on each time step.
        r0 (list): The average online recall0 across seed on each time step.
    """
    gmean_tt_ss, mcc_tt_ss, r1_tt_ss, r0_tt_ss, _ = load_across_seed(clf, project, wait_days, False, seed_lst)
    length = len(gmean_tt_ss)
    gmean = np.zeros(length)
    mcc = np.zeros(length)
    r1 = np.zeros(length)
    r0 = np.zeros(length)
    for i in range(length):
        gmean[i] = np.nanmean(gmean_tt_ss[i])
        mcc[i] = np.nanmean(mcc_tt_ss[i])
        r1[i] = np.nanmean(r1_tt_ss[i])
        r0[i] = np.nanmean(r0_tt_ss[i])
    return gmean, mcc, r1, r0


def get_online_pf_gt(clf, project, used_project, wait_days, seed_lst):
    """
    This method is used to load the average online performance of ground truth in RQ1 across seed on each time step.

    Args:
        clf (string): The name of base JIT-SDP model and the CP method.
        project (int): The index of target project (WP).
        used_project (list): The list of the index of projects which will be used in the JIT-SDP model.
        wait_days (int): The waiting time in online JIT-SDP.
        seed_lst (list): The random seeds need to be loaded.

    Returns:
        gmean (list): The average online gmean across seed on each time step.
        mcc (list): The average online mcc across seed on each time step.
        r1 (list): The average online recall1 across seed on each time step.
        r0 (list): The average online recall0 across seed on each time step.
    """
    gmean_tt_ss, mcc_tt_ss, r1_tt_ss, r0_tt_ss = load_across_seed_gt(clf, project, used_project, wait_days, False, seed_lst)
    length = len(gmean_tt_ss)
    gmean = np.zeros(length)
    mcc = np.zeros(length)
    r1 = np.zeros(length)
    r0 = np.zeros(length)
    for i in range(length):
        gmean[i] = np.nanmean(gmean_tt_ss[i])
        mcc[i] = np.nanmean(mcc_tt_ss[i])
        r1[i] = np.nanmean(r1_tt_ss[i])
        r0[i] = np.nanmean(r0_tt_ss[i])
    return gmean, mcc, r1, r0


if __name__ == "__main__":
    # gmean, mcc, r1, r0 = get_online_pf("oob", 0, 15)
    # print(gmean)
    # print(mcc)
    # print(np.nanmean(gmean))
    for i in range(23):
        for base_clf in ["odasc", "oob", "pbsa"]:
            for clf_post in ["", "_aio", "_filtering", "_cpps", "_cpps_ensemble"]:
                clf_name = base_clf+clf_post
                load_across_seed(clf_name, i, 15, to_csv=True, seed_lst=range(20), nb_test=-1)
    # for i in range(23):
    #     for j in ["oob", "odasc", "pbsa"]:
    #         clf_name = j + "_cpps_500_fs"
    #         load_across_seed(clf_name, i, 15, True)
    #         clf_name = j + "_sbp_l1"
    #         load_across_seed(clf_name, i, 15, True)
