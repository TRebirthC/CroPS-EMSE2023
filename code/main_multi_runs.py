"""
This file is used to run experiment in multithreading to save time.
"""
import multiprocessing
from itertools import product
from core_code import multi_run, multi_run_more, run_ground_truth, run_window_similarity, run_ground_truth_reverse, cpps_record
from fs_by_ga import run_metric_selection


def run_RQ23(project_id_lst=range(23), clf_lst=["odasc", "oob", "pbsa"]):
    """
    This method is used to run the experiment of RQ2 and RQ3.

    For official experiment, you can select base models with CP method to run. We use "_" to link the base model and
        CP method.
    For base models, "odasc" related to ODaSC, "oob" related to OOB, "pbsa" related to PBSA.
    For CP methods, "aio" related to AIO, "filtering" related to Filtering, "cpps" related to CroPS, "cpps_ensemble"
        related to Multi-CroPS.
    For example, "odasc_aio" means use ODaSC as base model with AIO as CP method, "oob_cpps_ensemble" means use OOB
        as base model with Multi-CroPS as CP method.

    Note:
    The experiment of CP method is based on the base JIT-SDP models, so you should run the experiment of base JIT-SDP
        models to do parameter tuning. For example, set the clf_lst as ["odasc", "oob", "pbsa"] to run the base
        JIT-SDP models to do parameter tuning. However, if you have already run run_RQ1, you can skip this step.
    The experiment of Multi-CroPS is based on CroPS, so you should run the experiment of CroPS on the same base model
        to do parameter tuning first.

    Args:
        project_id_lst (list): The index of projects shouble be running.
        clf_lst (list): The name of the base JIT-SDP models with CP method.
    """
    wait_days = 15
    if len(clf_lst) == 1:
        clf = clf_lst[0]
        runs_enu = product(project_id_lst)
        processes = []
        for rr, run in enumerate(runs_enu):
            project = run[0]
            process = multiprocessing.Process(
                target=multi_run_more, args=(project, wait_days, clf))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
    else:
        runs_enu = product(clf_lst, project_id_lst)
        processes = []
        for rr, run in enumerate(runs_enu):
            clf, project = run[0], run[1]
            process = multiprocessing.Process(
                target=multi_run_more, args=(project, wait_days, clf))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()


def run_RQ1(project_id_lst=range(23), clf_lst=["odasc", "oob", "pbsa"]):
    """
    This method is used to run "WP+1CP" experiments for RQ1.
    After running it, you can get the ground truth used for RQ1.

    Args:
        project_id_lst (list): The index of projects shouble be running.
        clf_lst (list): The name of the base JIT-SDP models with CP method.
    """
    wait_days = 15
    if len(clf_lst) == 1:
        clf = clf_lst[0]
        runs_enu = product(project_id_lst)
        processes = []
        for rr, run in enumerate(runs_enu):
            project = run[0]
            process = multiprocessing.Process(
                target=run_ground_truth, args=(clf, project))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
    else:
        runs_enu = product(clf_lst, project_id_lst)
        processes = []
        for rr, run in enumerate(runs_enu):
            clf, project = run[0], run[1]
            process = multiprocessing.Process(
                target=run_ground_truth, args=(clf, project))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()


if __name__ == '__main__':
    run_RQ1()


