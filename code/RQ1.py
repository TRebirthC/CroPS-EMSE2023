"""
This file is related to RQ1.
It is used to evaluate the effectiveness of calculated similarities between projects.
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from data.real_data_stream import data_id_2name


dir_name = "../results/feature_selection/5000/"
file_name = "gt_5000.csv"


def gt_to_2_class(file_name):
    """
    This method is used to transform ground truth as binary classification task about how to divide instructive CPs.

    Args:
        file_name (string): The directory of the file save the ground truth.
    """
    data = pd.read_csv(file_name)
    for each in ["odasc", "oob", "pbsa"]:
        is_clf = data["method"] == each
        data_clf = data[is_clf]
        class_clf = []
        for pid in range(23):
            project_name = data_id_2name(pid)
            is_project = data_clf["target_project"] == project_name
            data_project = data_clf[is_project]
            gmean_arr = np.array(data_project["gmean_bst"])
            base_gmean = gmean_arr[pid]
            class_project = []
            for i in range(23):
                if gmean_arr[i] >= base_gmean:
                    class_project.append(1)
                else:
                    class_project.append(0)
            class_clf.append(class_project)
        class_clf = np.array(class_clf)
        np.savetxt(dir_name + each + "_class.txt", class_clf, fmt='%d')
    return 0


def load_similarity(file_name):
    """
    This method is used to load the calculated similarities between projects.

    Args:
        file_name (string): The directory of the file save the calculated similarities.

    Returns:
        sims (list): The calculated similarities between projects.
    """
    sim = pd.read_csv(file_name)
    sims = []
    for i in range(23):
        a = np.array(sim.loc[i][2:25])
        sims.append(a)
    sims = np.array(sims)
    return sims


def evaluate_metrics(sims, clf):
    """
    This method is used to evaluate the effectiveness of calculated similarities.

    Args:
        sims (list): The calculated similarities between projects.
        clf (string): The name of the base JIT-SDP model.

    Returns:
        aucs (float): The AUC of ROC, which is used to evaluate the effectiveness of calculated similarities.
    """
    aucs = []
    class_clf = np.loadtxt(dir_name+clf+"_class.txt")
    for i in range(23):
        if np.sum(class_clf[i]) == 0 or np.sum(class_clf[i]) == 23:
            aucs.append(-1)
        else:
            roc = metrics.roc_auc_score(class_clf[i], sims[i])
            aucs.append(roc)
    aucs = np.array(aucs)
    np.savetxt(dir_name+clf+"_aucs.txt", aucs, fmt='%.3f')

    return aucs


if __name__ == '__main__':
    gt_to_2_class(dir_name+"gt_5000.csv")
    sims = load_similarity(dir_name+"cpps_similarity_5000.csv")
    aucs = evaluate_metrics(sims, "pbsa")
    aucs = evaluate_metrics(sims, "oob")
    aucs = evaluate_metrics(sims, "odasc")
