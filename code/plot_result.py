"""
This file is used to plot the online performance of motivation example, RQ2 and RQ3.
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from load_exist_result import get_online_pf, get_online_pf_gt
from data.real_data_stream import data_id_2name
import colorsys
import random
import os
import glob


def plot_online_pf_motivation_pl(clf, project_id, used_project, nb_test, save=False):
    """
    This method is used to plot the online performance of motivation example to show the advantages of project-level
        CP method. You must have the result first.
    Here recommend two example:
        plot_online_pf_motivation("oob", 11, [1, 2], 5000, save=True)
        plot_online_pf_motivation("odasc", 20, [2, 3], 5000, save=True)

    Args:
        clf (string): The name of base JIT-SDP model and the CP method.
        project_id (int): The index of target project (WP).
        used_project (list): The list of the index of projects which will be used in the JIT-SDP model.
        nb_test (int): The number of WP data used for prediction. "-1" means running on all WP data.
        save (boolean): If True, the plotted figure will be saved.
    """
    aio, _, _, _ = get_online_pf(clf + "_aio", project_id, 15, range(20))
    print("get aio")
    filtering, _, _, _ = get_online_pf(clf + "_filtering", project_id, 15, range(20))
    print("get filtering")
    wp, _, _, _ = get_online_pf(clf, project_id, 15, range(20))
    print("get wp")
    aio = aio[:nb_test]
    filtering = filtering[:nb_test]
    wp = wp[:nb_test]
    fig = plt.figure(dpi=150)
    xx = np.array(range(nb_test))
    for each in used_project:
        one_cp, _, _, _ = get_online_pf_gt(clf, project_id, [each], 15, range(20))
        print("get +1cp "+str(each))
        one_cp = one_cp[:nb_test]
        plt.plot(xx, one_cp, label="+"+data_id_2name(each), linewidth=0.75)
    plt.plot(xx, aio, label="aio", linewidth=0.75)
    plt.plot(xx, filtering, label="filtering", linewidth=0.75)
    plt.plot(xx, wp, label="wp", linewidth=0.75)
    plt.legend()
    plt.xlabel("time step")
    plt.ylabel("gmean")
    plt.title("online gmean of "+clf+" on "+data_id_2name(project_id))
    to_dir = "../results/rslt.plot/motivation/pl/" + data_id_2name(project_id) + "_" + str(clf) + "_" + str(nb_test)
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    return 0


def plot_online_pf_rq2(base_clf, project_id, save=False):
    """
    This method is used to plot the online performance of RQ2, including the WP, AIO, Filtering and CroPS.

    Args:
        base_clf (string): The name of base JIT-SDP model.
        project_id (int): The index of target project (WP).
        save (boolean): If True, the plotted figure will be saved.
    """
    project_name = data_id_2name(project_id)
    # b=base, a=aio, f=filtering, c=crops, m=multi-crops
    b_gmean, b_mcc, b_r1, b_r0 = get_online_pf(base_clf, project_id, 15, range(20))
    a_gmean, a_mcc, a_r1, a_r0 = get_online_pf(base_clf+"_aio", project_id, 15, range(20))
    f_gmean, f_mcc, f_r1, f_r0 = get_online_pf(base_clf+"_filtering", project_id, 15, range(20))
    c_gmean, c_mcc, c_r1, c_r0 = get_online_pf(base_clf+"_cpps", project_id, 15, range(20))
    # gmean
    fig = plt.figure(dpi=150)
    xx = np.array(range(len(c_gmean)))
    plt.plot(xx, b_gmean[:-1], label="WP", linewidth=1)
    plt.plot(xx, a_gmean, label="AIO", linewidth=1)
    plt.plot(xx, f_gmean, label="Filtering", linewidth=1)
    plt.plot(xx, c_gmean, label="CroPS", linewidth=1)
    plt.legend()
    to_dir = "../results/rslt.plot/rq2/" + project_name + "_" + base_clf + "_gmean"
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    # mcc
    fig = plt.figure(dpi=150)
    xx = np.array(range(len(c_gmean)))
    plt.plot(xx, b_mcc[:-1], label="WP", linewidth=1)
    plt.plot(xx, a_mcc, label="AIO", linewidth=1)
    plt.plot(xx, f_mcc, label="Filtering", linewidth=1)
    plt.plot(xx, c_mcc, label="CroPS", linewidth=1)
    plt.legend()
    to_dir = "../results/rslt.plot/rq2/" + project_name + "_" + base_clf + "_mcc"
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    # r1
    fig = plt.figure(dpi=150)
    xx = np.array(range(len(c_gmean)))
    plt.plot(xx, b_r1[:-1], label="WP", linewidth=1)
    plt.plot(xx, a_r1, label="AIO", linewidth=1)
    plt.plot(xx, f_r1, label="Filtering", linewidth=1)
    plt.plot(xx, c_r1, label="CroPS", linewidth=1)
    plt.legend()
    to_dir = "../results/rslt.plot/rq2/" + project_name + "_" + base_clf + "_r1"
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    # r0
    fig = plt.figure(dpi=150)
    xx = np.array(range(len(c_gmean)))
    plt.plot(xx, b_r0[:-1], label="WP", linewidth=1)
    plt.plot(xx, a_r0, label="AIO", linewidth=1)
    plt.plot(xx, f_r0, label="Filtering", linewidth=1)
    plt.plot(xx, c_r0, label="CroPS", linewidth=1)
    plt.legend()
    to_dir = "../results/rslt.plot/rq2/" + project_name + "_" + base_clf + "_r0"
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    return 0


def plot_online_pf_rq3(base_clf, project_id, save=False):
    """
    This method is used to plot the online performance of RQ2, including the WP, AIO, Filtering, CroPS and Multi-CroPS.

    Args:
        base_clf (string): The name of base JIT-SDP model.
        project_id (int): The index of target project (WP).
        save (boolean): If True, the plotted figure will be saved.
    """
    project_name = data_id_2name(project_id)
    # b=base, a=aio, f=filtering, c=crops, m=multi-crops
    b_gmean, b_mcc, b_r1, b_r0 = get_online_pf(base_clf, project_id, 15, range(20))
    a_gmean, a_mcc, a_r1, a_r0 = get_online_pf(base_clf + "_aio", project_id, 15, range(20))
    f_gmean, f_mcc, f_r1, f_r0 = get_online_pf(base_clf + "_filtering", project_id, 15, range(20))
    c_gmean, c_mcc, c_r1, c_r0 = get_online_pf(base_clf + "_cpps", project_id, 15, range(20))
    m_gmean, m_mcc, m_r1, m_r0 = get_online_pf(base_clf + "_cpps_ensemble", project_id, 15, range(20))
    # gmean
    fig = plt.figure(dpi=150)
    xx = np.array(range(len(c_gmean)))
    plt.plot(xx, b_gmean[:-1], label="WP", linewidth=1)
    plt.plot(xx, a_gmean, label="AIO", linewidth=1)
    plt.plot(xx, f_gmean, label="Filtering", linewidth=1)
    plt.plot(xx, c_gmean, label="CroPS", linewidth=1)
    plt.plot(xx, m_gmean, label="Multi-CroPS", linewidth=1)
    plt.legend()
    to_dir = "../results/rslt.plot/rq3/" + project_name + "_" + base_clf + "_gmean"
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    # mcc
    fig = plt.figure(dpi=150)
    xx = np.array(range(len(c_gmean)))
    plt.plot(xx, b_mcc[:-1], label="WP", linewidth=1)
    plt.plot(xx, a_mcc, label="AIO", linewidth=1)
    plt.plot(xx, f_mcc, label="Filtering", linewidth=1)
    plt.plot(xx, c_mcc, label="CroPS", linewidth=1)
    plt.plot(xx, m_mcc, label="Multi-CroPS", linewidth=1)
    plt.legend()
    to_dir = "../results/rslt.plot/rq3/" + project_name + "_" + base_clf + "_mcc"
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    # r1
    fig = plt.figure(dpi=150)
    xx = np.array(range(len(c_gmean)))
    plt.plot(xx, b_r1[:-1], label="WP", linewidth=1)
    plt.plot(xx, a_r1, label="AIO", linewidth=1)
    plt.plot(xx, f_r1, label="Filtering", linewidth=1)
    plt.plot(xx, c_r1, label="CroPS", linewidth=1)
    plt.plot(xx, m_r1, label="Multi-CroPS", linewidth=1)
    plt.legend()
    to_dir = "../results/rslt.plot/rq3/" + project_name + "_" + base_clf + "_r1"
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    # r0
    fig = plt.figure(dpi=150)
    xx = np.array(range(len(c_gmean)))
    plt.plot(xx, b_r0[:-1], label="WP", linewidth=1)
    plt.plot(xx, a_r0, label="AIO", linewidth=1)
    plt.plot(xx, f_r0, label="Filtering", linewidth=1)
    plt.plot(xx, c_r0, label="CroPS", linewidth=1)
    plt.plot(xx, m_r0, label="Multi-CroPS", linewidth=1)
    plt.legend()
    to_dir = "../results/rslt.plot/rq3/" + project_name + "_" + base_clf + "_r0"
    if save:
        plt.savefig(to_dir)
    else:
        plt.show()
    plt.close()
    return 0


if __name__ == "__main__":
    plot_online_pf_rq3("odasc", 18, True)
