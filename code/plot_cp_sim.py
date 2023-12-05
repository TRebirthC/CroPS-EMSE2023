"""
This file is used to plot the similarities between WP and CPs across time.
"""
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.real_data_stream import data_id_2name
from distinctipy import distinctipy


def plot_cp_sim(project_id=0, clf_name="oob_cpps", plot_cp=None, start_time=0, end_time=5000, refine=True, save=False):
    """
    This method is used to plot the histogram of similarities between WP and CPs across time.

    Args:
        project_id (int): The index of target project (WP).
        clf_name (string): The name of base JIT-SDP model and the CP method.
        plot_cp (list): The CPs need to be plotted. If None, it will plot all CPs.
        start_time (int): The start time step of WP to plot.
        end_time (int): The end time step of WP to plot.
        refine (boolean): If True, for the CPs which are not exist at the time step, their similairties will be set as
            nan.
        save (boolean): If True, the plotted figure will be saved.
    """
    data = pd.read_csv("../results/rslt.report/cpps_similarity.csv")
    project_name = data_id_2name(project_id)
    selected = data["project"] == project_name
    data = data[selected]
    selected = data["clf_name"] == clf_name
    data = data[selected]
    data.reset_index(drop=True, inplace=True)
    time_step = np.array(data["timestep"])
    cp_sim = np.array([np.array(data["sim0"]), np.array(data["sim1"]), np.array(data["sim2"]), np.array(data["sim3"]),
                       np.array(data["sim4"]), np.array(data["sim5"]), np.array(data["sim6"]), np.array(data["sim7"]),
                       np.array(data["sim8"]), np.array(data["sim9"]), np.array(data["sim10"]), np.array(data["sim11"]),
                       np.array(data["sim12"]), np.array(data["sim13"]), np.array(data["sim14"]),
                       np.array(data["sim15"]),
                       np.array(data["sim16"]), np.array(data["sim17"]), np.array(data["sim18"]),
                       np.array(data["sim19"]),
                       np.array(data["sim20"]), np.array(data["sim21"]), np.array(data["sim22"])])
    cp_len = np.array([np.array(data["len0"]), np.array(data["len1"]), np.array(data["len2"]), np.array(data["len3"]),
                       np.array(data["len4"]), np.array(data["len5"]), np.array(data["len6"]), np.array(data["len7"]),
                       np.array(data["len8"]), np.array(data["len9"]), np.array(data["len10"]), np.array(data["len11"]),
                       np.array(data["len12"]), np.array(data["len13"]), np.array(data["len14"]),
                       np.array(data["len15"]),
                       np.array(data["len16"]), np.array(data["len17"]), np.array(data["len18"]),
                       np.array(data["len19"]),
                       np.array(data["len20"]), np.array(data["len21"]), np.array(data["len22"])])
    update_period = data["update_period"][0]
    window_size = data["window_size"][0]
    threshold = data["threshold"][0]
    assert len(time_step) == len(cp_sim[0]) == len(cp_len[0]), "wrong"
    if start_time < time_step[0]:
        start_time = time_step[0]
    if end_time > time_step[-1]:
        end_time = time_step[-1]
    real_start_time = -1
    real_end_time = -1
    for i in range(len(time_step)):
        if time_step[i] >= start_time and real_start_time == -1:
            start_index = i
            real_start_time = time_step[i]
        if time_step[i] >= end_time and real_end_time == -1:
            end_index = i
            real_end_time = time_step[i]

    max_sim = 0
    if plot_cp is None:
        plot_cp = []
        for i in range(23):
            for j in range(start_index, end_index+1):
                if cp_sim[i][j] >= threshold and i not in plot_cp and cp_len[i][j] > 0:
                    plot_cp.append(i)
                    if cp_sim[i][j] != 1 and cp_sim[i][j] > max_sim:
                        max_sim = cp_sim[i][j]
    if refine:
        for i in range(len(time_step)):
            for j in plot_cp:
                if cp_len[j][i] == 0:
                    cp_sim[j][i] = np.nan

    valid_index = []
    for i in range(start_index, end_index+1):
        for each in plot_cp:
            if cp_sim[each][i] != 1 and cp_sim[each][i] > threshold and i not in valid_index:
                valid_index.append(i)


    x = np.arange(len(valid_index))
    total_width, n = 0.8, len(plot_cp)-1
    colors = distinctipy.get_colors(n)
    no_selected_cp = "no selected cp in this time step range (" + str(real_start_time)+","+str(real_end_time)+")"
    if n == 0:
        raise Exception(no_selected_cp)

    width = total_width / n

    x = x - (total_width - width) / 2

    index = 0
    for i in range(len(plot_cp)):
        if plot_cp[i] != project_id:
            plot_sim = []
            for each in valid_index:
                plot_sim.append(cp_sim[plot_cp[i]][each])
            plt.bar(x+index*width, plot_sim, width=width,
                    label=data_id_2name(plot_cp[i]), color=colors[index])
            index = index+1
    plt.axhline(threshold, ls='--', c='r', lw=1)
    plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0., fontsize=12)
    plt.title("cross project similarities on "+data_id_2name(project_id), fontsize=18)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.73)
    plt.xlabel("time step", fontsize=18)
    x_labels = []
    for each in valid_index:
        x_labels.append(str(time_step[each]))
    plt.xticks(x+(n-1)/2*width, labels=x_labels, fontsize=12)
    y = np.arange(threshold-0.05, threshold+0.05, 0.025)
    plt.yticks(y, fontsize=18)
    plt.ylabel("similarity", fontsize=18)
    plt.ylim(threshold-0.05, threshold+0.05)
    if save:
        to_dir = "../results/rslt.plot/similarity"
        os.makedirs(to_dir, exist_ok=True)
        filename = to_dir+"/"+project_name+"_"+clf_name+"_"+str(start_time)+","+str(end_time)
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def plot_cp_sim_line(project_id=0, clf_name="oob_cpps", plot_cp=None, start_time=0, end_time=5000, refine=True, save=False):
    """
    This method is used to plot the curve graph of similarities between WP and CPs across time.

    Args:
        project_id (int): The index of target project (WP).
        clf_name (string): The name of base JIT-SDP model and the CP method.
        plot_cp (list): The CPs need to be plotted. If None, it will plot all CPs.
        start_time (int): The start time step of WP to plot.
        end_time (int): The end time step of WP to plot.
        refine (boolean): If True, for the CPs which are not exist at the time step, their similairties will be set as
            nan.
        save (boolean): If True, the plotted figure will be saved.
    """
    data = pd.read_csv("../results/rslt.report/cpps_similarity.csv")
    project_name = data_id_2name(project_id)
    selected = data["project"] == project_name
    data = data[selected]
    selected = data["clf_name"] == clf_name
    data = data[selected]
    data.reset_index(drop=True, inplace=True)
    time_step = np.array(data["timestep"])
    cp_sim = np.array([np.array(data["sim0"]), np.array(data["sim1"]), np.array(data["sim2"]), np.array(data["sim3"]),
                       np.array(data["sim4"]), np.array(data["sim5"]), np.array(data["sim6"]), np.array(data["sim7"]),
                       np.array(data["sim8"]), np.array(data["sim9"]), np.array(data["sim10"]), np.array(data["sim11"]),
                       np.array(data["sim12"]), np.array(data["sim13"]), np.array(data["sim14"]),
                       np.array(data["sim15"]),
                       np.array(data["sim16"]), np.array(data["sim17"]), np.array(data["sim18"]),
                       np.array(data["sim19"]),
                       np.array(data["sim20"]), np.array(data["sim21"]), np.array(data["sim22"])])
    cp_len = np.array([np.array(data["len0"]), np.array(data["len1"]), np.array(data["len2"]), np.array(data["len3"]),
                       np.array(data["len4"]), np.array(data["len5"]), np.array(data["len6"]), np.array(data["len7"]),
                       np.array(data["len8"]), np.array(data["len9"]), np.array(data["len10"]), np.array(data["len11"]),
                       np.array(data["len12"]), np.array(data["len13"]), np.array(data["len14"]),
                       np.array(data["len15"]),
                       np.array(data["len16"]), np.array(data["len17"]), np.array(data["len18"]),
                       np.array(data["len19"]),
                       np.array(data["len20"]), np.array(data["len21"]), np.array(data["len22"])])
    update_period = data["update_period"][0]
    window_size = data["window_size"][0]
    threshold = data["threshold"][0]
    assert len(time_step) == len(cp_sim[0]) == len(cp_len[0]), "wrong"
    if start_time < time_step[0]:
        start_time = time_step[0]
    if end_time > time_step[-1]:
        end_time = time_step[-1]
    real_start_time = -1
    real_end_time = -1
    for i in range(len(time_step)):
        if time_step[i] >= start_time and real_start_time == -1:
            start_index = i
            real_start_time = time_step[i]
        if time_step[i] >= end_time and real_end_time == -1:
            end_index = i
            real_end_time = time_step[i]

    max_sim = 0
    if plot_cp is None:
        plot_cp = []
        for i in range(23):
            plot_cp.append(i)

    if refine:
        for i in range(len(time_step)):
            for j in plot_cp:
                if cp_len[j][i] == 0:
                    cp_sim[j][i] = np.nan

    x = np.arange(end_index-start_index+1)
    total_width, n = 0.8, len(plot_cp)
    colors = distinctipy.get_colors(n)
    no_selected_cp = "no selected cp in this time step range (" + str(real_start_time)+","+str(real_end_time)+")"
    if n == 0:
        raise Exception(no_selected_cp)

    for i in range(len(plot_cp)):
        if plot_cp[i] != project_id:
            plt.plot(x, cp_sim[plot_cp[i]][start_index:end_index+1], label=data_id_2name(plot_cp[i]), color=colors[i], linewidth=0.5)
    plt.axhline(threshold, ls='--', c='r', lw=1)
    plt.axhline(0.15, ls='--', c='r', lw=1)

    plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0., fontsize=12)
    plt.title("cross project similarities on "+data_id_2name(project_id), fontsize=18)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.73)
    plt.xlabel("time step", fontsize=18)
    x_labels = []
    for i in range(5+1):
        x_labels.append(real_start_time+i*int((real_end_time-real_start_time)/5))
    interval = int((real_end_time-real_start_time)/5/update_period)
    x_plot = []
    for i in range(len(x)):
        if x[i] % interval == 0:
            x_plot.append(i)
    plt.xticks(x_plot, labels=x_labels, fontsize=12)
    plt.ylabel("similarity", fontsize=18)
    plt.ylim(0.1, 0.25)

    if save:
        to_dir = "../results/rslt.plot/similarity"
        os.makedirs(to_dir, exist_ok=True)
        filename = to_dir+"/"+project_name+"_"+clf_name+"_"+str(start_time)+","+str(end_time)+"_line"
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()
    return 0


def plot_cp_sim_group(project_id, clf, time_steps, divide_step):
    """
    This method is used to divide the time steps of WP to plot a group of figures.
    It is mainly used when the WP data is too long to see the picture clearly.

    Args:
        project_id (int): The index of target project (WP).
        clf_name (string): The name of base JIT-SDP model and the CP method.
        time_steps (list): The start and end time steps of WP to plot.
        divide_step (int): The length of time steps for each sub figure.
    """
    groups = (time_steps[1] - time_steps[0]) / divide_step
    for i in range(math.ceil(groups)):
        start_time = time_steps[0] + i * divide_step
        end_time = time_steps[0] + (i + 1) * divide_step
        plot_cp_sim(project_id, clf, None, start_time, end_time, True, True)


if __name__ == "__main__":
    plot_cp_sim_line(18, "odasc_cpps", [0, 5, 11, 13, 15], 0, 70000, True, True)
