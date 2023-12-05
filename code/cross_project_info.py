"""
This file is used to show the number of available CP data for each WP.
It will show the information at the time steps as 0 (first WP data), 1000 (last WP data for parameter tuning)
    and -1 (last WP data).
"""

import os
import numpy as np
from data.real_data_preprocess import real_data_preprocess
from data.real_data_stream import set_test_stream, data_id_2name, class_data_ind_org, class_data_ind_reset
from core_code import my_norm_scaler, filtering_cross_data
import time


def strftime(timestamp, format_string='%Y-%m-%d %H:%M:%S'):
    """
    This method is used to transform time from timestamp into string.

    Args:
        timestamp (int): The value of timestamp.
        format_string (string): The format of the transformed string.

    Returns:
        time (string): The time as string.
    """
    return time.strftime(format_string, time.localtime(timestamp))


def strptime(string, format_string='%Y-%m-%d %H:%M:%S'):
    """
    This method is used to transform time from string into timestamp.
    Args:
        string (string): The time as string
        format_string (string): The format of the input string.

    Returns:
        time (string): The time as timestamp.
    """
    return time.mktime(time.strptime(string, format_string))


def load_all_data():
    """
    This method is used to load whole data stream of all project in chronological order.

    Returns:
        test_data_all (list): The whole data stream.
        len_project (list): The list of length of each project.
    """
    len_project = []
    project_name = data_id_2name(0)
    test_stream = set_test_stream(project_name)
    test_stream.X = np.hstack((test_stream.X, (np.ones(len(test_stream.X)) * 0).reshape(len(test_stream.X), 1)))
    X_org = test_stream.X[class_data_ind_org().id_X_np]
    XX, use_data = real_data_preprocess(X_org)
    yy = test_stream.y[use_data]
    feature_time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
    vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
    target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

    # handle negative nb_test
    n_data_all, n_fea = XX.shape[0], XX.shape[1]  # after fea conversion for jit-sdp

    # prepare all test samples
    test_data_all = np.hstack((feature_time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
    data_ind_reset = class_data_ind_reset(id_time=0, id_vl=-3, id_y=-2, id_target=-1, id_X_np=np.s_[:, 1:1 + n_fea],
                                          n_fea=n_fea)
    len_project.append(len(yy))
    """add cross project data"""
    for i in range(23):
        if i != 0:
            project_name_cp = data_id_2name(i)
            test_stream = set_test_stream(project_name_cp)
            test_stream.X = np.hstack((test_stream.X, (np.ones(len(test_stream.X)) * i).reshape(len(test_stream.X), 1)))
            X_org = test_stream.X[class_data_ind_org().id_X_np]
            # convert fea14 to fea13 and the test data stream
            XX, use_data = real_data_preprocess(X_org)
            yy = test_stream.y[use_data]
            time = test_stream.X[use_data, class_data_ind_org().id_time][:, np.newaxis]
            vl = test_stream.X[use_data, class_data_ind_org().id_vl][:, np.newaxis]
            target = test_stream.X[use_data, class_data_ind_org().id_target][:, np.newaxis]

            test_data_temp = np.hstack((time, XX, vl, yy, target))  # col=3+13 ~ (time, fea13, vl, yy)
            test_data_all = np.vstack([test_data_all, test_data_temp])

            len_project.append(len(yy))

    idx = test_data_all[:, 0].argsort()
    test_data_all = test_data_all[idx]
    return test_data_all, len_project


def list_to_str(list):
    """
    This method is used to transform list into string.

    Args:
        list (list): A list.

    Returns:
        string (string): The transformed string.
    """
    string = ""
    for i in range(len(list)):
        if i != len(list) - 1:
            string = string + str(list[i]) + ","
        else:
            string = string + str(list[i])
    return string


def cross_info_calculate(all_data, len_project):
    """
    This method is used to calculate the number of available CP data for each WP in different time steps.
    The results will be saved as csv file.

    Args:
        all_data (list): The whole data stream.
        len_project (list): The list of length of each project.
    """
    num_project = len(len_project)
    data_name = []
    start_time = []
    end_time = []
    start_info_lst = []
    para_info_lst = []
    end_info_lst = []
    for i in range(num_project):
        data_name.append(data_id_2name(i))
        start_time.append("")
        end_time.append("")
        start_info_lst.append([])
        para_info_lst.append([])
        end_info_lst.append([])
    num_data = np.zeros(num_project)
    num_defect = np.zeros(num_project)
    defect_ratio_para = np.zeros(num_project)
    defect_ratio_end = np.zeros(num_project)
    for each in all_data:
        target = int(each[-1])
        num_data[target] = num_data[target] + 1
        y = int(each[-2])
        if y == 1:
            num_defect[target] = num_defect[target] + 1
        if num_data[target] == 1:
            data_time = each[0]
            start_time[target] = strftime(data_time)
            start_info = []
            for i in range(num_project):
                start_info.append(num_data[i])
            start_info_lst[target] = start_info
        elif num_data[target] == 1000:
            para_info = []
            for i in range(num_project):
                para_info.append(num_data[i])
            para_info_lst[target] = para_info
            defect_ratio_para[target] = num_defect[target]/num_data[target]
        elif num_data[target] == len_project[target]:
            data_time = each[0]
            end_time[target] = strftime(data_time)
            end_info = []
            for i in range(num_project):
                end_info.append(num_data[i])
            end_info_lst[target] = end_info
            defect_ratio_end[target] = num_defect[target] / num_data[target]

    # save result into csv file
    dir_pre = "../results/rslt.report/cp_info/"
    os.makedirs(dir_pre, exist_ok=True)
    start_title = dir_pre + "cp_start_num.csv"
    para_title = dir_pre + "cp_para_num.csv"
    end_title = dir_pre + "cp_end_num.csv"
    info_title = dir_pre + "cp_info.csv"
    header = list_to_str(data_name)
    # csv for cp_start_num
    with open(start_title, "a+") as fh2:
        if not os.path.getsize(start_title):  # header
            print("project,"+header, file=fh2)
        for i in range(num_project):
            print(data_name[i]+","+list_to_str(start_info_lst[i]), file=fh2)

    # csv for cp_para_num
    with open(para_title, "a+") as fh2:
        if not os.path.getsize(para_title):  # header
            print("project," + header, file=fh2)
        for i in range(num_project):
            print(data_name[i] + "," + list_to_str(para_info_lst[i]), file=fh2)

    # csv for cp_end_num
    with open(end_title, "a+") as fh2:
        if not os.path.getsize(end_title):  # header
            print("project," + header, file=fh2)
        for i in range(num_project):
            print(data_name[i] + "," + list_to_str(end_info_lst[i]), file=fh2)

    # csv for cp_info
    with open(info_title, "a+") as fh2:
        if not os.path.getsize(info_title):  # header
            print("project," + header, file=fh2)
        print("start_time," + list_to_str(start_time), file=fh2)
        print("end_time," + list_to_str(end_time), file=fh2)
        print("defect_ratio_para," + list_to_str(defect_ratio_para), file=fh2)
        print("defect_ratio_end," + list_to_str(defect_ratio_end), file=fh2)

    # return start_info_lst, para_info_lst, end_info_lst, start_time, end_time, data_name


if __name__ == "__main__":
    # para_DenStream()
    all_data, len_project = load_all_data()
    cross_info_calculate(all_data, len_project)
