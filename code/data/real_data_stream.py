"""
This file is used to generate data stream.
This file is provided by Liyan Song who proposed ODaSC.
"""
import random

from skmultiflow.data.file_stream import FileStream
from utility import cvt_day2timestamp
import numpy as np

""" JIT-SDP data:
As we mainly use Hoeffding tree for classification, there is no need to normalize data features.
However, if later on we decide to adopt other ML methods such as distance-based clustering,
we should be careful that feature normalization may be required.
2022-7-28   alter this script
latest updated on   2022/8/9
"""


class class_data_ind_org:
    # Original JIT-SDP has 14 features that will then transform to 12 fea-s acc Kamei's paper.
    # Later, e.g., our codes will auto reset the index info for the 12 transformed fea-s.
    def __init__(self):
        self.nb_fea = 14
        self.nb_inputs = self.nb_fea + 2  # 14-fea + 1 commit time + 1 VL
        self.id_time = 0
        self.id_vl = -2
        self.id_target = -1
        self.id_X_np = np.s_[:, self.id_time + 1:self.id_vl]


class class_data_ind_reset:
    # Manual rest data index after fea conversion. 2022/8/8
    # After fea conversion: col=3+13 ~ (time, fea13, vl, yy)
    def __init__(self, id_time, id_vl, id_y, id_X_np: np, id_target, n_fea=12):
        self.id_time = id_time
        self.id_vl = id_vl
        self.id_y = id_y
        self.id_X_np = id_X_np
        self.n_fea = n_fea
        self.id_target = id_target


def data_id_2name(project_id):
    """2021-12-19. the below projects suffer issues individually as below.
    homebrew        our method arises an error, no available data
    neutron         ood, error happens at the 10,000 steps
    npm             oob, error n_data < 10,000
    spring-integration, <10,000
    """

    if project_id == 0:
        project_name = "ansible"
    elif project_id == 1:
        project_name = "brackets"
    elif project_id == 2:
        project_name = "broadleaf"
    elif project_id == 3:
        project_name = "camel"
    elif project_id == 4:
        project_name = "corefx"
    elif project_id == 5:
        project_name = "django"
    elif project_id == 6:
        project_name = "elasticsearch"
    elif project_id == 7:
        project_name = "fabric"
    elif project_id == 8:
        project_name = "googleflutter"
    elif project_id == 9:
        project_name = "homebrew"
    elif project_id == 10:
        project_name = "jgroups"
    elif project_id == 11:
        project_name = "neutron"
    elif project_id == 12:
        project_name = "node"
    elif project_id == 13:
        project_name = "nova"
    elif project_id == 14:
        project_name = "npm"
    elif project_id == 15:
        project_name = "panda"
    elif project_id == 16:
        project_name = "pytorch"
    elif project_id == 17:
        project_name = "rails"
    elif project_id == 18:
        project_name = "rust"
    elif project_id == 19:
        project_name = "tensorflow"
    elif project_id == 20:
        project_name = "tomcat"
    elif project_id == 21:
        project_name = "vscode"
    elif project_id == 22:
        project_name = "wp-calypso"
    else:
        raise Exception("undefined data id.")
    return project_name


def set_test_stream(project_name):
    """ load_test_stream
    Load the test data stream prepared in MATLAB previously.
    Note that data XX should have been sorted acc commit timestamps in ascending order already.

    param project_name: str, project name
    return: numpy, format - ((ts,XX,vl), y)

    2021-7-14 by Liyan Song
    """

    dir_load_data = "../data/data.inuse/"
    # data_stream: (ts, XX; y; vl)
    data_test_stream = FileStream(dir_load_data + project_name + "_vld_st.csv", target_idx=-2, allow_nan=True)
    # see skmultiflow.data.file_stream.FileStream for how to use the data_test_stream
    return data_test_stream


def set_train_stream(
        prev_test_time, curr_test_time, new_data, data_ind: class_data_ind_reset, data_buffer=None, wait_days=30):
    """ set training stream for jit-sdp
    Inputs:
        new_data: numpy, (n_sample, n_col) where n_col~(time, fea13, vl, yy), see data_ind
        data_ind: class_data_ind_reset
    Log:
        2021-11-1   retains only the "delay-noisy" case for JTI-SDP.
        2022-7-28   insert the class "data_ind"
    """

    # get data index
    id_time, id_vl, id_y, id_X, id_target = data_ind.id_time, data_ind.id_vl, data_ind.id_y, data_ind.id_X_np, data_ind.id_target
    if new_data.ndim == 1:  # debug
        new_data = new_data.reshape((1, -1))

    """store new_data into data_buffer~(time, 13-fea, vl, y)"""
    for dd in range(new_data.shape[0]):
        data_1new = new_data[dd, :].reshape((1, -1))
        # VIP overwrite clean data's VL to np.inf
        if data_1new[0, id_y] == 0:
            data_1new[0, id_vl] = np.inf
        # set data_buffer, (ts, XX, vl)
        if data_buffer is None:  # initialize
            data_buffer = data_1new
        else:
            data_buffer = np.vstack((data_buffer, data_1new))

    """create / update the training sets.
    Consider VL and label noise: if there are labeled training XX becomes available 
    between last time and current time, set the defect and the clean data sets, 
    and maintain the data_buffer carefully."""
    # 1) set train_data_defect and update data_buffer
    is_defect = curr_test_time > (data_buffer[:, id_time] + cvt_day2timestamp(data_buffer[:, id_vl]))
    train_data_defect = data_buffer[is_defect, :]
    # update data_buffer: pop out defect-inducing data
    data_buffer = data_buffer[~is_defect, :]  # (time, 13-fea, vl, y)

    # 2) set train_data_clean and update data_buffer
    wait_days_clean_upp = curr_test_time > data_buffer[:, id_time] + cvt_day2timestamp(wait_days)
    wait_days_clean_low = prev_test_time <= data_buffer[:, id_time] + cvt_day2timestamp(wait_days)
    wait_days_clean = wait_days_clean_low & wait_days_clean_upp
    train_data_clean = data_buffer[wait_days_clean, :]  # possible label noise

    # VIP update data_buffer: pop out the 'real' clean data
    actual_clean = data_buffer[:, id_y] == 0
    # actual_clean = np.isinf(data_buffer[:, ID_vl])  # NOTE clean data's VL should have been assigned to np.inf
    wait_actual_clean = wait_days_clean & actual_clean
    data_buffer = data_buffer[~wait_actual_clean, :]  # (ts, 13-fea, vl, y)

    # 3) set train_data_unlabeled, no need to update data_buffer
    idx_upp_time_unlabeled = data_buffer[:, id_time] < curr_test_time
    lowest_time = max(prev_test_time, curr_test_time - cvt_day2timestamp(wait_days))
    idx_low_time_unlabeled = data_buffer[:, id_time] >= lowest_time
    train_data_unlabeled = data_buffer[idx_upp_time_unlabeled & idx_low_time_unlabeled]

    return data_buffer, train_data_defect, train_data_clean, train_data_unlabeled


def set_train_stream_addcp_adp(
        prev_test_time, curr_test_time, new_data, CP_positive_data, CP_positive_size, data_ind: class_data_ind_reset,
        data_buffer=None, wait_days=30, threshold=0.5, notadd=0):
    """ set training stream for jit-sdp
    Inputs:
        new_data: numpy, (n_sample, n_col) where n_col~(time, fea13, vl, yy), see data_ind
        data_ind: class_data_ind_reset
    Log:
        2021-11-1   retains only the "delay-noisy" case for JTI-SDP.
        2022-7-28   insert the class "data_ind"
    """

    # get data index
    id_time, id_vl, id_y, id_X, id_target = data_ind.id_time, data_ind.id_vl, data_ind.id_y, data_ind.id_X_np, data_ind.id_target
    if new_data.ndim == 1:  # debug
        new_data = new_data.reshape((1, -1))

    """store new_data into data_buffer~(time, 13-fea, vl, y)"""
    for dd in range(new_data.shape[0]):
        data_1new = new_data[dd, :].reshape((1, -1))
        # VIP overwrite clean data's VL to np.inf
        if data_1new[0, id_y] == 0:
            data_1new[0, id_vl] = np.inf
        # set data_buffer, (ts, XX, vl)
        if data_buffer is None:  # initialize
            data_buffer = data_1new
        else:
            data_buffer = np.vstack((data_buffer, data_1new))

    """create and update the training sets
    Consider VL and label noise: if there are labeled training samples becomes available 
    after last time and until current time, set the defect and the clean data sets, 
    and maintain the data_buffer carefully."""
    # 1) set train_data_defect and update data_buffer
    is_defect = curr_test_time > (data_buffer[:, id_time] + cvt_day2timestamp(data_buffer[:, id_vl]))
    train_data_defect = data_buffer[is_defect, :]
    # divide WP and CP defect data
    CP = train_data_defect[:, id_target] == 0
    CP_defect = train_data_defect[CP, :]
    train_data_defect = train_data_defect[~CP, :]
    if len(CP_positive_data) == 0:
        CP_positive_data = CP_defect
    else:
        CP_positive_data = np.concatenate((CP_defect, CP_positive_data))
    if len(CP_positive_data) > CP_positive_size:
        CP_positive_data = CP_positive_data[-CP_positive_size:]

    # update data_buffer: pop out defect-inducing data
    data_buffer = data_buffer[~is_defect, :]  # (time, 13-fea, vl, y, target)

    # 2) set train_data_clean and update data_buffer
    wait_days_clean_upp = curr_test_time > data_buffer[:, id_time] + cvt_day2timestamp(wait_days)
    wait_days_clean_low = prev_test_time <= data_buffer[:, id_time] + cvt_day2timestamp(wait_days)
    wait_days_clean = wait_days_clean_low & wait_days_clean_upp
    WP = data_buffer[:, id_target] == 1
    wait_days_clean = wait_days_clean & WP
    train_data_clean = data_buffer[wait_days_clean, :]  # possible label noise

    # VIP update data_buffer: pop out the 'real' clean data
    actual_clean = data_buffer[:, id_y] == 0
    # actual_clean = np.isinf(data_buffer[:, idx_vl])  # NOTE clean data's VL should have been assigned to np.inf
    # todo 2021-12-6 maybe we can remove the codes that reassign vl of clean to inf
    wait_actual_clean = wait_days_clean & actual_clean
    data_buffer = data_buffer[~wait_actual_clean, :]  # (ts, 13-fea, vl, y, target)

    # 3) set train_data_unlabeled, no need to update data_buffer
    idx_upp_time_unlabeled = data_buffer[:, id_time] < curr_test_time
    lowest_time = max(prev_test_time, curr_test_time - cvt_day2timestamp(wait_days))
    idx_low_time_unlabeled = data_buffer[:, id_time] >= lowest_time
    train_data_unlabeled = data_buffer[idx_upp_time_unlabeled & idx_low_time_unlabeled]

    if notadd == 0:
        len_defect = len(train_data_defect)
        # WP = train_data_clean[:, idx_target] == 1
        len_clean = len(train_data_clean)
        len_CP = len(CP_positive_data)
        need_number = len_clean - len_defect
        start_point = int(len_CP * threshold)
        end_point = start_point + need_number
        if end_point > len_CP:
            end_point = len_CP
            start_point = end_point - need_number
        if len_clean > 0 and need_number > 0:
            if len_CP >= need_number:
                ave_dis = []
                for each in CP_positive_data:
                    difference = train_data_clean[:, 1:13] - each[1:13]
                    distance = np.linalg.norm(difference)
                    ave_dis.append(np.sum(distance))
                index = np.argsort(ave_dis)
                use_CP = index[start_point:end_point]
                not_use_CP = np.concatenate((index[:start_point], index[end_point:]))
                train_data_defect = np.concatenate((train_data_defect, CP_positive_data[use_CP, :]))
                CP_positive_data = CP_positive_data[not_use_CP, :]
            else:
                for i in range(len_CP):
                    train_data_defect = np.append(train_data_defect, CP_positive_data[i])
                CP_positive_data = []

    if train_data_defect.ndim == 1:  # debug
        train_data_defect = train_data_defect.reshape((1, -1))

    return data_buffer, train_data_defect, train_data_clean, train_data_unlabeled, CP_positive_data


def set_train_stream_save(train_data_type, prev_test_time, data_test_1new, y_test_1new,
                          data_buffer=None, wait_days=30):
    """ Training data stream during prev_test_time ~ current test time
    - 2021-11-1 this method would be eventually removed.

    Get the training data stream during prev_test_time and current test time,
    including defect-inducing, clean and unlabeled data points.

    :param train_data_type:
        1) "delay_noise":       consider VL delay and label noise (our learning framework)
        2) "delay_no_noise":    consider VL delay but suppose no label noise (clean set)
            o This setting is non-existent in practice.
            o A baseline to check if it is worthwhile to handle label noise.
        3) "no_delay_no_noise": conventional test-then-train online learning framework
            o A baseline that should perform the best.
            o In this case, wait_step is forced as 0 (reset).
    :param prev_test_time: previous timestamp, used to set the clean and unlabeled datasets
    :param data_test_1new: numpy (ts, XX, vl)
    :param y_test_1new: integer, 0 (for clean) or 1 (for defect-inducing)
    :param data_buffer: numpy (ts, XX, vl)
        o default "None" when data_buffer is first met.
        o updated at each test step
        o used to set the clean and defective training sets
    :param wait_days: integer
        o in the case of "no_delay_no_noise", wait_step should be set as "None"

    :return: train_data_defect
    :return: train_data_clean
    :return: train_data_unlabeled
    :return: data_buffer
        o store data committed until current time T_c that are NOT found defect-inducing, including
            (a) clean data we have not waited enough time to use them as clean labeled data.
            (b) defect-inducing data that have not been found at T_c yet.
        o NOTE data_buffer is not equivalent to train_data_clean.
            o Also contain defect-inducing data that were first mislabeled to "clean" for model training,
            and then true label will be found later.

    Liyan Song: songly@sustech.edu.cn
    2021-7-23
    2021-12-4    I think this method is not used, pls refer to set_train_stream_new() onwards
    """
    # data info
    n_feature = 14  # jit-sdp has 14 input features
    n_input_stream = n_feature + 2  # 14 features + 1 commit timestamps + 1 vl
    idx_time = 0
    idx_vl = -1
    idx_X_np_slice = np.s_[:, idx_time + 1:idx_vl]

    curr_test_time = data_test_1new[0, idx_time]

    # overwrite VL of clean data to Inf
    if y_test_1new == 0:
        data_test_1new[:, idx_vl] = np.inf

    """set data_buffer in the format (ts, XX, vl)"""
    # NOTE the test data is stored in the data_buffer
    if data_buffer is None:  # initialize
        data_buffer = np.array(data_test_1new)
    else:
        data_buffer = np.concatenate((data_buffer, np.array(data_test_1new)))

    if train_data_type.lower() == "delay_noise" or train_data_type.lower() == "delay_no_noise":
        """
        Consider VL delay and label noise
            If new labeled training data becomes available since last test timestamps until the current one,
            set the defect and the clean data set accordingly, and maintain the data_buffer carefully.
        """

        # 1) set train_data_defect and update data_buffer
        is_defect = curr_test_time > (data_buffer[:, idx_time] + cvt_day2timestamp(data_buffer[:, idx_vl]))
        train_data_defect = data_buffer[is_defect, :]
        # update data_buffer: pop out defect-inducing data
        data_buffer = data_buffer[~is_defect, :]  # (ts, XX, vl)

        # 2) set train_data_clean and update data_buffer
        wait_days_clean_upp = curr_test_time > data_buffer[:, idx_time] + cvt_day2timestamp(wait_days)
        wait_days_clean_low = prev_test_time <= data_buffer[:, idx_time] + cvt_day2timestamp(wait_days)
        wait_days_clean = wait_days_clean_low & wait_days_clean_upp
        actual_clean = np.isinf(data_buffer[:, idx_vl])  # NOTE clean data's VL was assigned to np.inf
        wait_actual_clean = wait_days_clean & actual_clean

        if train_data_type.lower() == "delay_noise":  # possible label noise
            train_data_clean = data_buffer[wait_days_clean, :]
        elif train_data_type.lower() == "delay_no_noise":  # no label noise
            train_data_clean = data_buffer[wait_actual_clean, :]
        # VIP update data_buffer: pop out 'real' clean data
        data_buffer = data_buffer[~wait_actual_clean, :]  # (ts, XX, vl)

        # 3) set train_data_unlabeled, no need to update data_buffer
        idx_upp_time_unlabeled = data_buffer[:, idx_time] < curr_test_time
        lowest_time = max(prev_test_time, curr_test_time - cvt_day2timestamp(wait_days))
        idx_low_time_unlabeled = data_buffer[:, idx_time] >= lowest_time
        train_data_unlabeled = data_buffer[idx_upp_time_unlabeled & idx_low_time_unlabeled]

    elif train_data_type.lower() == "no_delay_no_noise":
        # init
        data_buffer = np.empty((0, n_input_stream))
        train_data_defect = np.empty((0, n_input_stream))
        train_data_clean = np.empty((0, n_input_stream))
        train_data_unlabeled = np.empty((0, n_input_stream))

        # set the labeled training set
        if y_test_1new == 1:
            train_data_defect = np.array(data_test_1new)
        elif y_test_1new == 0:
            train_data_clean = np.array(data_test_1new)
        else:
            raise Exception("Data label should be either 1 or 0.")

    else:
        raise Exception("No such train_stream_type.")

    return data_buffer, train_data_defect, train_data_clean, train_data_unlabeled


if __name__ == "__main__":
    data_stream = set_test_stream("bracket")
