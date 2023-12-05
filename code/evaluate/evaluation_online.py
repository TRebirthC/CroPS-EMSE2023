"""
This file is used to calculate the online performance of the JIT-SDP model.
"""
import numpy as np
import warnings

# silence the warning
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def Gmean_compute(recall):
    """
    This method is used to calculate the Gmean.

    Args:
        recall (list): The recall of each class.

    Returns:
        Gmean (float): The calculated Gmean.
    """
    Gmean = 1
    for r in recall:
        Gmean = Gmean * r
    Gmean = pow(Gmean, 1/len(recall))
    return Gmean


def avg_acc_compute(recall):
    """
    This method is used to calculate the accuracy.

    Args:
        recall (list): The recall of each class.

    Returns:
        avg_acc (float): The calculated accuracy.
    """
    avg_acc = np.mean(recall)
    return avg_acc


def f1_compute(tr, prec, pos_class=1):
    """
    This method is used to calculate the f1 score.

    Args:
        tr (list): The tpr and tnr.
        prec (list): The precision.
        pos_class (int): The class label of positive class.

    Returns:
        f1_score (float): The calculated f1 score.
    """
    assert pos_class == 1 or pos_class == 0, "current version on 20221201 only works for binary class"
    f1_score = 2 * tr * prec / (tr + prec)
    return f1_score[pos_class]


def mcc_compute(tr, fr, pos_class=1):
    """
    This method is used to calculate the mcc.
    The implementation is based on https://blog.csdn.net/Winnycatty/article/details/82972902
    The undefined MCC that a whole row or column of the confusion matrix M is zero is treated the left column of page 5
    of the paper: Davide Chicco and Giuseppe Jurman. "The advantages of the matthews correlation coefficient (mcc) over
        f1 score and accuracy in binary classification evaluation". BMC Genomics, 21, 01, 2020
    The confusion matrix M is
        M = (tp fn
             fp tn)

    Args:
        tr (list): The tpr and tnr.
        fr (list): The fpr and fnr.
        pos_class (int): The class label of positive class.

    Returns:
        mcc (float): The calculated mcc.
    """
    fenzi = tr[0] * tr[1] - fr[0] * fr[1]
    tp = tr[pos_class]  # defined positive_class is positive
    tn = tr[1 - pos_class]
    fn = fr[pos_class]
    fp = fr[1 - pos_class]
    fenmu = pow((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn), 0.5)
    if fenmu == 0:
        if (tp or tn) and fn == 0 and fp == 0:  # M has only 1 non-0 entry & all are correctly predicted
            mcc = 1
        elif (fp or fn) and tp == 0 and tn == 0:  # M has only 1 non-0 entry & all are incorrectly predicted
            mcc = -1
        else:  # a row or a column of M are zero
            mcc = 0
    else:
        mcc = fenzi/fenmu
    return mcc


def pf_epoch(S, N, P, theta, t, y_t, p_t, pos_class=1):
    """
    This method is used to calculate the performance on a time step.
    Reference:
        Gama, Joao, Raquel Sebastiao, and Pedro Pereira Rodrigues.
        "On evaluating stream learning algorithms." Machine learning 90.3 (2013): 317-346.

    Args:
        S (list): The number of data which is predicted correctly of each class.
        N (list): The number of data of each class.
        P (list): The number of predicted label of each class.
        theta (float): The fading factor used for calculation.
        t (int): The current time step.
        y_t (int): The true label of the current data.
        p_t (int): The prediction label of the current data.
        pos_class (int): The class label of positive class.

    Returns:
        recall (list): The recall of each class.
        gmean (float): The calculated gmean.
        mcc (float): The calculated mcc.
        prec (float): The calculated precisioin.
        f1_score (float): The calculated f1 score.
        ave_acc (float): The calculated accuracy.
    """
    if t == 0:
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t)
        N[t, c] = 1
        P[t, c] = 1
    else:
        S[t, :] = S[t-1, :]
        N[t, :] = N[t-1, :]
        P[t, :] = P[t-1, :]
        c = int(y_t)  # class 0 or 1
        S[t, c] = (y_t == p_t) + theta * (S[t-1, c])
        N[t, c] = 1 + theta * N[t-1, c]
        p = int(p_t)  # the number of predicted positive data
        P[t, p] = 1 + theta * P[t-1, p]

    recall = S[t, :] / N[t, :]

    assert pos_class == 1 or pos_class == 0, "current version on 20221201 only works for binary class"
    tr = recall  # positive class is 1, then tpr = tr[1], tnr = tr[0]
    fr = 1 - tr  # positive class is 1, then fnr = fr[1], fpr = fr[0]
    prec = tr / (tr + np.flip(fr))
    prec = prec[pos_class]
    f1_score = f1_compute(tr, prec)
    mcc = mcc_compute(tr, fr)
    gmean = Gmean_compute(recall)
    ave_acc = avg_acc_compute(recall)
    return recall, gmean, mcc, prec, f1_score, ave_acc


def compute_online_PF(y_tru, y_pre, theta_eval=0.99):
    """
    This method is used to calculated online performance on each time step.

    Args:
        y_tru (int): The true label of the current data.
        y_pre (int): The prediction label of the current data.
        theta_eval (float): The fading factor used for calculation.

    Returns:
        pfs_dct (dict): A dictionary saved several types of performance.
    """
    S = np.zeros([len(y_tru), 2])
    N = np.zeros([len(y_tru), 2])
    P = np.zeros([len(y_tru), 2])
    recalls_tt = np.empty([len(y_tru), 2])
    Gmean_tt = np.empty([len(y_tru), ])
    ave_acc_tt = np.empty([len(y_tru), ])
    prec_tt = np.empty([len(y_tru), ])
    f1_tt = np.empty([len(y_tru), ])
    mcc_tt = np.empty([len(y_tru), ])
    # compute at each test step
    for t in range(len(y_tru)):
        y_t = y_tru[t]
        p_t = y_pre[t]
        recalls_tt[t, :], Gmean_tt[t], mcc_tt[t], prec_tt[t], f1_tt[t], ave_acc_tt[t] \
            = pf_epoch(S, N, P, theta_eval, t, y_t, p_t)
        recall0_tt = recalls_tt[:, 0]
        recall1_tt = recalls_tt[:, 1]
    # assign pfs
    pfs_dct = dict()
    pfs_dct["gmean_tt"] = Gmean_tt
    pfs_dct["recall1_tt"], pfs_dct["recall0_tt"] = recall1_tt, recall0_tt
    pfs_dct["mcc_tt"] = mcc_tt
    pfs_dct["precision_tt"] = prec_tt
    pfs_dct["f1_score_tt"] = f1_tt
    pfs_dct["ave_acc_tt"] = ave_acc_tt
    return pfs_dct


if __name__ == '__main__':
    theta = 0.99
    y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    p = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    pfs_dct = compute_online_PF(y, p, theta)
    recall0 = pfs_dct["recall0_tt"]
    recall1 = pfs_dct["recall1_tt"]
    Gmean = pfs_dct["gmean_tt"]
    mcc = pfs_dct["mcc_tt"]
    precision = pfs_dct["precision_tt"]
    f1_score = pfs_dct["f1_score_tt"]
    avg_acc = pfs_dct["ave_acc_tt"]
    # print
    print('Gmean: ', np.nanmean(Gmean))
    print('mcc: ', np.nanmean(mcc))
    print('recall0: ', np.nanmean(recall0))
    print('recall1: ', np.nanmean(recall1))
    print('precision: ', np.nanmean(precision))
    print('f1_score: ', np.nanmean(f1_score))
    print('avg_acc: ', np.nanmean(avg_acc))
