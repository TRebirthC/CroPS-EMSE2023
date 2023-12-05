"""
This file is used to do data preprocess.
This file is provided by Liyan Song who proposed ODaSC.
"""
import numpy as np


def real_data_preprocess(Fea14_org):
    """ feature pre-process: [2021-11-23]
    The original feature number is 14, and the converted feature number is reduced to 13.

    We pre-process the input features acc Kamei's 2013 paper.
    The implementation is from "2019 Local vs global models for jit-sdp -- online clustering"

    :para Fea14_org: (n_sample, n_fea=14)
    :return: preprocessed Fea14_org
    """
    _, n_fea = Fea14_org.shape
    assert n_fea == 14, "wrong dim of jit-sdp Fea14_org"
    id_fix, id_ns, id_nd, id_nf, id_entropy = 0, 1, 2, 3, 4
    id_la, id_ld, id_lt, id_ndev, id_age = 5, 6, 7, 8, 9
    id_nuc, id_exp, id_rexp, id_sexp = 10, 11, 12, 13

    """[2021-11-23] remove invalid Fea14_org entries. 
    Otherwise, "log2", due to potential log2(negative_value, will report two warnings as below:
        - RuntimeWarning: divide by zero encountered in log2
        - RuntimeWarning: invalid value encountered in log2 
    We may need to consider more features later on when dealing with more jit-sdp datasets.
    """
    # remove invalid data
    use_data = np.logical_and(Fea14_org[:, id_lt] >= 0,
                              Fea14_org[:, id_age] >= 0,
                              Fea14_org[:, id_rexp] >= 0)

    """calculate and add churn  FSE'16 utils.R"""
    Fea14_org = np.copy(Fea14_org[use_data, :])
    la = Fea14_org[:, id_la]
    ld = Fea14_org[:, id_ld]
    nf = Fea14_org[:, id_nf]
    lt = Fea14_org[:, id_lt]
    lt_ = lt * nf
    lt[lt == 0] = 1
    churn = (la + ld) * lt_ / 2

    """1 deal with multi-collinearity"""
    #  1.1 LA = LA / LT; LD = LD / LT
    select_lt = Fea14_org[:, id_lt] >= 1
    Fea14_org[select_lt, id_la] = Fea14_org[select_lt, id_la] / Fea14_org[select_lt, id_lt]
    Fea14_org[select_lt, id_ld] = Fea14_org[select_lt, id_ld] / Fea14_org[select_lt, id_lt]

    #  1.2 LT = LT / NF; NUC = NUC / NF
    select_nf = Fea14_org[:, id_nf] >= 1
    Fea14_org[select_nf, id_lt] = Fea14_org[select_nf, id_lt] / Fea14_org[select_nf, id_nf]
    Fea14_org[select_nf, id_nuc] = Fea14_org[select_nf, id_nuc] / Fea14_org[select_nf, id_nf]

    # 1.3 entropy = entropy / NF   refer TSE'13 Kamei predUtils.r
    select_nf = Fea14_org[:, id_nf] >= 2
    Fea14_org[select_nf, id_entropy] = Fea14_org[select_nf, id_entropy] / np.log2(Fea14_org[select_nf, id_nf])

    # (1.4) remove ND and REXP
    Fea14_org = Fea14_org[:, np.setdiff1d(range(n_fea), np.array((id_nd, id_rexp)))]

    """2 logarithmic transformation"""
    n_fea_new = Fea14_org.shape[1]
    ids2_ = np.setdiff1d(range(n_fea_new), id_fix)
    Fea14_org[:, ids2_] = Fea14_org[:, ids2_] + 1
    Fea14_org[:, ids2_] = np.log2(Fea14_org[:, ids2_])

    # Fea13 = np.hstack((Fea14_org, np.array([churn]).T))
    Fea12 = Fea14_org

    return Fea12, use_data


# def preprocessing(Fea14_org):
#     # calculate  churn  FSE'16 utils.R
#     la = Fea14_org[:, 4]
#     ld = Fea14_org[:, 5]
#     nf = Fea14_org[:, 2]
#     lt = Fea14_org[:, 6]
#     lt_ = lt*nf
#     lt[lt == 0] = 1
#     churn = (la + ld)*lt_/2
#
#     # (1)deal with multi-collinearity
#     #  1.1 LA = LA / LT; LD = LD / LT
#     select_lt = Fea14_org[:, 6] >= 1
#     Fea14_org[select_lt, 4] = Fea14_org[select_lt, 4] / Fea14_org[select_lt, 6]
#     Fea14_org[select_lt, 5] = Fea14_org[select_lt, 5] / Fea14_org[select_lt, 6]
#     #  1.2 LT = LT / NF; NUC = NUC / NF
#     select_nf = Fea14_org[:, 2] >= 1
#     Fea14_org[select_nf, 6] = Fea14_org[select_nf, 6] / Fea14_org[select_nf, 2]
#     Fea14_org[select_nf, 10] = Fea14_org[select_nf, 10] / Fea14_org[select_nf, 2]
#     # 1.3 entropy = entropy / NF   refer TSE'13 Kamei predUtils.r
#     select_nf = Fea14_org[:, 2] >= 2
#     Fea14_org[select_nf, 3] = Fea14_org[select_nf, 3] / np.log2(Fea14_org[select_nf, 2])
#     # 1.4 remove ND and REXP
#     Fea14_org = Fea14_org[:, (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13)]
#     # (2)logarithmic transformation
#     Fea14_org[:, (2, 3, 4, 5, 7, 8, 9, 10, 11)] = Fea14_org[:, (2, 3, 4, 5, 7, 8, 9, 10, 11)] + 1
#     Fea14_org[:, (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)] = np.log2(Fea14_org[:, (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)])
#     Fea14_org = Fea14_org[:, (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 3, 4)]
#     return np.hstack((Fea14_org, np.array([churn]).T))
