"""
This file is used to do metric selection in further discussion in RQ1.
We use Genatic Algorithm to find the best combination of metrics.
"""
import os

import pandas as pd
import numpy as np
import scipy.stats
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from data.real_data_stream import data_id_2name
from sklearn import metrics


def data_name_to_id(project_name):
    """
    This method is used to convert project name into project id.

    Args:
        project_name (string): The name of the project.

    Returns:
        project_id (int): The index of the project.
    """
    if project_name == "ansible":
        project_id = 0
    elif project_name == "brackets":
        project_id = 1
    elif project_name == "broadleaf":
        project_id = 2
    elif project_name == "camel":
        project_id = 3
    elif project_name == "corefx":
        project_id = 4
    elif project_name == "django":
        project_id = 5
    elif project_name == "elasticsearch":
        project_id = 6
    elif project_name == "fabric":
        project_id = 7
    elif project_name == "googleflutter":
        project_id = 8
    elif project_name == "homebrew":
        project_id = 9
    elif project_name == "jgroups":
        project_id = 10
    elif project_name == "neutron":
        project_id = 11
    elif project_name == "node":
        project_id = 12
    elif project_name == "nova":
        project_id = 13
    elif project_name == "npm":
        project_id = 14
    elif project_name == "panda":
        project_id = 15
    elif project_name == "pytorch":
        project_id = 16
    elif project_name == "rails":
        project_id = 17
    elif project_name == "rust":
        project_id = 18
    elif project_name == "tensorflow":
        project_id = 19
    elif project_name == "tomcat":
        project_id = 20
    elif project_name == "vscode":
        project_id = 21
    elif project_name == "wp-calypso":
        project_id = 22
    return project_id


dir_name = "../results/feature_selection/5000/"


def load_Ds(dir):
    """
    This method is used to load the calculated distance of each domain-aware metric between projects.

    Args:
        dir (string): The directory of the file which save the distance of each domain-aware metric between projects.

    Returns:
        start (list): The distance of metric "Strating Time" between projects.
        core (list): The distance of metric "Core Dev" between projects.
        license (list): The distance of metric "License" between projects.
        language (list): The distance of metric "Language" between projects.
        domain (list): The distance of metric "Domain" between projects.
        company (list): The distance of metric "Company" between projects.
        user_interface (list): The distance of metric "User Interface" between projects.
        use_database (list): The distance of metric "Database" between projects.
        localized (list): The distance of metric "Local" between projects.
        single_pl (list): The distance of metric "Single Pl" between projects.
    """
    wd_df = pd.read_csv(dir)
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
    return start, core, license, language, domain, company, user_interface, use_database, localized, single_pl


def load_gt(clf):
    """
    This method is used to load the ground truth.

    Args:
        clf (string): The name of the base JIT-SDP model.

    Returns:
        gt (list): The ground truth on the base JIT-SDP model.
    """
    gt = np.loadtxt(dir_name+clf+"_class.txt")
    return gt


def load_window(dir):
    """
    This method is used to load the calculated distance of each data based metric between projects.

    Args:
        dir (string): The directory of the file which save the distance of each data based metric between projects.

    Returns:
        defect (list): The distance of metric "Defect ratio" between projects.
        commit (list): The distance of metric "n_commit" between projects.
        median (list): The distance of metric "Median" between projects.
        max (list): The distance of metric "Maximum" between projects.
        std (list): The distance of metric "Standard deviation" between projects.
        sp (list): The distance of metric "Spearman correlation" between projects.
        js (list): The distance of metric "JS divergence" between projects.
    """
    wd_df = pd.read_csv(dir)
    defect = []
    commit = []
    median = []
    max = []
    std = []
    sp = []
    js = []
    for i in range(23):
        defect_method = []
        commit_method = []
        median_method = []
        max_method = []
        std_method = []
        sp_method = []
        js_method = []
        for j in range(1):
            defect_across_cp = []
            commit_across_cp = []
            median_across_cp = []
            max_across_cp = []
            std_across_cp = []
            sp_across_cp = []
            js_across_cp = []
            for k in range(23):
                index = k + j * 23 + i * 23 * 1
                defect_across_cp.append(wd_df["defect_ratio"][index])
                commit_across_cp.append(wd_df["n_commit"][index])
                median_across_cp.append(wd_df["median_feature"][index])
                max_across_cp.append(wd_df["maximum_feature"][index])
                std_across_cp.append(wd_df["std_feature"][index])
                sp_across_cp.append(wd_df["spearman_cor"][index])
                js_across_cp.append(wd_df["js_div"][index])
            defect_method.append(defect_across_cp)
            commit_method.append(commit_across_cp)
            median_method.append(median_across_cp)
            max_method.append(max_across_cp)
            std_method.append(std_across_cp)
            sp_method.append(sp_across_cp)
            js_method.append(js_across_cp)
        defect.append(defect_method)
        commit.append(commit_method)
        median.append(median_method)
        max.append(max_method)
        std.append(std_method)
        sp.append(sp_method)
        js.append(js_method)
    defect = np.array(defect)
    commit = np.array(commit)
    median = np.array(median)
    max = np.array(max)
    std = np.array(std)
    sp = np.array(sp)
    js = np.array(js)
    return defect, commit, median, max, std, sp, js


def calculate_sims(temp_features, used_feature, L, weight):
    """
    This method is used to calculate the similarities between WP and CPs based on the selected metrics.

    Args:
        temp_features (list): The index of the selected metrics.
        used_feature (list): The distance of each metric between WP and CPs.
        L (string): The type of method to calculate the sum of all distance on each metric.
        weight (list): The weight of each metric.

    Returns:
        similarity (list): The calculated similarities based on the selected metrics between WP and CPs.
    """
    similarity = []
    if L == "L1":
        for i in range(23):
            temp_sim = []
            for j in range(len(used_feature)):
                if j in temp_features:
                    temp_sim.append(weight[j] * used_feature[j][i])
            if len(temp_sim) == 0:
                dis = 0
            else:
                if False in np.isnan(temp_sim):
                    dis = np.nanmean(temp_sim)
                else:
                    dis = 0
            if dis < 0:
                dis = dis * -1
            similarity.append(1/(1+dis))
        similarity = np.array(similarity)
        rank = np.argsort(-similarity)
    elif L == "L2":
        for i in range(23):
            temp_sim = 0
            for j in range(len(used_feature)):
                if j in temp_features:
                    temp_sim = temp_sim + weight[j] * (used_feature[j][i] * used_feature[j][i])
            similarity.append(temp_sim)
        similarity = np.array(similarity)
        rank = np.argsort(-similarity)
    return similarity


def feature_selection(start, core, license, language, domain, company, user_interface, use_database, localized,
                      single_pl, defect, commit, median, max, std, sp, js, gt, base_method):
    """
    This method is the core part of metrics selection.

    Args:
        start (list): The distance of metric "Strating Time" between projects.
        core (list): The distance of metric "Core Dev" between projects.
        license (list): The distance of metric "License" between projects.
        language (list): The distance of metric "Language" between projects.
        domain (list): The distance of metric "Domain" between projects.
        company (list): The distance of metric "Company" between projects.
        user_interface (list): The distance of metric "User Interface" between projects.
        use_database (list): The distance of metric "Database" between projects.
        localized (list): The distance of metric "Local" between projects.
        single_pl (list): The distance of metric "Single Pl" between projects.
        defect (list): The distance of metric "Defect ratio" between projects.
        commit (list): The distance of metric "n_commit" between projects.
        median (list): The distance of metric "Median" between projects.
        max (list): The distance of metric "Maximum" between projects.
        std (list): The distance of metric "Standard deviation" between projects.
        sp (list): The distance of metric "Spearman correlation" between projects.
        js (list): The distance of metric "JS divergence" between projects.
        gt (list): The ground truth on the base JIT-SDP model.
        base_method (string): The name of the base JIT-SDP model.
    """
    used_method = 0
    feature_selected_across_tp = []
    saved_performance = []
    for i in range(23):
        used_feature = [start[i], core[i], license[i], language[i], domain[i], company[i], user_interface[i],
                        use_database[i], localized[i], single_pl[i],
                        defect[i][used_method], commit[i][used_method], median[i][used_method],
                        max[i][used_method], std[i][used_method], sp[i][used_method], js[i][used_method]]
        used_feature = np.array(used_feature)

        class FeatureSelection(Problem):

            def __init__(self):
                super().__init__(n_var=17, n_obj=1, n_ieq_constr=0, xl=np.zeros(17, dtype=int),
                                 xu=np.ones(17, dtype=int))

            def _evaluate(self, x, out, *args, **kwargs):
                out_F = []
                for xi in range(len(x)):
                    selected_features = []
                    for xj in range(len(x[xi])):
                        if x[xi][xj] > 0.5:
                            selected_features.append(xj)
                    if len(selected_features) == 0:
                        out_F.append(np.inf)
                    else:
                        temp_sims = calculate_sims(selected_features, used_feature, "L1", np.ones(17, dtype=int))
                        temp_performance = metrics.roc_auc_score(gt[i], temp_sims)
                        out_F.append(-temp_performance)
                out["F"] = out_F

        if np.sum(gt[i]) == 0 or np.sum(gt[i]) == 23:
            feature_selected_across_tp.append(np.zeros(17))
            saved_performance.append(-1)
        else:
            problem = FeatureSelection()
            algorithm = GA(pop_size=100)
            res = minimize(problem,
                           algorithm,
                           ('n_gen', 1000),
                           seed=1,
                           verbose=False)
            feature_selected_across_tp.append(res.X)
            saved_performance.append(-res.F)

    saved_feature_selected_across_tp = np.array(feature_selected_across_tp)
    saved_selected = []
    for i in range(saved_feature_selected_across_tp.shape[0]):
        temp_save = []
        for j in range(saved_feature_selected_across_tp.shape[1]):
            if saved_feature_selected_across_tp[i][j] > 0.5:
                temp_save.append(j)
        saved_selected.append(temp_save)
    saved_selected = np.array(saved_selected)
    save_path1 = dir_name + "selected_features_" + base_method + ".txt"
    np.savetxt(save_path1, saved_selected, fmt='%s')
    save_path2 =  dir_name + "selected_features_" + base_method + "_performance.txt"
    np.savetxt(save_path2, saved_performance, fmt='%s')


def load_selected_features(clf, project_id):
    """
    This method is used to load the result of metrics selection.

    Args:
        clf (string): The name of the base JIT-SDP model.
        project_id (int): The index of the project.

    Returns:
        sf (list): The index of the selected metrics.
    """
    dir = "../results/feature_selection/selected_features_" + clf + ".txt"
    with open(dir) as f:
        a = f.read().splitlines()
    sf_v = []
    for each in a:
        temp_sf = []
        temp = each[1:-1]
        temp = temp.replace(" ", "")
        b = temp.split(',')
        if len(b) > 0 and b[0] != "":
            for every in b:
                temp_sf.append(int(every))
        sf_v.append(temp_sf)
    if len(sf_v[project_id]) < 1:
        return_sf = []
        for i in range(17):
            return_sf.append(i)
        return return_sf
    else:
        return sf_v[project_id]


def run_metric_selection(clf):
    """
    This method is used to run the metrics selection process.

    Args:
        clf (string): The name of the base JIT-SDP model.
    """
    Ds_dir = "../results/20230103_D_similarity.csv"
    start, core, license, language, domain, company, user_interface, use_database, localized, single_pl = load_Ds(
        Ds_dir)
    # gt_dir = "../results/20230215_ground_truth.csv"
    gt = load_gt(clf)
    window_dir = "../results/20230420_window.csv"
    defect, commit, median, max, std, sp, js = load_window(window_dir)
    base_method = clf
    result = feature_selection(start, core, license, language, domain, company, user_interface, use_database,
                               localized, single_pl, defect, commit, median, max, std, sp, js, gt, base_method)


if __name__ == "__main__":
    run_metric_selection("odasc")
    run_metric_selection("oob")
    run_metric_selection("pbsa")
