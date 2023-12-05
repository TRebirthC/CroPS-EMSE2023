# CroPS
The repository contains:

* Python implementation of data-level online cross-project approaches (All-In-One and Filtering) proposed in "An Investigation of Cross-Project Learning in Online Just-In-Time Software Defect Prediction" (ICSE'20) and "Cross-Project Online Just-In-Time Software Defect Prediction" (TSE'22).
* Python implementation of project-level online cross-project approaches (CroPS and Multi-CroPS) proposed in "Effective Online Cross-Project Approach with Project-Level Similarity in Just-In-Time Software Defect Prediction"(EMSE'23).
* Python implementation of state-of-the-art JIT-SDP models (OOB, ODaSC, PBSA) proposed in "Resampling-Based Ensemble Methods
  for Online Class Imbalance Learning"(TKDE'15), "A Novel Data Stream Learning Approach to Tackle
  One-Sided Label Noise From Verification Latency"(IJCNN'22) and "Towards Reliable Online Just-in-time Software
  Defect Prediction"(TSE'22).
* Opensource datasets used for the experiments and hyper-parameter tuning.

## Abstract
Need to be added later

### Authors of the paper
* Cong Teng (12132358@mail.sustech.edu.cn)
* Liyan Song (songly@sustech.edu.cn)
* Xin Yao (xiny@sustech.edu.cn)

### Environment details:
* Python 3.9

#### Package requirement:
* numpy
* scikit-learn
* pandas
* scipy
* matplotlib
* datetime
* pickle
* scikit-multiflow
* detecta
* pymoo
* distinctipy

### Directory introduction
* "code" related to the code of this project.
* "data" related to the dataset used in this project.
* "results" related to the experimental result for this project. Because the result files are too large (more than 200 GB), we only provide tuned parameters to save running time. In addition, we provide ground truth in RQ1 and total results of RQ2 and RQ3.

## To run experiments
### RQ1 (Main)
Because we provide the ground truth and calculated similarities of RQ1, you can just follow the following steps:
* Go to the directory "code"
* Run the experiment file "RQ1.py"

The results will be saved in "results/feature_selection/5000/". "xxx_aucs.txt" related to the calculated AUC value. (xxx is the state-of-the-art JIT-SDP models, i.e. oob, odasc and pbsa)
***
If you want to run experiment to get ground truth, please follow the following steps:
* Go to the experiment file "code/main_multi_runs.py"
* Edit the main function as `run_RQ1()`
* Run the edited main function

The result will be saved into "result/rslt.report/pf_bst_ave20_p1000_n5000_ground_truth.csv".
To use this ground truth, you should follow the following steps to preprocess the csv file:
* Delete duplicate value in the csv file
* Sort the csv file as the ascending order of "method", "target_project" and "used_project"
* Rename the csv file as "gt_5000.csv"
* Put the csv file into "result/feature_selection/5000/"
***
If you want to run experiment to get calculated similarities, please follow the following steps:
* Go to the experiment file "code/core_code.py"
* Edit the main function as
```
for pid in range(23):
    cpps_record(project_id=pid, clf_name="oob_cpps", nb_para_tune=1000, nb_test=5000, wait_days=15, verbose_int=0, use_fs=False, is_RQ1=True)
```
* Run the edited main function

The result will be saved into "result/rslt.report/cpps_similarity.csv".
To use this calculated similarities, you should follow the following steps to preprocess the csv file:
* Delete duplicate value in the csv file
* Sort the csv file as the ascending order of "project"
* Rename the csv file as "cpps_similarity_5000.csv"
* Put the csv file into "result/feature_selection/5000/"
### RQ1 (Metrics Selection)
To run this experiment, we should do metrics selection first.
However, the results are provided in "result/feature_selection/5000/".
If you want to do metrics selection, please follow the following steps:
* Go to the directory "code"
* Run the experiment file "fs_by_ga.py"

The results will be saved into "result/feature_selection/5000/".
"selected_features_xxx.txt" related to the selected metrics for each project.
"selected_features_xxx_performance.txt" related to the AUC value calculated by the selected metrics.
(xxx is the state-of-the-art JIT-SDP models, i.e. oob, odasc and pbsa)


### RQ2 and RQ3
To run the experiment, please follow the following steps:
* Go to the experiment file "code/main_multi_runs.py"
* Edit the main function as `run_RQ23(projects, clfs)` ("projects" and "clfs" should be modified by yourself)
  * "projects" is a list of the project id you want to run. The project id is range from 0 to 22. You can set "projects" as `range(23)` to run all projects
  * "clfs" is a list of the classifiers you want to run. It contains the WP methods `oob`, `odasc` and `pbsa`. It also contains the CP methods `aio`, `filtering`, `cpps`(CroPS) and `cpps_ensemble`(Multi-CroPS). The CP methods should be added after the WP name, such as `oob_aio`
* Run the edited main function

Here give an example to run all experiments on "oob", you can edit it to run what you wanted:
```
run_RQ23(range(23), ["oob", "oob_aio", "oob_filtering", "oob_cpps", "oob_cpps_ensemble"])
```

The results will be saved in "results/rslt.report/pf_bst_ave20_p1000_n-1.csv"

We provide the total results in "results/rslt.report/average_rslt.csv" and results across seed in "results/rslt.report/rslt_across_seed.csv".

The results of double Scott-Knott ESD test is provided in "results/rslt.report/Double Scott-Knott ESD test"