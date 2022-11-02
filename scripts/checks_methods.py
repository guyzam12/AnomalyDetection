import pandas as pd
import numpy as np
import torch as th
from sklearn import metrics
from scipy import stats

DIST_SMALL,DIST_BIG = 0,50
LOF_SMALL,LOF_BIG = 10,15
ISOLATION_SMALL,ISOLATION_BIG = 0,1
LOSS_SMALL,LOSS_BIG = 0,1

def normalize(input):
    if isinstance(input, np.ndarray):
        return (input - np.min(input)) / (np.max(input) - np.min(input))
    return (input - min(input)) / (max(input) - min(input))

def auc(x,labels):
    if sum(x.values) != 0:
        fpr, tpr, thresholds = metrics.roc_curve(labels, -x, pos_label=0)
        auc = metrics.auc(fpr, tpr)
        return auc
    return 0

df_results = pd.read_csv("/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/anomaly_scores/breast_cancer_test.csv")
#df_results = pd.read_csv("/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/train_test/breast_cancer_train_512bit_500000steps_summary.csv")
labels = df_results['Labels']
temp_df_results = df_results.filter(regex='{}|{}|{}|{}'.format('Dist[0-9]','LOF[0-9]','Loss','Isolation'))
methods_list = temp_df_results.columns.values
methods_list_new = []
for i in range(DIST_SMALL,DIST_BIG):
    methods_list_new.append("Dist{}nn".format(i))
for i in range(LOF_SMALL,LOF_BIG):
    methods_list_new.append("LOF{}nn".format(i))
for i in range(ISOLATION_SMALL,ISOLATION_BIG):
    methods_list_new.append("Isolation")
for i in range(LOSS_SMALL,LOSS_BIG):
    methods_list_new.append("Loss")
auc_list = []
lof_auc, lof_count = 0,0
dist_auc, dist_count = 0,0
mean_dist, mean_lof = 0,0
for method in methods_list_new:
    loss = df_results[method]
    if method == "Loss":
        loss_auc = auc(loss,labels)
    if  method == "Isolation":
        isolation_auc = auc(loss,labels)
    if "LOF" in method:
        lof_auc += auc(loss,labels)
        lof_count += 1
        mean_lof += loss
    if "Dist" in method:
        dist_auc += auc(loss,labels)
        dist_count += 1
        mean_dist += loss

lof_auc /= lof_count
mean_lof /= lof_count
dist_auc /= dist_count
mean_dist /= dist_count
mean_dist_auc = auc(mean_dist,labels)
mean_lof_auc = auc(mean_lof,labels)
print("AUC of mean KNN: {}".format(mean_dist_auc))
print("Mean AUC of all KNN: {}".format(dist_auc))
print("AUC of mean LOF: {}".format(mean_lof_auc))
print("Mean AUC of all LOF: {}".format(lof_auc))
print("Iforest AUC: {}".format(isolation_auc))
print("Loss AUC: {}".format(loss_auc))
df_output = pd.DataFrame(columns=["method","AUC"])
row = {"method":"average knn","AUC": mean_dist_auc}
df_output = df_output.append(row,ignore_index=True)
row = {"method":"average knn auc","AUC": dist_auc}
df_output = df_output.append(row,ignore_index=True)
row = {"method":"average lof", "AUC": mean_lof_auc}
df_output = df_output.append(row,ignore_index=True)
row = {"method":"average lof auc", "AUC": lof_auc}
df_output = df_output.append(row,ignore_index=True)
row = {"method":"iforest auc","AUC": isolation_auc}
df_output = df_output.append(row,ignore_index=True)
row = {"method":"loss auc","AUC": loss_auc}
df_output = df_output.append(row,ignore_index=True)
print("/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/train_test/auc_results.csv")
df_output.to_csv('/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/train_test/auc_results.csv')




