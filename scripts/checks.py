import pandas as pd
import numpy as np
import torch as th
from sklearn import metrics


def normalize(input):
    if isinstance(input, np.ndarray):
        return (input - np.min(input)) / (np.max(input) - np.min(input))
    return (input - min(input)) / (max(input) - min(input))

def auc(x,labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, -x, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    return (auc)

df = pd.read_csv('/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/satellite_512bit_150000steps_summary.csv')
loss = df['Loss']
labels = df['Labels']
min_mse_dist = 1000000
min_mse_lof = 1000000
output_mse = 0
output_lof = 0
auc_dist_org_ar, auc_dist_ar = [], []
dist_org_ar, dist_mse_ar, dist_mse_total_ar = [], [], []
auc_lof_org_ar, auc_lof_ar = [], []
lof_org_ar, lof_mse_ar, lof_mse_total_ar = [], [], []

for i in range(0,49):
    dist_name = "Dist{}nn".format(i)
    distance = df[dist_name]
    auc_dist_org_ar.append(auc(distance,labels))
    dist_org_ar.append(distance)
    params = np.polyfit(loss,distance,deg=1)
    estimation = params[0]*loss+params[1]
    dist_mse = (np.abs(estimation - distance))
    dist_mse_ar.append(dist_mse)
    auc_dist = auc(dist_mse,labels)
    auc_dist_ar.append(auc_dist)
    tot_dist_mse = sum(dist_mse)
    dist_mse_total_ar.append(tot_dist_mse)
    if tot_dist_mse < min_mse_dist:
        output_dist_mse = pd.Series(dist_mse)
        min_mse_dist = tot_dist_mse
        best_dist = dist_name
        best_dist_data = distance

for j in range(0,5):
    lof_name = "LOF{}nn".format(j)
    lof = df[lof_name]
    auc_lof_org_ar.append(auc(lof,labels))
    lof_org_ar.append(lof)
    params = np.polyfit(loss,lof,deg=1)
    estimation = params[0]*loss+params[1]
    lof_mse = (np.abs(estimation - lof))
    lof_mse_ar.append(lof_mse)
    auc_lof = auc(lof_mse,labels)
    auc_lof_ar.append(auc_lof)
    tot_lof_mse = sum(lof_mse)
    lof_mse_total_ar.append(tot_lof_mse)
    if tot_lof_mse < min_mse_lof:
        output_lof_mse = pd.Series(lof_mse)
        min_mse_lof = tot_lof_mse
        best_lof = lof_name
        best_lof_data = lof

output = normalize(output_dist_mse)**2 + normalize(output_lof_mse)**2
output += best_dist_data + best_lof_data
best_auc = auc(output,labels)

print(min_mse_lof,min_mse_dist)

print("best AUC: {}".format(best_auc))
print("average dist AUC: {}".format(np.average(auc_dist_org_ar)))
print("average lof AUC: {}".format(np.average(auc_lof_org_ar)))
print("dest AUC mean: {}".format(auc(np.average(dist_org_ar,axis=0),labels)))
print("lof AUC mean: {}".format(auc(np.average(lof_org_ar,axis=0),labels)))
print(best_lof, best_dist)
print(auc_dist_org_ar)
print(auc_lof_org_ar)


