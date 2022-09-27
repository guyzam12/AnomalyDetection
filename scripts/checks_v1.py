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

def get_mse(loss,input):
    params = np.polyfit(loss, input, deg=1)
    estimation = params[0] * loss + params[1]
    mse = np.abs(estimation - input)
    return mse


dataset_name = "/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/"
print("Dataset name: {} ".format(dataset_name))
df = pd.read_csv("/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/satellite_512bit_150000steps_summary.csv")
loss = df['Loss']
labels = df['Labels']
min_mse_dist = 10000000
min_mse_lof = 10000000
output_mse = 0
output_lof = 0
auc_dist_org_ar, auc_dist_ar = [], []
dist_org_ar, dist_mse_ar, dist_mse_total_ar = [], [], []
auc_lof_org_ar, auc_lof_ar = [], []
lof_org_ar, lof_mse_ar, lof_mse_total_ar = [], [], []
number_of_samples_in_bulk = 200
temp = []
for i in range(0,49):
    dist_name = "Dist{}nn".format(i)
    distance = df[dist_name]
    auc_dist_org_ar.append(auc(distance,labels))
    dist_org_ar.append(distance)
    cur_df = df[['Index', 'Labels', 'Loss', dist_name]].copy()
    cur_df.sort_values(by=[dist_name], inplace=True)
    cur_df['mse'] = 0
    num_of_rows = distance.shape[0]
    bulk_num = int(num_of_rows / number_of_samples_in_bulk)
    cur_tot_dist_mse = 0
    max_bulk_mse = 0
    to_break = False
    tot_corr = 0
    for j in range(bulk_num+1):
        small_i, big_i = j*number_of_samples_in_bulk, (j+1)*number_of_samples_in_bulk
        if num_of_rows - big_i < number_of_samples_in_bulk/2:
            big_i = num_of_rows+1
            to_break = True
        cur_bulk = cur_df.iloc[small_i:big_i]
        cur_dist, cur_loss, cur_labels = cur_bulk[dist_name], cur_bulk['Loss'], cur_bulk['Labels']
        cur_dist_mse = get_mse(cur_loss,cur_dist)
        cur_index = cur_dist_mse.index.values
        cur_df.loc[cur_index,"mse"] = cur_dist_mse
        cur_tot_dist_mse += sum(cur_dist_mse)
        #corr = np.corrcoef(cur_dist_mse, cur_dist)[0, 1]
        #tot_corr += corr
        if sum(cur_dist_mse) > max_bulk_mse:
            max_bulk_mse = sum(cur_dist_mse)
        if to_break == True:
            break

    cur_tot_dist_mse -= max_bulk_mse
    #print(tot_corr,i,cur_tot_dist_mse)
    dist_mse = get_mse(loss, distance)
    tot_dist_mse = sum(dist_mse)
    auc_dist = auc(dist_mse,labels)
    corr = np.corrcoef(loss,distance)[0,1]
    #cur_tot_dist_mse /= np.abs(corr)
    if cur_tot_dist_mse < min_mse_dist:
        output_dist_mse = pd.Series(dist_mse)
        min_mse_dist = cur_tot_dist_mse
        best_dist = dist_name
        best_dist_data = distance
        best_dist_corr = np.abs(corr)

    temp.append(cur_tot_dist_mse)
    dist_mse_total_ar.append(tot_dist_mse)
    dist_mse_ar.append(dist_mse)
    auc_dist_ar.append(auc_dist)

for j in range(0,5):
    lof_name = "LOF{}nn".format(j)
    lof = df[lof_name]
    auc_lof_org_ar.append(auc(lof,labels))
    lof_org_ar.append(lof)
    cur_df = df[['Index', 'Labels', 'Loss', lof_name]].copy()
    cur_df.sort_values(by=[lof_name], inplace=True)
    cur_df['mse'] = 0
    num_of_rows = lof.shape[0]
    bulk = int(num_of_rows / 10)
    bulk_num = int(num_of_rows / number_of_samples_in_bulk)
    cur_tot_lof_mse = 0
    to_break = False
    max_bulk_mse = 0
    prev_max_bulk_lof = 0
    for j in range(bulk_num+1):
        small_i, big_i = j*number_of_samples_in_bulk, (j+1)*number_of_samples_in_bulk
        if num_of_rows - big_i < number_of_samples_in_bulk/2:
            big_i = num_of_rows+1
            to_break = True
        cur_bulk = cur_df.iloc[small_i:big_i]
        cur_lof, cur_loss, cur_labels = cur_bulk[lof_name], cur_bulk['Loss'], cur_bulk['Labels']
        cur_lof_mse = get_mse(cur_loss, cur_lof)
        if sum(cur_lof_mse) > max_bulk_mse:
            max_bulk_mse = sum(cur_lof_mse)
        cur_index = cur_lof_mse.index.values
        cur_df.loc[cur_index,"mse"] = cur_lof_mse
        cur_tot_lof_mse += sum(cur_lof_mse)
        if to_break == True:
            break

    cur_tot_lof_mse -= max_bulk_mse
    lof_mse = get_mse(loss, lof)
    lof_mse_ar.append(lof_mse)
    auc_lof = auc(lof_mse,labels)
    auc_lof_ar.append(auc_lof)
    tot_lof_mse = sum(lof_mse)
    lof_mse_total_ar.append(tot_lof_mse)
    corr = np.corrcoef(loss,lof)[0,1]
    cur_tot_lof_mse /= np.abs(corr)
    if cur_tot_lof_mse < min_mse_lof:
        output_lof_mse = pd.Series(lof_mse)
        #output_lof_mse = cur_df["mse"]
        min_mse_lof = cur_tot_lof_mse
        best_lof = lof_name
        best_lof_data = lof
        best_lof_corr = np.abs(np.corrcoef(loss, lof)[0, 1])

output = normalize(output_dist_mse)**2 + normalize(output_lof_mse)**2
output = normalize(output_lof_mse)**2
output = normalize(output_dist_mse)**2
#output = (output_dist_mse)# + best_dist_data# + normalize(output_lof_mse)**2
#output += (best_dist_data + best_lof_data)
#output += (best_lof_data)
if (min_mse_dist < min_mse_lof):
    print("choosed dist")
    print("min mse dist: {}, min mse lof: {}".format(min_mse_dist,min_mse_lof))
    #output = (normalize(output_dist_mse))
    output = normalize(output_dist_mse)**2
    output += best_dist_data
else:
    print("choosed lof")
    print("min mse dist: {}, min mse lof: {}".format(min_mse_dist,min_mse_lof))
    output = normalize(output_lof_mse)**2
    output += best_lof_data


output_lof = auc(normalize(output_lof_mse) ** 2 + best_lof_data,labels)
output_dist = auc(normalize(output_dist_mse) ** 2 + best_dist_data,labels)
best_auc = auc(output,labels)
print("dist AUC: {}, lof AUC: {}".format(output_dist,output_lof))
#print(best_dist_corr,best_lof_corr)
#print(min_mse_dist,min_mse_lof)
print("best AUC: {}".format(best_auc))
#print("average on AUC of knn: {}".format(np.average(auc_dist_org_ar)))
#print("average on AUC of lof: {}".format(np.average(auc_lof_org_ar)))
#print("mean on knn - AUC: {}".format(auc(np.average(dist_org_ar,axis=0),labels)))
#print("mean on lof - AUC: {}".format(auc(np.average(lof_org_ar,axis=0),labels)))
print("best dist: {}, best lof: {}".format(best_lof, best_dist))
print(auc_dist_org_ar)
print(auc_lof_org_ar)


