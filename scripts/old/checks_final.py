import pandas as pd
import numpy as np
import torch as th
from sklearn import metrics
from scipy import stats

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

def get_mse(x, y):
    xy = pd.concat([x, y], axis=1)
    number_of_samp_in_bulk = 50
    mse_total_min = 10000000
    best_mse = 0
    for num_samp_in_bulk in [3]:
        count = 0
        mse_total = 0
        for itr in range(200):
            sample = xy.sample(num_samp_in_bulk)
            x_sample, y_sample = sample.iloc[:,0], sample.iloc[:,1]
            if sum(x_sample.values) != 0:
                params = np.polyfit(x_sample, y_sample, deg=1)
                estimation = params[0] * x + params[1]
                mse = np.abs(estimation - y)
                mse_total += mse
                count += 1

        if isinstance(mse_total, int):
            continue

        mse_total /= count
        mse_zscore = stats.zscore(mse_total)
        mse_zscore_sum = sum(abs(mse_zscore))
        if mse_zscore_sum < mse_total_min:
            mse_total_min = mse_zscore_sum
            best_mse = mse_total
            best_num_samp = num_samp_in_bulk

    if isinstance(best_mse, int):
        return 0
    else:
        return best_mse

def create_df_main(df_orig, df_results):
    df_orig_rows, df_orig_cols = df_orig.shape
    features_num = df_orig_cols - 1
    samples_num = df_orig_rows
    df_main = df_results[["Index", "Labels", "Loss"]].copy()
    for i in range(features_num):
        df_main["f{}".format(i)] = normalize(df_orig[i])
    for i in range(5):
        df_main["LOF{}nn".format(i)] = normalize(df_results["LOF{}nn".format(i)])
    for i in range(49):
        df_main["Dist{}nn".format(i)] = normalize(df_results["Dist{}nn".format(i)])
    #for i in range(1):
    #    df_main["Isolation"] = normalize(df_results["Isolation"])
    return df_main



df_orig = pd.read_csv("/datasets/aloi-unsupervised-ad.csv", header=None)
df_results = pd.read_csv("/final_models/aloi_512bit_500000steps_summary.csv")
df_main = create_df_main(df_orig, df_results)
df_orig_rows, df_orig_cols = df_orig.shape
features_num = df_orig_cols - 1
samples_num = df_orig_rows
loss = df_main['Loss']
labels = df_main['Labels']

#for i in range(1):
#    feature = df_main["Isolation"]
#    mse = get_mse(feature,loss)
#    df_main["Isolation_mse"] = mse
#    if isinstance(mse, int):
#        df_main["Isolation_mse_norm"] = 0
#    else:
#        df_main["Isolation_mse_norm"] = normalize(mse)


for i in range(49):
    feature = df_main["Dist{}nn".format(i)]
    mse = get_mse(feature,loss)
    df_main["Dist{}nn_mse".format(i)] = mse
    if isinstance(mse, int):
        df_main["Dist{}nn_mse_norm".format(i)] = 0
    else:
        df_main["Dist{}nn_mse_norm".format(i)] = normalize(mse)


for i in range(5):
    feature = df_main["LOF{}nn".format(i)]
    mse = get_mse(feature,loss)
    df_main["LOF{}nn_mse".format(i)] = mse
    if isinstance(mse, int):
        df_main["LOF{}nn_mse_norm".format(i)] = 0
    else:
        df_main["LOF{}nn_mse_norm".format(i)] = normalize(mse)


for i in range(features_num):
    feature = df_main["f{}".format(i)]
    mse = get_mse(feature,loss)
    df_main["f{}_mse".format(i)] = mse
    if isinstance(mse, int):
        df_main["f{}_mse_norm".format(i)] = 0
    else:
        df_main["f{}_mse_norm".format(i)] = normalize(mse)

print("loss AUC: {}".format(auc(loss,labels)))
minimum_mse_sum = 10000000
total = 0
auc_list = []
feature_list = []

#for i in range(1):
#    feature_name = "Isolation"
#    feature_mse = df_main["Isolation_mse"]
#    norm_feature_mse = df_main["Isolation_mse_norm"]
#    mse_sum = sum(norm_feature_mse)#/sum(norm_feature_mse)
#    mse_sum_zscore = sum(abs(stats.zscore(norm_feature_mse, ddof=1)))
#    mse_sum = mse_sum_zscore
#    if mse_sum != 0:
#        total += feature_mse
#    print("Isolation AUC: {}".format(auc(feature_mse, labels)))
#    print("Isolation sum mse: {}".format(sum(norm_feature_mse)))
#    print("zscore Isolation mse sum: {}".format(mse_sum_zscore))
#    if mse_sum == 0:
#        continue
#    if mse_sum < minimum_mse_sum:
#        best_auc = auc(feature_mse, labels)
#        minimum_mse_sum = mse_sum
#        best_feature_mse = feature_mse
#        best_feature_name = feature_name

for i in range(0,49):
    feature_name = "Dist{}nn".format(i)
    feature_mse = df_main["Dist{}nn_mse".format(i)]
    norm_feature_mse = df_main["Dist{}nn_mse_norm".format(i)]
    mse_sum = sum(norm_feature_mse)
    mse_sum_zscore = sum(abs(stats.zscore(norm_feature_mse)))
    feature_mse_sum = sum(feature_mse)
    auc_list.append(auc(feature_mse, labels))
    feature_list.append(feature_name)
    print("Dist{}nn AUC: {}".format(i, auc(feature_mse, labels)))
    print("zscore Dist{}nn mse sum: {}".format(i, mse_sum_zscore))
    print("sum Dist{}nn mse: {}".format(i,mse_sum))
    if mse_sum == 0:
        continue
    if mse_sum_zscore < minimum_mse_sum:
        best_auc = auc(feature_mse, labels)
        minimum_mse_sum = mse_sum_zscore
        best_feature_mse = feature_mse
        best_feature_name = feature_name


for i in range(5):
    feature_name = "LOF{}nn".format(i)
    feature_mse = df_main["LOF{}nn_mse".format(i)]
    norm_feature_mse = df_main["LOF{}nn_mse_norm".format(i)]
    mse_sum = sum(norm_feature_mse)
    mse_sum_zscore = sum(abs(stats.zscore(norm_feature_mse)))
    feature_mse_sum = sum(feature_mse)
    auc_list.append(auc(feature_mse, labels))
    feature_list.append(feature_name)
    print("LOF{}nn AUC: {}".format(i, auc(feature_mse, labels)))
    print("zscore LOF{}nn mse sum: {}".format(i, mse_sum_zscore))
    print("sum LOF{}nn mse: {}".format(i,mse_sum))
    if mse_sum == 0:
        continue
    if mse_sum_zscore < minimum_mse_sum:
        best_auc = auc(feature_mse, labels)
        minimum_mse_sum = mse_sum_zscore
        best_feature_mse = feature_mse
        best_feature_name = feature_name

for i in range(features_num):
    feature_name = "f{}".format(i)
    feature_mse = df_main["f{}_mse".format(i)]
    norm_feature_mse = df_main["f{}_mse_norm".format(i)]
    mse_sum = sum(norm_feature_mse)
    mse_sum_zscore = sum(abs(stats.zscore(norm_feature_mse)))
    feature_mse_sum = sum(feature_mse)
    auc_list.append(auc(feature_mse, labels))
    feature_list.append(feature_name)
    print("f{} AUC: {}".format(i, auc(feature_mse, labels)))
    print("zscore f{} mse sum: {}".format(i, mse_sum_zscore))
    print("sum f{} mse: {}".format(i,mse_sum))
    if mse_sum == 0:
        continue
    if mse_sum_zscore < minimum_mse_sum:
        best_auc = auc(feature_mse, labels)
        minimum_mse_sum = mse_sum_zscore
        best_feature_mse = feature_mse
        best_feature_name = feature_name


print("Best AUC: {}".format(best_auc))
best_feature_mse.to_csv('/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/datasets/results1.csv')
print("Best feature: {}".format(best_feature_name))
print("Minimum mse sum: {}".format(minimum_mse_sum))
print(auc_list[np.argmax(auc_list)],feature_list[np.argmax(auc_list)])

