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
    number_of_samp_in_bulk = 3
    mse_zscore_min = 10000000
    best_mse = 0
    for num_samp_in_bulk in [number_of_samp_in_bulk]:
        count = 0
        mse_total = 0
        for itr in range(200):
            sample = xy.sample(num_samp_in_bulk)
            x_sample, y_sample = sample.iloc[:,0], sample.iloc[:,1]
            if sum(x_sample.values) != 0:
                A = np.vstack([x_sample, np.ones(len(x_sample))]).T
                m, c = np.linalg.lstsq(A, y_sample, rcond=None)[0]
                estimation = m * x + c
                mse = np.abs(estimation - y)
                mse_total += mse
                count += 1

        if isinstance(mse_total, int):
            continue

        mse_zscore = stats.zscore(mse_total)
        mse_zscore_sum = sum(abs(mse_zscore))
        if mse_zscore_sum < mse_zscore_min:
            mse_zscore_min = mse_zscore_sum
            best_mse = mse_total/count
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

def add_mse(df,loss,name,small_num,big_num):
    for i in range(small_num,big_num):
        if name == "Dist" or name == "LOF":
            feature_name = "{}{}nn".format(name,i)
        else:
            feature_name = "{}{}".format(name,i)
        feature = df[feature_name]
        mse = get_mse(feature, loss)
        df["{}_mse".format(feature_name)] = mse
        if isinstance(mse, int):
            df["{}_mse_norm".format(feature_name)] = 0
        else:
            df["{}_mse_norm".format(feature_name)] = normalize(mse)
    return df


def process_df(
    df,
    name,
    small_num,
    big_num,
    auc_list,
    feature_list,
    minimum_zscore_sum,
    best_auc,
    best_feature_name
):
    for i in range(small_num,big_num):
        if name == "Dist" or name == "LOF":
            feature_name = "{}{}nn".format(name,i)
        else:
            feature_name = "{}{}".format(name,i)
        feature_mse = df["{}_mse".format(feature_name)]
        norm_feature_mse = df["{}_mse_norm".format(feature_name)]
        mse_sum = sum(norm_feature_mse)
        mse_zscore_sum = sum(abs(stats.zscore(norm_feature_mse)))
        auc_list.append(auc(feature_mse, labels))
        feature_list.append(feature_name)
        print("{} AUC: {}".format(feature_name, auc(feature_mse, labels)))
        print("zscore {} mse sum: {}".format(feature_name, mse_zscore_sum))
        print("sum {} mse: {}".format(feature_name, mse_sum))
        if mse_sum == 0:
            continue
        if mse_zscore_sum < minimum_zscore_sum:
            best_auc = auc(feature_mse, labels)
            minimum_zscore_sum = mse_zscore_sum
            best_feature_mse = feature_mse
            best_feature_name = feature_name

    return minimum_zscore_sum, auc_list, feature_list, best_auc, best_feature_name

df_orig = pd.read_csv("/datasets/annthyroid-unsupervised-ad.csv", header=None)
df_results = pd.read_csv("/final_models/annthyroid_512bit_200000steps_summary.csv")
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


df_main = add_mse(df_main,loss,"Dist",0,49)
df_main = add_mse(df_main,loss,"LOF",0,5)
df_main = add_mse(df_main,loss,"f",0,features_num)
print("loss AUC: {}".format(auc(loss,labels)))
minimum_zscore_sum = 10000000
auc_list = []
feature_list = []
best_auc = ""
best_feature_name = ""
minimum_zscore_sum, auc_list, feature_list, best_auc, best_feature_name = process_df(df_main,"Dist",0,49,auc_list,feature_list,minimum_zscore_sum, best_auc, best_feature_name)
minimum_zscore_sum, auc_list, feature_list, best_auc, best_feature_name = process_df(df_main,"LOF",0,5,auc_list,feature_list,minimum_zscore_sum, best_auc, best_feature_name)
minimum_zscore_sum, auc_list, feature_list, best_auc, best_feature_name = process_df(df_main,"f",0,features_num,auc_list,feature_list,minimum_zscore_sum, best_auc, best_feature_name)
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

print("Best AUC: {}".format(best_auc))
#best_feature_mse.to_csv('/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/datasets/results1.csv')
print("Best feature: {}".format(best_feature_name))
print("Minimum zscore sum: {}".format(minimum_zscore_sum))
print(auc_list[np.argmax(auc_list)],feature_list[np.argmax(auc_list)])

