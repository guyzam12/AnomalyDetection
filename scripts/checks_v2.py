import pandas as pd
import numpy as np
import torch as th
from sklearn import metrics


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
    for num_samp_in_bulk in [50,60,100,200,500]:
        count = 0
        mse_total = 0
        for itr in range(100):
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
        mse_total_norm = normalize(mse_total)
        if sum(mse_total_norm.values) < mse_total_min:
            mse_total_min = sum(mse_total_norm.values)
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
    return df_main



df_orig = pd.read_csv("/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/datasets/letter-unsupervised-ad.csv", header=None)
df_results = pd.read_csv("/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/letter_512bit_500000steps_summary.csv")
df_main = create_df_main(df_orig, df_results)
df_orig_rows, df_orig_cols = df_orig.shape
features_num = df_orig_cols - 1
samples_num = df_orig_rows
loss = df_main['Loss']
labels = df_main['Labels']

for i in range(features_num):
    feature = df_main["f{}".format(i)]
    mse = get_mse(feature,loss,i)
    df_main["f{}_mse".format(i)] = mse
    if isinstance(mse, int):
        df_main["f{}_mse_norm".format(i)] = 0
    else:
        df_main["f{}_mse_norm".format(i)] = normalize(mse)


print("loss AUC: {}".format(auc(loss,labels)))
minimum_mse_sum = 10000000
total = 0
for i in range(features_num):
    feature_mse = df_main["f{}_mse_norm".format(i)]
    mse_sum = sum(feature_mse)
    if mse_sum!=0:
        total += feature_mse / mse_sum
    print("f{} AUC: {}".format(i, auc(feature_mse, labels)))
    print("sum f{} mse: {}".format(i,sum(feature_mse)))
    if mse_sum == 0:
        continue
    if mse_sum < minimum_mse_sum:
        best_auc = auc(feature_mse, labels)
        minimum_mse_sum = mse_sum

print("Best AUC: {}".format(best_auc))
print(auc(normalize(total)+loss,labels))
print("hi")


