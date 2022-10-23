import pandas as pd
import numpy as np
import torch as th
from sklearn import metrics
from scipy import stats

LOF_SMALL,LOF_BIG = 10,15
DIST_SMALL,DIST_BIG = 0,50
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
    for i in range(50):
        df_main["LOF{}nn".format(i)] = normalize(df_results["LOF{}nn".format(i)])
    for i in range(50):
        df_main["Dist{}nn".format(i)] = normalize(df_results["Dist{}nn".format(i)])
    for i in range(1):
        df_main["Isolation"] = normalize(df_results["Isolation"])
    return df_main

def add_mse(df,loss,name,small_num,big_num):
    for i in range(small_num,big_num):
        if name == "Dist" or name == "LOF":
            feature_name = "{}{}nn".format(name,i)
        else:
            if name == "Loss" or name == "Isolation":
                feature_name = "{}".format(name)
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
    methods_list,
    auc_list,
    feature_list,
    minimum_zscore_sum,
    best_auc,
    best_feature_name,
    zscore_sum_list
):
    methods_list.append("f")
    features_num = (df.filter(regex='f[0-9]+$', axis=1)).shape[1]
    for method in methods_list:
        if method == "Dist":
            small_num, big_num = DIST_SMALL, DIST_BIG
        elif method == "LOF":
            small_num, big_num = LOF_SMALL, LOF_BIG
        elif method == "Isolation":
            small_num, big_num = ISOLATION_SMALL, ISOLATION_BIG
        elif method == "Loss":
            small_num, big_num = LOSS_SMALL, LOSS_BIG
        elif method == "f":
            small_num, big_num = 0, features_num

        for i in range(small_num, big_num):
            if method == "Dist" or method == "LOF":
                feature_name = "{}{}nn".format(method,i)
            else:
                if method == "Loss" or method == "Isolation":
                    feature_name = "{}".format(method)
                else:
                    feature_name = "{}{}".format(method,i)
            feature_mse = df["{}_mse".format(feature_name)]
            norm_feature_mse = df["{}_mse_norm".format(feature_name)]
            mse_sum = sum(norm_feature_mse)
            mse_zscore_sum = sum(abs(stats.zscore(norm_feature_mse)))
            #mse_zscore_sum_t1 = sum(norm_feature_mse/np.std(norm_feature_mse))
            zscore_sum_list.append(mse_zscore_sum)
            auc_list.append(auc(feature_mse, labels))
            feature_list.append(feature_name)
            #print("{} AUC: {}".format(feature_name, auc(feature_mse, labels)))
            #print("zscore {} mse sum: {}".format(feature_name, mse_zscore_sum))
            #print("sum {} mse: {}".format(feature_name, mse_sum))
            if mse_sum == 0:
                continue
            if mse_zscore_sum < minimum_zscore_sum:
                best_auc = auc(feature_mse, labels)
                minimum_zscore_sum = mse_zscore_sum
                best_feature_mse = feature_mse
                best_feature_name = feature_name

    return minimum_zscore_sum, auc_list, feature_list, best_auc, best_feature_name, best_feature_mse, zscore_sum_list

def create_cur_df_main(cur_df_main,loss,method,features_num):
    cur_df_main = add_mse(cur_df_main, loss, "f", 0, features_num)
    if method == "Loss":
        cur_df_main = add_mse(cur_df_main, loss,"Dist", DIST_SMALL,DIST_BIG)
        cur_df_main = add_mse(cur_df_main, loss,"LOF",LOF_SMALL, LOF_BIG)
        cur_df_main = add_mse(cur_df_main,loss,"Isolation",ISOLATION_SMALL, ISOLATION_BIG)
    elif method == "Dist":
        cur_df_main = add_mse(cur_df_main, loss,"LOF", LOF_SMALL, LOF_BIG)
        cur_df_main = add_mse(cur_df_main,loss,"Isolation",ISOLATION_SMALL, ISOLATION_BIG)
        cur_df_main = add_mse(cur_df_main,loss,"Loss",LOSS_SMALL, LOSS_BIG)

    elif method == "LOF":
        cur_df_main = add_mse(cur_df_main, loss,"Dist", DIST_SMALL, DIST_BIG)
        cur_df_main = add_mse(cur_df_main,loss,"Isolation",ISOLATION_SMALL, ISOLATION_BIG)
        cur_df_main = add_mse(cur_df_main,loss,"Loss",LOSS_SMALL, LOSS_BIG)

    elif method == "Isolation":
        cur_df_main = add_mse(cur_df_main, loss,"Dist", DIST_SMALL, DIST_BIG)
        cur_df_main = add_mse(cur_df_main, loss,"LOF", LOF_SMALL, LOF_BIG)
        cur_df_main = add_mse(cur_df_main,loss,"Loss",LOSS_SMALL, LOSS_BIG)

    return cur_df_main

df_orig = pd.read_csv("/datasets/pen-global-unsupervised-ad.csv", header=None)
df_results = pd.read_csv("/final_models/pen_global_512bit_500000steps_summary.csv")
df_main = create_df_main(df_orig, df_results)
df_orig_rows, df_orig_cols = df_orig.shape
features_num = df_orig_cols - 1
samples_num = df_orig_rows
loss = df_main['Loss']
labels = df_main['Labels']

best_auc_final = 0
auc_list = []
zscore_sum_list = []
auc_per_method = []
mse_per_method = []
feature_list = []
best_possible_auc_per_method = []
zscore_sum_of_best_possible_auc_per_method = []
zscore_sum_auc_per_method = []
methods_list = ["Loss","Dist","LOF","Isolation"]
df_methods_auc = pd.DataFrame(columns=["Method","Best AUC","Best Possible AUC"])
small,big = 0,1
for method in methods_list:
    minimum_zscore_sum_final = 100000000
    best_feature_mse = 0
    auc_list = []
    feature_list = []
    zscore_sum_list = []
    if method == "Loss":
        small, big = LOSS_SMALL, LOSS_BIG
    elif method == "Dist":
        small, big = DIST_SMALL, DIST_BIG
    elif method == "LOF":
        small, big = LOF_SMALL, LOF_BIG
    elif method == "Isolation":
        small, big = ISOLATION_SMALL, ISOLATION_BIG

    for i in range(small, big):
        if method == "Dist":
            method_name = "Dist{}nn".format(i)
        elif method == "LOF":
            method_name = "LOF{}nn".format(i)
        elif method == "Isolation":
            method_name = "Isolation"
        elif method == "Loss":
            method_name = "Loss"
        loss = df_main[method_name]
        cur_df_main = df_main.copy()
        cur_df_main = create_cur_df_main(cur_df_main,loss,method,features_num)
        minimum_zscore_sum = 10000000
        best_auc = ""
        best_feature_name = ""
        new_methods_list = methods_list.copy()
        new_methods_list.remove(method)
        minimum_zscore_sum, auc_list, feature_list, best_auc, best_feature_name, cur_feature_mse, zscore_sum_list = process_df(cur_df_main,new_methods_list,auc_list,feature_list,minimum_zscore_sum, best_auc, best_feature_name, zscore_sum_list)
        if minimum_zscore_sum < minimum_zscore_sum_final:
            minimum_zscore_sum_final = minimum_zscore_sum
            best_auc_final = best_auc
            best_feature_mse = cur_feature_mse
            print("Cur Best AUC: {}".format(best_auc))
        print("Iteration: {}".format(i))

    auc_per_method.append(best_auc_final)
    mse_per_method.append(best_feature_mse)
    zscore_sum_auc_per_method.append(minimum_zscore_sum_final)
    best_possible_auc_per_method.append(auc_list[np.argmax(auc_list)])
    zscore_sum_of_best_possible_auc_per_method.append(zscore_sum_list[np.argmax(auc_list)])

df_methods_auc["Method"] = methods_list
df_methods_auc["Best AUC"] = auc_per_method
df_methods_auc["Zscore Sum Best AUC"] = zscore_sum_auc_per_method
df_methods_auc["Best Possible AUC"] = best_possible_auc_per_method
df_methods_auc["Zscore Sum of Best Possible AUC"] = zscore_sum_of_best_possible_auc_per_method

print("Best feature: {}".format(best_feature_name))
print("Minimum zscore sum: {}".format(minimum_zscore_sum))
print(auc_list[np.argmax(auc_list)],feature_list[np.argmax(auc_list)])
print("\nFinal Best AUC: {}\n".format(best_auc_final))
df_methods_auc.to_csv('/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/final_models/results_temp.csv')

