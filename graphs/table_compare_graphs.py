import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from table_evaluator import load_data, TableEvaluator
import pandas as pd
import re
## generate the data and plot it for an ideal normal curve

samples_file_name = "credit_full_50kstep_100samp.npz"
generated_samples_dir = "/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/generated_samples/"
graphs_dir = "/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/graphs/"
samples_file_path = generated_samples_dir+samples_file_name
output_graphs_dir = re.sub(".npz","",graphs_dir + samples_file_name)
npz = np.load(samples_file_path)['arr_0']
npz_new = np.reshape(npz,(npz.shape[0],npz.shape[-1]))
fake = pd.DataFrame(npz_new)
real = pd.read_csv("/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/datasets/creditcard_2.csv", usecols = [i for i in range(npz.shape[-1])], header=None)
real_1 = real.loc[205000:210000]
table_evaluator = TableEvaluator(real, fake)
table_evaluator.visual_evaluation(output_graphs_dir)