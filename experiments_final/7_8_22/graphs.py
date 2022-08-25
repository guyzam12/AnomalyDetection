import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from table_evaluator import load_data, TableEvaluator
import pandas as pd
## generate the data and plot it for an ideal normal curve

npz = np.load('/generated_samples/temp.npz')['arr_0']
npz_new = np.reshape(npz,(1000,5))
temp = pd.DataFrame(npz_new)
real,fake = load_data('/datasets/gaussian_5samp.csv', '/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/datasets/gaussian_5samp.csv')
fake = temp
real.columns = [0,1,2,3,4]
table_evaluator = TableEvaluator(real, fake)
<<<<<<< HEAD
table_evaluator.visual_evaluation()
=======
table_evaluator.visual_evaluation()
#table_evaluator.evaluate(target_col=1)
print("ho")
>>>>>>> 52726769506eac2702c6f4fc047503fbe2a77baf
