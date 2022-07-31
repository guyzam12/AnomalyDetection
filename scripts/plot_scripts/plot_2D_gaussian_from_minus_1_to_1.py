import torch as th
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

temp = np.load("/Users/guyzamberg/PycharmProjects/git/AnomalyDiffusion/generated_samples/gaussian_v2_generated_samples.npz")
x_arr,y_arr = [],[]
for i in temp['arr_0']:
    x,y = i[0][0],i[0][1]
    x_arr.append(x)
    y_arr.append(y)

x_data = np.arange(-1, 1, 0.001)
y_data = stats.norm.pdf(x_data, 0, 1)
plt.plot(x_arr, y_arr,'ro',markersize=3)
plt.plot(x_data,y_data,linewidth=2)
plt.show()

print(np.mean(temp['arr_0'],axis=0))
