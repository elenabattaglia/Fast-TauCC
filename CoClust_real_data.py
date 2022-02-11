import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from algorithms.coclust_incremental import CoClust
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari



dataset = 'cstr'           # classic3, cstr, tr11, tr41, hitech, k1b, reviews, sports
init = 'extract_centroids' # this is the only initialization considered in the paper


dt = pd.read_csv(f'./datasets/matrices/{dataset}.txt')
t = pd.read_csv(f'./datasets/matrices/{dataset}_target.txt', header = None)
target = np.array(t).T[0]

n = len(dt.doc.unique())
m = len(dt.word.unique())
T = np.zeros((n,m), dtype = int)


for g in dt.iterrows():
    T[g[1].doc,g[1].word] = g[1].cluster


model = CoClust(initialization = init, verbose = True)
model.fit(T)

print(f"nmi: {nmi(target, model.row_labels_)}")
print(f"ari: {ari(target, model.row_labels_)}")

#### uncomment the lines below to plot tau functions
##
##fig, ax = plt.subplots()
##ax.plot(model.tau_x)
##ax.plot(model.tau_y)
##plt.plot([(model.tau_x[i] + model.tau_y[i])/2 for i in range(len(model.tau_x))])
##ax.legend(['tau x','tau y','avg tau'])
##ax.set_xlabel('iterations')
##ax.set_ylabel('tau')
##plt.show()


