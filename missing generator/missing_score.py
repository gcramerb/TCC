import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import pandas as pd

df = pd.read_csv("C://Users/gcram/Desktop/UFMG/TCC/missing_score.csv", sep = ';',index_col =0)
ms = dict()
list_aux = []
for col in df.columns:
    for i in df[[col]].values:
        stri = i[0].replace(',','.')
        list_aux.append(np.float(stri))
    ms[col] = list_aux
    list_aux = []

df = pd.DataFrame(ms)
df.index = [5,10,20,25,35,50,70,90]

ax = df.plot(style=['ro-','ko-','go-','bo-'],title = 'Missing data impact')
ax.set_xlabel('Missing data rate (%)')
ax.set_ylabel('Accuracy (%)')
plt.show()