from numpy import linspace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

terning = [1,2,3,4,5,6]

Mean_T_true = sum(terning)/len(terning)

Træk = []
Sandsynlighed = []

for i in range(100):
    Træk.append(np.random.choice(terning))

Mean_T = sum(Træk)/len(Træk)



linje = []

for i in np.linspace(start=1, stop=-1, num=1000):
    er = np.exp(i)
    linje.append(er)

plt.clf()

plt.plot(linje)
plt.grid()

plt.show()
