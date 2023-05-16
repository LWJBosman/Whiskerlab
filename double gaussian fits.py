# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:49:45 2022

@author: Randy
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
from sklearn import mixture
import matplotlib.pyplot
import matplotlib.mlab
import numpy as np
from pylab import *
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

y1=[]
df = pd.read_excel("C:\\LabProject\\LadderProject\\Data\\Touches_ercc1_mut.xlsx")

df = df.drop(df[df.sessionnr <= 5].index)
df["touch"] = df["touch end"] - df["touch begin"]
# df2["touch"] = df2["touch end"] - df2["touch begin"]

y1.extend(df["touch"])

y1 = np.array(y1)
y1 = y1[ (y1 >= 0) & (y1 <= 800) ]

data=y1
y,x,_=plt.hist(data, 100, alpha=.3, label='data')
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)

def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

expected = (30, 30, 40000, 80, 30, 10000)
params, cov = curve_fit(bimodal, x, y, expected)
sigma=np.sqrt(np.diag(cov))
x_fit = np.linspace(0, 500, 100)
#plot combined...
plt.plot(x_fit, bimodal(x_fit, *params), color='red', lw=3, label='model')
#...and individual Gauss curves
plt.plot(x_fit, gauss(x_fit, *params[:3]), color='red', lw=1, ls="--", label='distribution 1')
plt.plot(x_fit, gauss(x_fit, *params[3:]), color='red', lw=1, ls=":", label='distribution 2')
#and the original data points if no histogram has been created before
#plt.scatter(x, y, marker="X", color="black", label="original data")
plt.legend()
plt.xlim([30,40])
print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=bimodal.__code__.co_varnames[1:]))
plt.show() 

idx = np.argwhere(np.diff(np.sign(gauss(x_fit, *params[:3]) - gauss(x_fit, *params[3:])))).flatten()






