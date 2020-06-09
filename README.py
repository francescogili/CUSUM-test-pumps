# CUSUM-test-pumps
import numpy as np
import pandas as pd
import os


df = pd.read_csv('output.txt')

df







%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np

plt.figure(figsize=(12, 4))
a = sns.violinplot(df.powerclean)

plt.plot(df.powerclean)
plt.show()








import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import *
import math

def find_min_array(index, array):
    i=0
    minimum = array[0]
    while i <= index:
        if minimum <= array[i]:
            i += 1
        else:
            minimum = array[i]
            i += 1
    return minimum


s = df.powerclean


sigma = 3

    #print(x)
sum_up = np.zeros(99706)

theta_z = input("Enter the mean before the changepoint: ") 
theta_o = input("Enter the mean after the changepoint: ") 
theta_1 = float(theta_o)
theta_0 = float(theta_z)

p_0 = 1
p_1 = 1


change_point_detected = 0

i=1

l = [x for x in s if ~np.isnan(x)]

def split_list(a_list):
    half = len(a_list)//1000
    u = len(a_list) - 12000
    v = len(a_list) -10000
    return a_list[u:v]

k=split_list(l)
len_k = len(k)



while i<len_k:
    x[i] = k[i]

    p_0 = np.exp((-1/(2*((sigma)**2)))*((x[i] - theta_0)**2))
    p_1 = np.exp((-1/(2*((sigma)**2)))*((x[i] - theta_1)**2))
    
    sum_up[i] = sum_up[i-1] + np.log(p_1 / p_0)
    
    g_n = sum_up[i] - find_min_array(i,sum_up)
    h = 2
    if g_n < h:
        i += 1
    else:
        change_point_detected = i
        i = len_k


print("This is the changepoint detected: " + str(change_point_detected))

p =0 
sum_up_plotting = np.zeros(200)

while p < 200:
    sum_up_plotting[p] = sum_up[p]
    p += 1
    

plt.plot(sum_up_plotting)
plt.xlabel(r'$Time$')
plt.ylabel(r'$\sum_i^n \log{f_{\theta_1} \left( X_i \right)} / {f_{\theta_0} \left( X_i \right)}$')
plt.title('Plot of the behaviour of S_n')
plt.show()







from scipy import stats



s = df.powerclean
l = [x for x in s if ~np.isnan(x)]

def split_list(a_list):
    half = len(a_list)//1000
    u = len(a_list) - 12000
    v = len(a_list) -10000
    return a_list[u:v]

k=split_list(l)
print(len(k))
print(k)

tester = np.zeros(120)
m=0
while m < 120:
    tester[m] = k[250+ m]
    m += 1
    
k2, p = stats.normaltest(tester)
alpha = 5e-3
print("p = {:g}".format(p))

if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import *
import math

def find_min_array(index, array):
    i=0
    minimum = array[0]
    while i <= index:
        if minimum <= array[i]:
            i += 1
        else:
            minimum = array[i]
            i += 1
    return minimum


s = df.powerclean

l = [x for x in s if ~np.isnan(x)]

def split_list(a_list):
    half = len(a_list)//1000
    u = len(a_list) - 12000
    v = len(a_list) -10000
    return a_list[u:v]

k=split_list(l)
len_k = len(k)
print(len_k)
vettore_differenze = np.zeros(len_k)
w=0

while w < len_k-1:
    vettore_differenze[w] = k[w+1]-k[w]
    
    w +=1


sigma = 3

    #print(x)
sum_up = np.zeros(99706)

theta_z = input("Enter the mean before the changepoint: ") 
theta_o = input("Enter the mean after the changepoint: ") 
theta_1 = float(theta_o)
theta_0 = float(theta_z)

p_0 = 1
p_1 = 1


change_point_detected = 0

i=1


x = np.zeros(len_k + 1)


while i<len_k-1:
    x[i] = vettore_differenze[i]

    p_0 = np.exp((-1/(2*((sigma)**2)))*((x[i] - theta_0)**2))
    p_1 = np.exp((-1/(2*((sigma)**2)))*((x[i] - theta_1)**2))
    
    sum_up[i] = sum_up[i-1] + np.log(p_1 / p_0)
    
    g_n = sum_up[i] - find_min_array(i,sum_up)
    h = 2
    if g_n < h:
        i += 1
    else:
        change_point_detected = i
        i = len_k


print("This is the changepoint detected: " + str(change_point_detected))

p =0 
sum_up_plotting = np.zeros(200)

while p < 200:
    sum_up_plotting[p] = sum_up[p]
    p += 1
    

plt.plot(sum_up_plotting)
plt.xlabel(r'$Time$')
plt.ylabel(r'$\sum_i^n \log{f_{\theta_1} \left( X_i \right)} / {f_{\theta_0} \left( X_i \right)}$')
plt.title('Plot of the behaviour of S_n')
plt.show()










import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import *
import math

s = df.powerclean

l = [x for x in s if ~np.isnan(x)]

def split_list(a_list):
    half = len(a_list)//1000
    u = len(a_list) - 12000
    v = len(a_list) -10000
    return a_list[u:v]

k=split_list(l)
len_k = len(k)
print(len_k)
vettore_differenze = np.zeros(len_k)
w=0

while w < len_k-1:
    vettore_differenze[w] = k[w+1]-k[w]
    #print(vettore_differenze[w])
    
    w +=1

plt.plot(vettore_differenze)
plt.show()

plt.figure(figsize=(12, 4))
a = sns.violinplot(vettore_differenze)
