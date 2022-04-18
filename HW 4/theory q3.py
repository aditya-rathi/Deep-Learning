import numpy as np
Q = np.zeros((1500,1100))
i = 0
flag = 1
while flag or np.all((Q[i+1,:]-Q[i,:])<0.001):
    flag=0
    Q[i+1, :] = Q[i, :] + 1 * (1 + 0.5*(np.roll(Q[i, :],1)) - Q[i, :])
    i=i+1
    if i==1499:
        break
    
print(Q)
    