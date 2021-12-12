
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib

#%%

def PointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r+np.random.normal(-30,30),math.sin(2*math.pi/n*x)*r+np.random.normal(-30,30)) for x in range(1,n+1)]

df = pd.DataFrame(PointsInCircum(500,1000))
df = df.append(PointsInCircum(300,700))
print(df)

#%%
plt.figure(figsize=(5,5))
plt.scatter(df[0],df[1],s=15,color='grey')

plt.title('Dataset',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)

plt.show()
#%%

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=30,min_samples=6)

dbscan.fit(df[[0,1]])

df['DBSCAN_opt_labels']=dbscan.labels_

print(df['DBSCAN_opt_labels'].value_counts())

#%%

plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['DBSCAN_opt_labels'],s=15)
plt.title('DBSCAN Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()