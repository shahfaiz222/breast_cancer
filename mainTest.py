import cv2
import csv
import pandas as pd
from PIL import Image
import numpy as np
import pickle
model_pkl_file ="breastcancer_model_RF_2.pkl"

with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)
      

df = pd.read_csv("data.csv")
testfile = pd.read_csv("btestm1.csv",sep='\t')
df=df.dropna(axis=1)
df=df.iloc[:,2:32]
df=pd.concat([testfile, df]).reset_index(drop = True)
X=df.iloc[:,0:30].values
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X)
result=model.predict([X[0]])
print(result)
