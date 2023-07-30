import os
import tensorflow as tf
import numpy as np

from PIL import Image
import cv2
import csv
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pickle
import pandas as pd


app = Flask(__name__)

model_pkl_file ="breastcancer_model_RF_2.pkl"

with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)
#model =load_model('breastcancer_model.pkl')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Breast cancer"
	elif classNo==1:
		return "Yes Breast Cancer"


def getResult(test):
    df = pd.read_csv("data.csv")
    #testfile = pd.read_csv("datahtml.csv",sep=',',header=None)
    testfile = pd.read_csv(test,sep='\t')
    df=df.dropna(axis=1)
    df=df.iloc[:,2:32]
    df=pd.concat([testfile, df]).reset_index(drop = True)
    X=df.iloc[:,0:30].values
    from sklearn.preprocessing import StandardScaler
    X=StandardScaler().fit_transform(X)
    result=model.predict([X[0]])  
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        radius_mean=request.form.get("radius_mean")
        texture_mean=request.form.get("texture_mean")
        perimeter_mean=request.form.get("perimeter_mean")
        area_mean=request.form.get("area_mean")
        smoothness_mean=request.form.get("smoothness_mean")
        compactness_mean=request.form.get("compactness_mean")
        concavity_mean=request.form.get("concavity_mean")
        concavepoints_mean=request.form.get("concave points_mean")
        symmetry_mean=request.form.get("symmetry_mean")
        fractal_dimension_mean=request.form.get("fractal_dimension_mean")
        radius_se=request.form.get("radius_se")
        texture_se=request.form.get("texture_se")
        perimeter_se=request.form.get("perimeter_se")
        area_se=request.form.get("area_se")
        smoothness_se=request.form.get("smoothness_se")
        compactness_se=request.form.get("compactness_se")
        concavity_se=request.form.get("concavity_se")
        concavepoints_se=request.form.get("concave points_se")
        symmetry_se=request.form.get("symmetry_se")
        fractal_dimension_se=request.form.get("fractal_dimension_se")
        radius_worst=request.form.get("radius_worst")
        texture_worst=request.form.get("texture_worst")
        perimeter_worst=request.form.get("perimeter_worst")
        area_worst=request.form.get("area_worst")
        smoothness_worst=request.form.get("smoothness_worst")
        compactness_worst=request.form.get("compactness_worst")
        concavity_worst=request.form.get("concavity_worst")
        concavepoints_worst=request.form.get("concave points_worst")
        symmetry_worst=request.form.get("symmetry_worst")
        fractal_dimension_worst=request.form.get("fractal_dimension_worst")
        data=[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concavepoints_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concavepoints_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concavepoints_worst,symmetry_worst,fractal_dimension_worst]
        with open('datahtml.csv', 'w', encoding='UTF8') as f:
             writer = csv.writer(f)
             writer.writerow(data)
             #writer.writerow(data)             
          
        
        #value=getResult("datahtml.csv")
        value=getResult(file_path)
        #value=getResult("btestm1.csv")
        print(value[0])        
        result=get_className(value[0])
        
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)