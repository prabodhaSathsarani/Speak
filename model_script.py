# ## Importing necessary Libraries

#import libraries
import pandas as pd #It provides a variety of utilities, such as processing numerous file formats and converting an entire data table to a NumPy matrix array.
import numpy as np #NumPy requires far less memory to store data and gives a method for defining data types. 
from sklearn.preprocessing import LabelEncoder #This transformer should be used to encode target values
from sklearn.model_selection import train_test_split #train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets: for training data and for testing data
#from tensorflow.keras.utils import to_categorical #Converts a class vector (integers) to binary class matrix.
from sklearn.neural_network import MLPClassifier #For MLP classifier
from sklearn.metrics import classification_report #Build a text report showing the main classification metrics.
import seaborn as sns #making statistical graphics in Python
import matplotlib.pyplot as plt #provide various tools for data visualization in Python
#It provides interactivity with the backend in the frontends like the jupyter notebook.

# ## Calculating average for the 4 nodes

#This is a function for calculating Average for the 4 nodes.
#It takes data from TP9, AF7, AF8, and TP10. The muse monitor extracts many values, 20 of which are relevant. 
# Delta_TP9, Theta_TP9, Alpha_TP9, Beta_TP9,Gamma_TP9 These 5 values are given by 4 nodes, for a total of 20 characteristics. 
# We take the average and use 4 features.
def cal_average(df):
    nodes = ["AF7" , "AF8" , "TP9" , "TP10"]
    o = list()
    for i in range(0,len(df)):
        t = list()

        for x in nodes:
            a = df["Delta_"+x][i]
            b = df["Theta_"+x][i]
            c = df["Gamma_"+x][i]
            d = df["Alpha_"+x][i]
            e = df["Beta_"+x][i]

            t.append((a+b+c+d+e)/5)
        o.append(t)
    return o

# ## Dropping unecessary columns

#There are unnecessary columns. Those are 'TimeStamp' , 'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'AUX_RIGHT', 'Accelerometer_X', 
# 'Accelerometer_Y', 'Accelerometer_Z', 'Gyro_X','Gyro_Y', 'Gyro_Z', 'HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8',
# 'HSI_TP10', 'Battery' So I have dropped them usin dropColumns() function.
def dropColumns(df):
    
    drop_columns = ['TimeStamp' , 'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'AUX_RIGHT',
       'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyro_X',
       'Gyro_Y', 'Gyro_Z', 'HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8',
       'HSI_TP10', 'Battery']

    for i in drop_columns:
        df = df.drop(i , axis = 1)

    return df

import pickle
filename = 'model.sav'
model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

def prediction(filename):
    nodes = ["AF7" , "AF8" , "TP9" , "TP10"]

    input_frame = pd.read_csv(filename)
    input_frame = input_frame.drop("Elements" ,axis = 1)
    input_frame = dropColumns(input_frame)
    average = cal_average(input_frame)
    frame2 = pd.DataFrame (average, columns = nodes)
    frame2 = frame2.dropna()
    fin_list = frame2.values.tolist()

    pred_val = list()
    for i in fin_list:
        y_pred=model.predict([i])
        if y_pred[0] == 2:
            pred_val.append("Happy")
        if y_pred[0]== 1:
            pred_val.append("Sad")
        if y_pred[0] == 3:
            pred_val.append("Fear")
        else:
            pred_val.append("Fear")

    my_dict = {i:pred_val.count(i) for i in pred_val}

    return max(my_dict, key=my_dict.get)