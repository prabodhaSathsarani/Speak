#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 
plt.style.use('seaborn-bright')

from pandas import DataFrame

import sklearn
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics

#from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from scipy import stats
from scipy.stats import norm, skew
import statsmodels.formula.api as sm

import warnings
import re
warnings.filterwarnings('ignore')

import pickle

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importdata
data = pd.read_csv('BinaryDecisionDataset.csv')

print("Dataset : \n", data.head())
print("")
print ("Dataset Shape : ", data.shape)
print("")
data.info()
print("")

#check for null values
print("Check for null values : \n", data.isnull().sum())
print("")

#check for unique values of the attributes of class
print("unique values of Intention :", data.Intention.unique())
print("")

#drop extra columns from dataset
data = data.drop(["Delta_Pz", "Delta_Oz", "Theta_P4", "Theta_PO4", "Gamma_P8", "Gamma_PO8", "Alpha_PO7", "Alpha_LMAST", "Beta_PO3", "Beta_IZ", "HeadBandOn", "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10", "Battery"], axis=1)
    
#pairwise correlation
print("Pairwise correlation : \n", data.corr())
print("")


# In[44]:


dplot = pd.DataFrame(data, columns = list('XY'))
dplot

figure = plt.figure()
sns.set_style('darkgrid')
sns.scatterplot(data = data, x = 'Alpha_AF7', y = 'Theta_AF8')
plt.show()


# In[6]:


#sns.set_style('darkgrid')

#sns.pairplot(data);


# In[3]:


data.columns


# In[4]:


#split dataset in to train data and test data

# Separating the target variable
X,Y = data[['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10', 'Theta_TP9',
       'Theta_AF7', 'Theta_AF8', 'Theta_TP10', 'Gamma_TP9', 'Gamma_AF7',
       'Gamma_AF8', 'Gamma_TP10', 'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8',
       'Alpha_TP10', 'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']], data[['Intention']]
    
# Splitting the dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.25, random_state=42)


# In[5]:


X_test


# In[6]:


Y_test


# # Logistic Regression Classifier

# In[7]:


# Create an instance of the model.
logreg = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# Training the model.
logreg.fit(X_train.values, Y_train)


# In[8]:


#model reuslts prediction
predicted = logreg.predict(X_test)


# In[9]:


predicted


# In[10]:


# Accuray report AUC
accuracy = metrics.accuracy_score(Y_test, predicted)
#auc = metrics.roc_auc_score(Y_test, predicted_prob)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
#print("Auc:", round(auc,2))


# In[11]:


#confusion matrix
cm = confusion_matrix(Y_test, predicted)
print("confusion matrix : \n", cm)

#plot confusion matrix
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap = plt.cm.Wistia)
classNames = ['Negative', 'Positive']
plt.title('Confusion Matrix - Test Data')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation = 45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN','TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+ " = "+str(cm[i][j]))
plt.show()

#prediction probability
predicted_prob = logreg.predict_proba(X_test)[:,1]

# Accuray report AUC
accuracy = metrics.accuracy_score(Y_test, predicted)
auc = metrics.roc_auc_score(Y_test, predicted_prob)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc,2))

# Precision e Recall
recall = metrics.recall_score(Y_test, predicted)
precision = metrics.precision_score(Y_test, predicted)
F1_score = metrics.f1_score(Y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("F1 score :", round(F1_score,2))
print("Detail:")
print(metrics.classification_report(Y_test, predicted, target_names=[str(i) for i in np.unique(Y_test)]))

#generate the pickel file

pickle.dump(logreg, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

# # Calculate average for node frame 20 features

# In[12]:


#def cal_average(data):
#    nodes = ["AF7" , "AF8" , "TP9" , "TP10"]
 #   o = list()
  #  for i in range(0,len(data)):
   #     t = list()

#        for x in nodes:
 #           a = data["Delta_"+x][i]
  #          b = data["Theta_"+x][i]
   #         c = data["Gamma_"+x][i]
    #        d = data["Alpha_"+x][i]
     #       e = data["Beta_"+x][i]
            
      #      t.append((a+b+c+d+e)/5)
       # o.append(t)
 #   return o


# # Drop extra columns in fro prediction

# In[13]:


#def dropColumns(data):
    
 #   drop_columns = ["Delta_Pz", "Delta_Oz", "Theta_P4", "Theta_PO4", "Gamma_P8", "Gamma_PO8", "Alpha_PO7", "Alpha_LMAST", "Beta_PO3", "Beta_IZ", "HeadBandOn", "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10", "Battery"]

  #  for i in drop_columns:
   #     data = data.drop(i , axis = 1)

    #return data


# # Prediction

# In[14]:


#def prediction(filename):
 #   nodes = ["AF7" , "AF8" , "TP9" , "TP10"]
    
  #  input_frame = pd.read_csv(filename)
  #  input_frame = input_frame.drop("Intention" ,axis = 1)
  #  input_frame = dropColumns(input_frame)
  #  average = cal_average(input_frame)
  #  frame2 = pd.DataFrame (average, columns = nodes)
  #  frame2 = frame2.dropna()
  #  fin_list = input_frame.values.tolist()

  #  pred_val = list()
  #  for i in fin_list:
  #      y_pred=logreg.predict([i])
  #      if y_pred[0] == 1:
  #          pred_val.append("Yes")
  #      if y_pred[0]== 0:
  #          pred_val.append("No")
        
  #  my_dict = {i:pred_val.count(i) for i in pred_val}

  #  return max(my_dict, key=my_dict.get)


# In[25]:


#filename = "BinaryDecisionDatasetTesting1.csv"
#decision = prediction(filename)


# In[24]:


#decision = "No"


# In[26]:


#print("The identified decision is : " + decision)


# In[27]:


#print(decision)


# In[47]:


#def sound(decision):
 #   if decision == "yes":
  #      s1 = decision + " it is"
   #     print(s1)
    #    return s1
   # if decision == "no":
    #    s2 = decision + " it's not"
     #   print(s2)
      #  return s2


# # Generate Yes/No Speech Sound

# In[18]:


#import gtts
#from playsound import playsound

#generate speech sound
#sound = gtts.gTTS(decision)

#save speech sound in mp3 file
#sound.save("decision.mp3")

#if decision == "yes":
 #   sound1 = gtts.gTTS(decision + " it is")
    
    #save speech sound in mp3 file
  #  sound1.save("decision.mp3")
#if decision == "no":
 #   sound2 = gtts.gTTS(decision + " it's not")
    
    #save speech sound in mp3 file
  #  sound2.save("decision.mp3")


# In[20]:


#execute mp3 file to output the speech sound

#playsound("decision.mp3")

#sound = sound(decision)
#if decision == "yes":
  #  playsound("s1.mp3")
#if decision == "no":
 #   playsound("s2.mp3")


# In[ ]:




