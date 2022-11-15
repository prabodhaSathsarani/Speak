# %% [markdown]
# ## Importing necessary Libraries

# %%
#import libraries
import pandas as pd #It provides a variety of utilities, such as processing numerous file formats and converting an entire data table to a NumPy matrix array.
import numpy as np #NumPy requires far less memory to store data and gives a method for defining data types. 
from sklearn.preprocessing import LabelEncoder #This transformer should be used to encode target values
from sklearn.model_selection import train_test_split #train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets: for training data and for testing data
from tensorflow.keras.utils import to_categorical #Converts a class vector (integers) to binary class matrix.
from sklearn.neural_network import MLPClassifier #For MLP classifier
from sklearn.metrics import classification_report #Build a text report showing the main classification metrics.
import seaborn as sns #making statistical graphics in Python
import matplotlib.pyplot as plt #provide various tools for data visualization in Python
# %matplotlib inline 
#It provides interactivity with the backend in the frontends like the jupyter notebook.

# %% [markdown]
# ## Calculating average for the 4 nodes

# %%
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

# %% [markdown]
# ## Dropping unecessary columns

# %%
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


#start to predict the emotions using previous created model
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
        y_pred=classifier.predict([i])
        if y_pred[0] == 2:
            pred_val.append("Happy")
        if y_pred[0]== 1:
            pred_val.append("High Happy")
        if y_pred[0] == 3:
            pred_val.append("Sad")
        else:
            pred_val.append("Fear")

    my_dict = {i:pred_val.count(i) for i in pred_val}

    return max(my_dict, key=my_dict.get)
# %% [markdown]
# ## Reading Data

# %%
#Read the data
def pre_process(df):
    df = pd.read_csv(df)

    # %%
    df.head() #Return the 5 rows.

    # %%
    #describe the all data
    df.describe()

    # %%
    df.isnull().sum() #returns the number of null values in the data set.

    # %%
    df.columns #The column labels of the DataFrame.

    # %% [markdown]
    # ## Checking if the data labels are balanced

    # %%
    #Data visualization

    #Target column data
    plt.figure(figsize=(15,5))
    sns.countplot(x ='Label', data = df)
    plt.xlabel('Label', size = 15)
    plt.ylabel('Identified number of signals', size = 15)
    plt.title('Emotion classification based on EEG', color = 'green', size = 25)
    plt.show()

    # %%
    df = dropColumns(df)

    # %%
    df.head() ##Return the first 5 rows.

    # %%
    nodes = ["AF7" , "AF8" , "TP9" , "TP10"] #colums of data frame

    # %%
    df.columns 

    # %%
    o = cal_average(df)

    # %%
    #after getting avg values, we create a new dataframe
    df1 = pd.DataFrame (o, columns = nodes)
    df1["Label"] = df["Label"]
    df1.tail()

    # %% [markdown]
    # ## Splitting into x and y variables

    # %%
    x = df1.iloc[:,:-1]
    z=x.to_numpy()
    y = df1.iloc[:,-1]

    # %% [markdown]
    # ## Encoding labels

    # %%
    x = pd.DataFrame(z , columns = nodes)
    labelencoder = LabelEncoder()
    y[:] = labelencoder.fit_transform(y[:])
    # y = to_categorical(y)
    #variables should be differenciated
    #happy = 1, high happy =2 ....


    # %%
    #Get categorical data from label column, and converts into numerical values
    test = []
    for i in y:
        r = [i]
        test.append(r)

    y = test

    # %%
    y = pd.DataFrame(y , columns=["Labels"])# y values --->data frame

    # %%
    y['Labels'].unique()

    # %% [markdown]
    # ## Oversampling since data labels are imbalance

    # %%
    #import libraries for oversampling since data labels ara imbalanced
    from imblearn.over_sampling import RandomOverSampler 

    under = RandomOverSampler()
    x, y = under.fit_resample(x, y)
    # y = to_categorical(y)



    # %% [markdown]
    # ## Splitting into training and testing set

    # %%
    X_train , X_test , Y_train , Y_test = train_test_split(x,y,test_size=0.20)
    #Here, the dataset is divided 80/20

    # %% [markdown]
    # ## KNN calssifier

    # %%
    from sklearn.neighbors import KNeighborsClassifier #Classifier implementing the k-nearest neighbors vote.

    classifier = KNeighborsClassifier(n_neighbors=5 , metric = "minkowski" , p=2)
    classifier.fit(X_train , Y_train)

    #prediction the test set results
    y_pred = classifier.predict(X_test)

    #marking the confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test, y_pred)

    # %% [markdown]
    # ## Confusion Matrix

    # %%
    #call confusion matrix
    cm

    # %%
    #call the classification report for precision, recall, f1-score and support width
    print(classification_report(Y_test,y_pred))

    # %%
    from sklearn.metrics import accuracy_score #Accuracy classification score.

    #Find the accuracy from KNN
    y_pred=classifier.predict(X_test)
    accuracy=accuracy_score(y_true=Y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy*100))

    # %% [markdown]
    # ## MLP Classifier

    # %%
    model=MLPClassifier(alpha=0.05, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(500,), learning_rate='constant', max_iter=700 , solver="adam" , activation= 'tanh')

    # %%
    #fit the model
    model.fit(X_train,Y_train)

    # %%
    #find the accuracy of emotional analysis
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_true=Y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy*100))

    # %%
    y_pred

    # %%
    filename = "sad_test.csv"
    emotion = prediction(filename)

    # %%
    #identify the emotional feeling
    print("The identified emotion is : fear " )

    # %%
    #print the emotional feeling
    print("Now I am in  " + emotion + " time")

    # %% [markdown]
    # ## Generate Emotional Sound

    # %%
    #import libraries for generate the sound for emotional voice
    import gtts
    from playsound import playsound

    # %%
    # make request to google to get synthesis
    tts = gtts.gTTS("Now I am in  " + emotion + " time")

    # %%
    #save the emotional sound file
    tts.save("sad23.mp3")

    # %%
    #play the sound of emotions
    playsound("sad23.mp3")


