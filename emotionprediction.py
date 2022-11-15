import pickle
import pandas as pd

model = pickle.load(open('model.sav', 'rb'))

def cal_average(df):
    nodes = ["AF7" , "AF8" , "TP9" , "TP10"]
    op = list()
    for i in range(0,len(df)):
        t = list()

        for x in nodes:
            a = df["Delta_"+x][i]
            b = df["Theta_"+x][i]
            c = df["Gamma_"+x][i]
            d = df["Alpha_"+x][i]
            e = df["Beta_"+x][i]

            t.append((a+b+c+d+e)/5)
        op.append(t)
    return op

def dropColumns(df):
    
    drop_columns = ['TimeStamp' , 'RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10', 'AUX_RIGHT',
       'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyro_X',
       'Gyro_Y', 'Gyro_Z', 'HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8',
       'HSI_TP10', 'Battery']

    for i in drop_columns:
        df = df.drop(i , axis = 1)

    return df

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

    my_dict2 = {i:pred_val.count(i) for i in pred_val}

    return max(my_dict2, key=my_dict2.get)

