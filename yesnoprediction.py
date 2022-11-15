import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))

def cal_average(data):
    nodes = ["AF7" , "AF8" , "TP9" , "TP10"]
    o = list()
    for i in range(0,len(data)):
        t = list()

        for x in nodes:
            a = data["Delta_"+x][i]
            b = data["Theta_"+x][i]
            c = data["Gamma_"+x][i]
            d = data["Alpha_"+x][i]
            e = data["Beta_"+x][i]
            
            t.append((a+b+c+d+e)/5)
        o.append(t)
    return o

def dropColumns(data):
    
    drop_columns = ["Delta_Pz", "Delta_Oz", "Theta_P4", "Theta_PO4", "Gamma_P8", "Gamma_PO8", "Alpha_PO7", "Alpha_LMAST", "Beta_PO3", "Beta_IZ", "HeadBandOn", "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10", "Battery"]

    for i in drop_columns:
        data = data.drop(i , axis = 1)

    return data

def prediction(filename):
    nodes = ["AF7" , "AF8" , "TP9" , "TP10"]
    
    input_frame = pd.read_csv(filename)
    input_frame = input_frame.drop("Intention" ,axis = 1)
    input_frame = dropColumns(input_frame)
    average = cal_average(input_frame)
    frame2 = pd.DataFrame (average, columns = nodes)
    frame2 = frame2.dropna()
    fin_list = input_frame.values.tolist()

    pred_val = list()
    for i in fin_list:
        y_pred=model.predict([i])
        if y_pred[0] == 1:
            pred_val.append("Yes")
        if y_pred[0]== 0:
            pred_val.append("No")
        
    my_dict = {i:pred_val.count(i) for i in pred_val}

    return max(my_dict, key=my_dict.get)

