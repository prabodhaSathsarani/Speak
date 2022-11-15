#from crypt import methods
from fileinput import filename
import os
from flask import Flask, json, request, jsonify, render_template
import urllib.request
from werkzeug.utils import secure_filename
import json

import pandas as pd
import numpy as np
import pickle

import time
import gtts
from playsound import playsound

#from yesnoprediction import *
#from model_script import *

import yesnoprediction as h_model
import emotionprediction as p_model

app = Flask(__name__)

#model = pickle.load(open('model.pkl', 'rb'))

app.secret_key = "1234"

UPLOAD_FOLDER = 'static/uploads'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def main():
    #return 'Homepage'
    return render_template('home.html')

@app.route('/heshiki', methods=['GET', 'POST'])
def Heshiki():
    #return 'Homepage'
    return render_template('index.html')

@app.route('/praboda', methods=['GET', 'POST'])
def Praboda():
    #return 'Homepage'
    return render_template('index2.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload_file_heshiki():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #new_filename = f'{filename.split(".")[0]}_{str(datetime.now)}.csv'
            #file.save(os.path.join('input', filename)) #can save with new filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True

            decision = h_model.prediction('./static/uploads/'+filename)
            print(decision)

            #generate speech sound
            #sound = gtts.gTTS(decision)
            #save speech sound in mp3 file
            #sound.save("decision.mp3")
            #os.remove('decision.mp3')
            #sound.save("decision.mp3")
            #time.sleep(1)
            #playsound("decision.mp3")

            return jsonify({'result':decision})

            #def textOut(res):
            #    return (predout(output))

        else:
            errors[file.filename] = 'File type is not allowed'

        

    #check if the post request has the file part
    #if 'files[]' not in request.files:
    #    resp = jsonify({'message' : 'No file part in the request'})
    #    resp.status_code = 400
    #    return resp

    #files = request.files.getlist('files[]')

    errors = {}
    success = False

@app.route('/predict', methods=['POST'])
def upload_file_praboda():
   if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error':'please upload .csv file'})

        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # 
            emotion = p_model.prediction('./static/uploads/'+filename)
            print(emotion)

            #generate speech sound
            #sound = gtts.gTTS(emotion)
            #save speech sound in mp3 file
            #sound.save("emotion.mp3")
            #os.remove('emotion.mp3')
            #sound.save("emotion.mp3")
            #time.sleep(1)
            #playsound("emotion.mp3")

            return jsonify({'result':emotion})

        return jsonify({'error':'please upload .csv file'})

    #for file in files:
    #    if file and allowed_file(file.filename):
    #        filename = secure_filename(file.filename)
    #        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #        success = True
            #print("upload success")

    #    else:
    #        errors[file.filename] = 'File type is not allowed'

        #read csv to predict
        #data = file.read()
        #dataset = [np.array(data)]
        #dataset2D = np.expand_dims(dataset, 0)
        #dataset2D = dataset.array.reshape(-1, 1)
        #prediction = model.predict(dataset2D)
        #output = prediction[0]
        #print(output)

        #if output == 0:
            #resp = jsonify(({'message' : 'NO'}).format(output))
            #return resp
        #else:
            #resp = jsonify({'message' : 'YES'})
            #return resp

        #mewa hrida dn naaa predict
        #data = file.read()
        #dataset = [float(x) for x in data]
        #dataarray = [np.array(dataset)]
        #prediction = model.predict(dataarray)
        #output = prediction[0]
        #print(output)

        #if output == 0:
            #resp = jsonify(({'message' : 'NO'}).format(output))
        #    print("NO")
        #    return ('NO'.format(output))
        #else:
            #resp = jsonify({'message' : 'YES'})
        #    print("YES")
        #    return 'YES'

#-------------------------------------------------------------

#    if success and errors:
#        data = file.read()
#        filepath =  f'static/uploads/{file.filename}'
#        print(filepath)
#        data = pd.read_csv(filepath)
#        print("CSV", data.columns)
#        df = pd.DataFrame(data)
#       df = df.drop(["Delta_Pz", "Delta_Oz", "Theta_P4", "Theta_PO4", "Gamma_P8", "Gamma_PO8", "Alpha_PO7", "Alpha_LMAST", "Beta_PO3", "Beta_IZ", "HeadBandOn", "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10", "Battery", "Intention"], axis=1)
#        print(df.columns)
#        array = df.to_numpy()
#        prediction = model.predict(array)
#        output = prediction[0]
#        print("output",output)

#        def predout(output):
#            if output == 0:
#                #resp = jsonify(({'message' : 'NO'}).format(output))
#                print("NO")
#                return ("NO".format(output))
#                #res = {"prediction_text": "NO".format(output)}
#                #return res['prediction_text']
#            else:
#                #resp = jsonify({'message' : 'YES'})
#                print("YES")
#                return ("YES")
#                #res = {"prediction_text": "YES"}
#                #return res['prediction_text']

#        errors['message'] = 'File successfully uploadded'
#        resp = jsonify(errors)
#        resp.status_code = 500
#        print("uploaded")
#        return resp

#    if success:
#        data = file.read()
#        filepath =  f'static/uploads/{file.filename}'
#        print(filepath)
#        data = pd.read_csv(filepath)
#        print("CSV", data.columns)
#        df = pd.DataFrame(data)
#        df = df.drop(["Delta_Pz", "Delta_Oz", "Theta_P4", "Theta_PO4", "Gamma_P8", "Gamma_PO8", "Alpha_PO7", "Alpha_LMAST", "Beta_PO3", "Beta_IZ", "HeadBandOn", "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10", "Battery", "Intention"], axis=1)
#        print(df.columns)
#        array = df.to_numpy()
#        prediction = model.predict(array)
#        output = prediction[0]
#        print("output",output)

#        def predout(output):
#            if output == 0:
#                #resp = jsonify(({'message' : 'NO'}).format(output))
#                print("NO")
#                return ("NO".format(output))
#                #res = {"prediction_text": "NO".format(output)}
#                #return res['prediction_text']
#            else:
#                #resp = jsonify({'message' : 'YES'})
#                print("YES")
#                return ("YES")
#                #res = {"prediction_text": "YES"}
#                #return res['prediction_text']

#        resp = jsonify({'message' : 'Files successfully uploaded'})
#        resp.status_code = 201
#        #print("upload success 2")
#        return resp

#    else:
#        data = file.read()
#        filepath =  f'static/uploads/{file.filename}'
#        print(filepath)
#        data = pd.read_csv(filepath)
#        print("CSV", data.columns)
#        df = pd.DataFrame(data)
#        df = df.drop(["Delta_Pz", "Delta_Oz", "Theta_P4", "Theta_PO4", "Gamma_P8", "Gamma_PO8", "Alpha_PO7", "Alpha_LMAST", "Beta_PO3", "Beta_IZ", "HeadBandOn", "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10", "Battery", "Intention"], axis=1)
#        print(df.columns)
#        array = df.to_numpy()
#        prediction = model.predict(array)
#        output = prediction[0]
#        print("output",output)

#        def predout(output):
#            if output == 0:
#                #resp = jsonify(({'message' : 'NO'}).format(output))
#                print("NO")
#                respond = jsonify({'message' : 'NO'})
#                return respond #("NO".format(output))
#                #res = {"prediction_text": "NO".format(output)}
#                #return res['prediction_text']
#            else:
#                #resp = jsonify({'message' : 'YES'})
#                print("YES")
#                respond = jsonify({'message' : 'YES'})
#                return respond #("YES")
#                #res = {"prediction_text": "YES"}
#                #return res['prediction_text']
        
#        respond = predout(output)

#        resp = jsonify(respond)
#        resp.status_code = 500
#        print("upload error")
#        return resp


#@app.route('/sendresult', methods=['POST', 'GET'])
#def sendresult(res)

if __name__ == '__main__':
    #app.run(host = '192.168.8.107', port=3000, debug=False)
    app.run(debug=False)