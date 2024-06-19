import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#import ridge and regressor model
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

#Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        chlorides=float(request.form.get("chlorides"))
        density=float(request.form.get("density"))
        pH=float(request.form.get("pH"))
        alcohol=float(request.form.get("alcohol"))

        new_data_scaled=standard_scaler.transform([[chlorides,density,pH,alcohol]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')
    




if __name__=="__main__":
    app.run(host="0.0.0.0")
