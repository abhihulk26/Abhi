import numpy as np 
import pandas as pd
from flask import Flask,request,jsonify,render_template
import joblib
app=Flask(__name__)
model1=joblib.load("model.pkl")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    # '''
    int_features = [[int(x) for x in request.form.values()]]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0],2)
    #print(output)
    # index_target=pd.Series(["Extremely Weak","Weak" ,"Normal" ,"Overweight","Obesity" ,"Extreme Obesity"])
    

    # input=["present_price",'kms_driven','fuel_type','seller_type','transmission']
    result=model1.predict(int_features)

    #result=list(result.values)
    #result=str(result)
    return render_template('index.html', prediction_text='Current car price is Rs.{} lacs'.format(round(result[0],2)))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model1.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)