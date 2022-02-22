from flask import Flask,render_template,request
import numpy as np
import math
import pickle
app=Flask(__name__)
#to load the pickle file in the python file
model=pickle.load(open('blood.pkl','rb'))

@app.route('/')
def home():
    return render_template('blood_predict.html')

@app.route('/predict',methods=['post'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    return render_template('blood_predict.html',prediction_text="{}".format(math.floor(prediction)))

if __name__ == '__main__':
    app.run(debug=True)
