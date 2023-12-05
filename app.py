from flask import Flask,request,render_template,url_for
import pandas as pd
import pickle

app = Flask(__name__)

model1 = pickle.load(open('model1.pkl', 'rb')) 

@app.route('/')
def  index():
    return render_template('index.html') 

@app.route('/service')
def  service():
    return render_template('service.html') 

@app.route('/ui')
def  ui():
    return render_template('ui.html') 

@app.route('/doctor')
def  doctor():
    return render_template('doctor.html') 

@app.route('/about')
def  about():
    return render_template('about.html') 


@app.route('/predict', methods=['POST', 'GET']) 
def predict():
    
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    
    
    df1 = pd.DataFrame([{'gender': int(data1) ,'age': float(data2) ,'hypertension': int(data3) ,'heart_disease': int(data4) ,'smoking_history': int(data5) ,'bmi': float(data6) ,'HbA1c_level': float(data7) ,'blood_glucose_level': int(data8)}])

    prediction = model1.predict(df1)
    
    if prediction==0:
        prediction= " dont have diabetes"
    else:
        prediction=" have diabetes"
     
    print(prediction) 
    
   
    return render_template('op.html', pred='you{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
