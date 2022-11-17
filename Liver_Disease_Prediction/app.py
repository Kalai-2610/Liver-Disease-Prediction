from flask import Flask, request, render_template, jsonify
import flask_cors
from model import RFT_Model
from model_copy import SVC_Model
import pandas as pd

app = Flask(__name__)
flask_cors.CORS(app)
ML_model = RFT_Model()

@app.route('/')
@flask_cors.cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict')
@flask_cors.cross_origin()
def predict():
    global ML_model
    age = int(request.args.get('age'))
    gender = int(request.args.get('gender'))
    tb = float(request.args.get('tb'))
    db = float(request.args.get('db'))
    ap = float(request.args.get('ap'))
    aa = float(request.args.get('aa'))
    asa = float(request.args.get('asa'))
    a = float(request.args.get('a'))
    tp = float(request.args.get('tp'))
    agr = float(request.args.get('agr'))

    count = 0
    if( 0.22<=tb and tb<=1 ):
        count += 1
    if( 0<=db and db<=.2 ):
        count += 1
    if( 110<=ap and ap<=310 ):
        count += 1
    if( 5<=aa and aa<=45 ):
        count += 1
    if( 5<=asa and asa<=40 ):
        count += 1
    if( 3.5<=a and a<=5 ):
        count += 1
    if( 7.2<=tp and tp<=8 ):
        count += 1
    if( 1.7<=agr and agr<=2.2 ):
        count += 1
    if( 0.5<= count/8):
        count = True
    else:
        count = False

    data = pd.DataFrame({'Age':[age],'Gender':[gender],'Total_Bilirubin':[tb],'Direct_Bilirubin':[db],'Alkaline_Phosphotase':[ap],'Alamine_Aminotransferase':[aa],'Aspartate_Aminotransferase':[asa],'Albumin':[a],'Total_Protiens':[tp],'AG_Ratio':[agr]})
    res = ML_model.predict(data)
    res = int(res*100)
    if count:
        res = 100 - res
    response = jsonify({'result': res})
    return response

if __name__ == '__main__':
    app.run(host="127.0.0.1",port="5000",debug=True)