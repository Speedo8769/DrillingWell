from flask import Flask,render_template,request
import joblib

app=Flask(__name__)

model=joblib.load('models/model.h5')
scaler=joblib.load('models/scaler.h5')

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    all_data=request.args

    depth=float(all_data['Depth(m)'])
    weight_on_bit=float(all_data['weight on bit (kg)'])
    temp_out=float(all_data['Temp Out( degC)'])
    temp_in=float(all_data['Temp In(degC)'])
    pump_press=float(all_data['Pump Press (KPa)'])
    hookload=float(all_data['Hookload (kg)'])
    surface_torque=float(all_data['Surface Torque (KPa)'])
    rotary_speed=float(all_data['Rotary Speed (rpm)'])
    flow_in=float(all_data['Flow In(liters/min)'])
    WH_pressure=float(all_data['WH Pressure (KPa)'])
    Temp_rate=temp_out/temp_in

    data=[depth, weight_on_bit, pump_press, hookload, surface_torque, rotary_speed, flow_in, WH_pressure, Temp_rate]
    
    pred=round(model.predict(scaler.transform([data]))[0])
    return render_template('prediction.html',x=pred)


if __name__=='__main__':
    app.run()