import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import time
from datetime import datetime

app = Flask(__name__)
#Loading Random forest model, which is giving least mse
model = joblib.load('rf.pkl') #decision_tree.pkl

# interval obtained after calculating prediction interval
interval=[49188.693,48970]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    dp=np.zeros(53)
    dp[0]=int(features[0])
    dp[1]=int(features[1])
    dp[2]=int(features[2])
    dp[3]=bool(features[3])
    dp[4]=bool(features[4])
    dp[5]=bool(features[5])
    
    opened_date=pd.to_datetime(features[6],format='%d/%m/%Y %H:%M')
    updated_date=pd.to_datetime(features[7],format='%d/%m/%Y %H:%M')  
    
    opened_time=time.mktime(datetime.strptime(str(opened_date),"%Y-%m-%d %H:%M:%S").timetuple())
        
    opened_hour=opened_date.hour
    dp[6] = np.sin(2 * np.pi * opened_hour/23.0)
    dp[7] = np.cos(2 * np.pi * opened_hour/23.0)
      
    opened_day=opened_date.strftime("%w")
    dp[8]= np.sin(2 * np.pi * int(opened_day)/6)
    dp[9] = np.cos(2 * np.pi *int(opened_day)/6)
    
    updated_hour=updated_date.hour
    dp[10] = np.sin(2 * np.pi * updated_hour/23.0)
    dp[11] = np.cos(2 * np.pi * updated_hour/23.0)
      
    updated_day=updated_date.strftime("%w")
    dp[12]= np.sin(2 * np.pi * int(updated_day)/6)
    dp[13] = np.cos(2 * np.pi * int(updated_day)/6)
    
    dp[14]=(updated_date-opened_date).total_seconds()
    
    incident_state={'Active':1,'New':2,'Resolved':3,'Closed':4,'Awaiting User Info':5}
    incident_state_index=incident_state[features[8]]
    dp[14+incident_state_index]=1
    
    made_sla={'True':1,'False':0}
    dp[20]=made_sla[features[3]]
    
    caller_ids={'Caller 1904':1,'Caller 4514':2,'Caller 290':3,'Caller 1441':4,
                'Caller 93':5,'Caller 4414':6,'Caller 3763':7,'Caller 2471':8,
                'Caller 3479':9,'Caller 3160':10,'Caller 2737':11,'Caller 298':12,
                'Caller 1270':13,'Caller 5093':14,'Caller 1531':15}
    caller_index=caller_ids[features[9]]
    dp[20+caller_index]=1

    impact={'Medium':1,'Low':2}
    impact_index=impact[features[10]]
    dp[35+impact_index]=1

    priority={'Moderate':1,'Low':2,'High':3}
    pr_index=priority[features[11]]
    dp[37+pr_index]=1
    
    knowledge={'True':0,'False':1}
    dp[41]=knowledge[features[4]]
    
    upc={'True':0,'False':1}
    dp[42]=upc[features[5]]
    
    codes={'code 6':1,'code 7':2,'code 9':3,'code 8':4,'code 5':5,
           'code 1':6,'code 10':7,'code 11':8,'code 16':9,'code 4':10}
    
    code_index=codes[features[12]]
    dp[42+code_index]=1
    
    prediction = model.predict(dp.reshape(1,-1))
    
    # we have applied log to target variables while training, so applying exp to predicted values
    resolved_exp=np.expm1(prediction[:,0])
    #Adding opned_at time and resolved time to get unix time 
    resolved_unix=opened_time+resolved_exp
    
    #Converting unix time stamp to date time format
    dt=datetime.fromtimestamp(resolved_unix)
    resolved_dt=dt.strftime('%Y-%m-%d %H:%M')
    print('Resolved:',resolved_exp,opened_time,resolved_unix)
    
    #Lower limit of prediction interval
    resolved_exp=np.expm1(prediction[:,0])-interval[0]
    resolved_unix=opened_time+resolved_exp
    dt=datetime.fromtimestamp(resolved_unix)
    resolved_lower_dt=dt.strftime('%Y-%m-%d')
    print('Lower range of resolved:',resolved_exp)
    
    #Upper limit of prediction interval
    resolved_exp=np.expm1(prediction[:,0])+interval[0]
    resolved_unix=opened_time+resolved_exp
    dt=datetime.fromtimestamp(resolved_unix)
    resolved_upper_dt=dt.strftime('%Y-%m-%d')
    print('Upper range of resolved:',resolved_exp)
    
    closed_exp=np.expm1(prediction[:,1])
    closed_unix=opened_time+closed_exp
    dtc=datetime.fromtimestamp(closed_unix)
    closed_dt=dtc.strftime('%Y-%m-%d %H:%M') 
    
    closed_exp=np.expm1(prediction[:,1])-interval[1]
    closed_unix=opened_time+closed_exp
    dtc=datetime.fromtimestamp(closed_unix)
    closed_lower_dt=dtc.strftime('%Y-%m-%d') 
    print('Lower range of Closed:',closed_exp)
    
    closed_exp=np.expm1(prediction[:,1])+interval[1]
    closed_unix=opened_time+closed_exp
    dtc=datetime.fromtimestamp(closed_unix)
    closed_upper_dt=dtc.strftime('%Y-%m-%d') 
    print('Upper range of Closed:',closed_exp)
    
    return render_template('predict.html',
                           resolved_date='Probable resolving date: {}'.format(resolved_dt),
                           resolved_range='Resolved date lies between {} and {}'.format(resolved_lower_dt,resolved_upper_dt),                          
                           closed_date='Probable closing date: {}'.format(closed_dt),
                           closed_range='Closed date lies between {} and {}'.format(closed_lower_dt,closed_upper_dt))

if __name__ == "__main__":
    app.run()
    #app.run(host='0.0.0.0', port=7898)