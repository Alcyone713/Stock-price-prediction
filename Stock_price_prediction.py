import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 

df = data.get_data_tiingo('GOOGL', api_key="14a6f948d2203e5110a13d65f0599d4360cca0f5" )
df=df.reset_index()
df=df.drop(['date','symbol','adjClose','adjLow','adjHigh','adjOpen','adjVolume','divCash','splitFactor'], axis=1)
#print(df.tail(10))
plt.plot(df.close)
ma30=df.close.rolling(30).mean()
#print(ma30)
plt.plot(df.close, "#e52165")
plt.plot(ma30,'#0d1137')
plt.show()
training = pd.DataFrame(df['close'][0:int(len(df)*0.7)]) 
testing = pd.DataFrame(df['close'][int(len(df)*0.7):int(len(df))])
#print(training.shape)
#print(testing.shape)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
training_array = scaler.fit_transform(training)
#print(training_array)
x_train=[]
y_train=[]
for i in range(100,training_array.shape[0]):
  x_train.append(training_array[i-100:i])
  y_train.append(training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape)
print(y_train.shape)



