import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Load Dataset
store_sales = pd.read_csv('C:/Users/dell/Desktop/DS/india.csv')
store_sales.head(10)
store_sales.info()
#disregard the columns representing the Store
store_sales = store_sales.drop(['GDP_Construction_Rs_Crs','GDP_Real Estate_Rs_Crs','Oveall_GDP_Growth%','water_source','limestone','Coal(in metric tones)','Home_Interest_Rate','Trasportation_Cost','order_quantity_milliontonne','unit_price','Total_Price','month'], axis=1)

#Convert ‘date’ column in the dataset is of the ‘object’ datatype to DateTime datatype
store_sales['Date'] = pd.to_datetime(store_sales['Date'])
store_sales['Date'] = store_sales['Date'].dt.to_period('M')
monthly_sales = store_sales.groupby('Date').sum().reset_index()

monthly_sales['Date'] = monthly_sales['Date'].dt.to_timestamp()
monthly_sales.head(10)

#visualize the monthly sales
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['Date'], monthly_sales['Sales'])
plt.xlabel('Date')
plt.xlabel('Sales')
plt.title("Monthly Customer Sales")
plt.show()

#call the diff on the ‘sales’ columns to make our sale data stationery:
monthly_sales['sales_diff'] = monthly_sales['Sales'].diff()
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)  

#Monthly sales in different plot
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['Date'], monthly_sales['sales_diff'])
plt.xlabel('Date')
plt.xlabel('Sales')
plt.title("Monthly Customer Sales Diff")
plt.show()    

#dealing with the stationary sale data 
supverised_data = monthly_sales.drop(['Date','Sales'], axis=1)    
# not required
for i in range(1,13):
    col_name = 'month_' + str(i)
    supverised_data[col_name] = supverised_data['sales_diff'].shift(i)
supverised_data = supverised_data.dropna().reset_index(drop=True)
supverised_data.head(10)

#split the data into training and test data:
train_data = supverised_data[:12]
test_data = supverised_data[-12:]
print('Train Data Shape:', train_data.shape)
print('Test Data Shape:', test_data.shape)

#MinMaxScaler():
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)


test_data = scaler.transform(test_data)    
#Train the model
X_train, y_train = train_data[:,1:], train_data[:,0:1]
X_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()
print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print('X_test Shape:', X_test.shape)
print('y_test Shape:', y_test.shape)

sales_dates = monthly_sales['Date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

#extract the actual monthly sale values of the last 13 months
act_sales = monthly_sales['Sales'][-13:].to_list()


#Forecast Sales using Linear Regression
linreg_model = LinearRegression()
linreg_model.fit(X_train, y_train)
linreg_pred = linreg_model.predict(X_test)

# call the ‘inverser_transform’ function of the MinMaxScaler
linreg_pred = linreg_pred.reshape(-1,1)
linreg_pred_test_set = np.concatenate([linreg_pred,X_test], axis=1)
linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)

result_list = []
for index in range(0, len(linreg_pred_test_set)):
    result_list.append(linreg_pred_test_set[index][0] + act_sales[index])
linreg_pred_series = pd.Series(result_list,name='linreg_pred')
predict_df = predict_df.merge(linreg_pred_series, left_index=True, right_index=True)

#trained Linear Regression model by comparing the predicted sale values with the actual sale values
linreg_rmse = np.sqrt(mean_squared_error(predict_df['linreg_pred'], monthly_sales['Sales'][12:]))
linreg_mae = mean_absolute_error(predict_df['linreg_pred'], monthly_sales['Sales'][12:])
linreg_r2 = r2_score(predict_df['linreg_pred'], monthly_sales['Sales'][12:])
print('Linear Regression RMSE: ', linreg_rmse)
print('Linear Regression MAE: ', linreg_mae)
print('Linear Regression R2 Score: ', linreg_r2)
    
#visualize the predictions against the actual values:
plt.figure(figsize=(15,7))
plt.plot(monthly_sales['Date'], monthly_sales['Sales'])
plt.plot(predict_df['Date'], predict_df['linreg_pred'])
plt.title("Customer Sales Forecast using Linear Regression")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show() 

#Forecast Sales using Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)   

#inverse_transform’ function of the MinMaxScaler():
rf_pred = rf_pred.reshape(-1,1)
rf_pred_test_set = np.concatenate([rf_pred,X_test], axis=1)
rf_pred_test_set = scaler.inverse_transform(rf_pred_test_set)

#evaluate the model metrics by comparing the predicted sale amount with the actual sale value:
rf_rmse = np.sqrt(mean_squared_error(predict_df['rf_pred'], monthly_sales['Sales'][12:]))
rf_mae = mean_absolute_error(predict_df['rf_pred'], monthly_sales['Sales'][12:])
rf_r2 = r2_score(predict_df['rf_pred'], monthly_sales['Sales'][12:])
print('Random Forest RMSE: ', rf_rmse)
print('Random Forest MAE: ', rf_mae)
print('Random Forest R2 Score: ', rf_r2)

plt.figure(figsize=(15,7))
plt.plot(monthly_sales['ate'], monthly_sales['Sales'])
plt.plot(predict_df['Date'], predict_df['rf_pred'])
plt.title("Customer Sales Forecast using Random Forest")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Original Sales", "Predicted Sales"])
plt.show()