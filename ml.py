import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


cols=["Time_stamp","Location","latitude","longitude","temperature","Humidity_p","Rain_mm","wind_speed","pressure","Cloud_cover","precipitation"]
data=pandas.read_csv("/content/drive/MyDrive/kolkata_west_bengal_weather_historic_template.csv",names=cols)
print(data)
time=data["Time_stamp"]
temp=data["temperature"]
humidity=data["Humidity_p"]
rain=data["Rain_mm"]
wind_speed=data["wind_speed"]
pressure=data["pressure"]
cloud=data["Cloud_cover"]
precip=data["precipitation"]


#temp VS pressure
plt.scatter(temp,pressure,color="hotpink")
plt.title("temperature VS pressure")
plt.xlabel("temperature")
plt.ylabel("pressure")
plt.show()
#pressure VS humidity
plt.scatter(temp,humidity,color="orange")
plt.title("temperature VS humidity")
plt.xlabel("temperature")
plt.ylabel("humidity")
plt.show()

#pressure VS windspeed
plt.scatter(pressure,wind_speed,color="green")
plt.title("pressure VS windspeed")
plt.xlabel("pressure")
plt.ylabel("windspeed")
plt.show()

#cloudData VS precipatation
plt.scatter(cloud,precip,color="blue")
plt.title("cloud VS precipitation")
plt.xlabel("cloud")
plt.ylabel("precipitation")
plt.show()

#humidity VS precipitation
plt.scatter(humidity,precip,color="#8E27F5")
plt.title("humidity VS precipitation")
plt.xlabel("humidity")
plt.ylabel("precipitation")
plt.show()





from threading import settrace_all_threads
#scaler functions
def stand_scaler(scale_arr):
  scale=StandardScaler()
  scaled_data = scale.fit_transform(scale_arr)
  return scaled_data

def minmax_scaler(scale_arr):
  scale=MinMaxScaler()
  scaled_data = scale.fit_transform(scale_arr)
  return scaled_data



scaled_rain=minmax_scaler(rain.values.reshape(-1, 1))
scaled_precip=minmax_scaler(precip.values.reshape(-1,1))
scaled_temp=stand_scaler(temp.values.reshape(-1, 1))
scaled_humidity=stand_scaler(humidity.values.reshape(-1, 1))
scaled_wind_speed=stand_scaler(wind_speed.values.reshape(-1,1))
scaled_pressure=stand_scaler(pressure.values.reshape(1,-1))
scaled_cloud=stand_scaler(cloud.values.reshape(-1,1))




usable_rain=scaled_rain.flatten()
usable_precip=scaled_precip.flatten()  
usable_temp=scaled_temp.flatten()
usable_humidity=scaled_humidity.flatten() 
usable_wind_speed=scaled_wind_speed.flatten()
usable_pressure=scaled_pressure.flatten()
usable_cloud=scaled_cloud.flatten()




mean_temp=np.mean(temp)
mean_humidity=np.mean(humidity)
mean_rain=np.mean(rain)
mean_precip=np.mean(precip)
mean_wind_speed=np.mean(wind_speed)
mean_pressure=np.mean(pressure)
mean_cloud=np.mean(cloud)




sd_temp=np.std(temp)
sd_humidity=np.std(humidity)
sd_rain=np.std(rain)
sd_precip=np.std(precip)
sd_wind_speed=np.std(wind_speed)
sd_pressure=np.std(pressure)
sd_cloud=np.std(cloud)




def prediction(newdata):
  predict= (newdata-mean_temp)/sd_temp
  if(predict<1.5):
    return "the conditions are normal"
  if(predict>1.5 and predict<2.0):
    return "the conditions are moderate"
  if(predict>2.0 and predict<3.0):
    return "the conditions are severe"
  if(predict>=3.0):
    return "the conditions are very severe"




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def get_condition(scaled_temp_value):
    predict_value = scaled_temp_value
    if predict_value < 1.5:
        return "normal"
    elif 1.5 <= predict_value < 2.0:
        return "moderate"
    elif 2.0 <= predict_value < 3.0:
        return "severe"
    else:
        return "very severe"


if len(usable_temp) == len(data):
    data['condition'] = [get_condition(t) for t in usable_temp]
else:
    print("Warning: Length of usable_temp does not match original data. Re-evaluating...")
    z_scores_temp = (temp - mean_temp) / sd_temp
    data['condition'] = [get_condition(t_z) for t_z in z_scores_temp]


print("Distribution of conditions:")
display(data['condition'].value_counts())


# 2. Prepare features (X) and target (y)
# We'll use the 'usable_' scaled data as features
X = pandas.DataFrame({
    'usable_humidity': usable_humidity,
    'usable_rain': usable_rain,
    'usable_precip': usable_precip,
    'usable_wind_speed': usable_wind_speed,
    'usable_pressure': usable_pressure,
    'usable_cloud': usable_cloud
})

y = data['condition']

# 3. Split the data into training and testing sets
# Using a 70/30 split and a random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)




# 4. Choose and Train a Model (Logistic Regression)
model = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter for convergence
model.fit(X_train, y_train)

# 5. Evaluate the Model
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# Example of new data input
#datainput
new_humidity = 80
new_rain = 0.5
new_precip = 0.5
new_wind_speed = 10
new_pressure = 1012
new_cloud = 45

# Scale the new data using the previously defined scaler functions
scaled_new_humidity = stand_scaler(np.array([[new_humidity]]))
scaled_new_rain = minmax_scaler(np.array([[new_rain]]))
scaled_new_precip = minmax_scaler(np.array([[new_precip]]))
scaled_new_wind_speed = stand_scaler(np.array([[new_wind_speed]]))
scaled_new_pressure = stand_scaler(np.array([[new_pressure]]))
scaled_new_cloud = stand_scaler(np.array([[new_cloud]]))

# Create a DataFrame for the new data, matching the structure of X_train
new_data_df = pandas.DataFrame({
    'usable_humidity': scaled_new_humidity.flatten(),
    'usable_rain': scaled_new_rain.flatten(),
    'usable_precip': scaled_new_precip.flatten(),
    'usable_wind_speed': scaled_new_wind_speed.flatten(),
    'usable_pressure': scaled_new_pressure.flatten(),
    'usable_cloud': scaled_new_cloud.flatten()
})

# Make a prediction
predicted_condition = model.predict(new_data_df)

print(f"For the new input data, the predicted condition is: {predicted_condition[0]}")