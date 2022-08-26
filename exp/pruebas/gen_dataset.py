import pandas as pd
import numpy as np
import os
import sys

cwd = os.getcwd()
BASE_DIR = os.path.dirname(os.path.dirname(cwd))
sys.path.insert(0, BASE_DIR)
name = "bs_as_temp_dataset"
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Stations, Daily, Hourly

# Configuration
Daily.cores = 12

# Time period
start = datetime(2021, 1, 1)
end = datetime(2022, 8, 1)

# Get closest weather station
stations = Stations()
stations = stations.bounds((-34.49859175010478, -58.86302175193546), (-34.89129339488858, -58.0695427780878)) #EZE;AEP y capital
stations = stations.inventory("daily", (start, end))
station = stations.fetch()
print(station)
#Get daily data
data = Hourly(station, start, end)
#data["prcp"].fillna(0,inplace=True)
data = data.normalize()
#data = data.interpolate()
data = data.fetch()
#
# Plot chart
print(data)
data[data.index.get_level_values('station') == '87582'].plot(y=["temp", "dwpt", "rhum", "prcp","wspd","wdir","pres"], subplots=True)
data[data.index.get_level_values('station') == '87585'].plot(y=["temp", "dwpt", "rhum", "prcp","wspd","wdir","pres"], subplots=True)
data[data.index.get_level_values('station') == '87576'].plot(y=["temp", "dwpt", "rhum", "prcp","wspd","wdir","pres"], subplots=True)

data=data.reset_index(["station"]).sort_index()

data=data.drop(["station","snow","wpgt","tsun","coco"],axis=1)
data=data.groupby(data.index).mean()#.apply(lambda g: g.mean(skipna=False))
print(data)
data["prcp"].fillna(0,inplace=True)
data.interpolate(method='linear',axis=0,inplace=True)
data.plot(y=["temp", "dwpt", "rhum", "prcp","wspd","wdir","pres"], subplots=True)
data.to_csv('datasets\%s.csv' % name,columns=["temp", "dwpt", "rhum", "prcp","wspd","wdir","pres"])
plt.show()
print(data)
