import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import warnings

import datetime
#warnings.filterwarnings("ignore", category=DeprecationWarning) 

import yahoo_finance
from yahoo_finance import Share
import pandas as pd
#import test3


#name = text_contents
name= input("Please enter the stock keyword:   ")
#date_start = input("enter start date in the order YY-MM-DD  ")
date_start= "2015-01-01"
date_end = datetime.datetime.now().strftime ("%Y-%m-%d")
#date_end = input("enter end date: ")
symbol = Share(name)
google_data = symbol.get_historical(date_start, date_end)
google_df = pd.DataFrame(google_data)

# Output data into CSV
output_path = "D:\workspace"
def make_filename(name, directory ="stockprediction"):
    return output_path + "/" + directory + "/" + name +  ".csv"
google_df.to_csv(make_filename(name,) )

print("data received \n")
dates = []
prices = []

model = SVR(cache_size=7000)
#name = 'SBIN.NS.csv'
#print("hello \n")

def get_data(name):
    with open(name, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in reversed(list(csvFileReader)):
            dates.append(int(row[0].split('-')[0]))
            print(dates)
            prices.append(float(row[1]))
            #print(prices)
        return


def get_high(name):
    with open(name, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in reversed(list(csvFileReader)):
            dates.append(int(row[0].split('-')[0]))
            print(dates)
            prices.append(float(row[4]))
            #print(prices)
        return

def get_open(name):
    with open(name, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in reversed(list(csvFileReader)):
            dates.append(int(row[0].split('-')[0]))
            print(dates)
            prices.append(float(row[6]))
            #print(prices)
        return

def get_low(name):
    with open(name, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in reversed(list(csvFileReader)):
            dates.append(int(row[0].split('-')[0]))
            print(dates)
            prices.append(float(row[5]))
           # print(prices)
        return

  
  #
  #gaurav anex, a-17, saraswat bank, near lajja show room. ground floor.  

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))  # @UndefinedVariable
    
    #svr_lin = SVR(kernel='linear', C=1e3)
    #svr_poly = SVR(kernel = 'poly', C=1e3 , degree = 2 )
    svr_rbf = SVR(kernel = 'rbf', C=1e3 , gamma = 0.1)
    #svr_lin.fit(dates, prices)    
    #svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.scatter(dates, prices, color='black', label = 'data')
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label ='RBF model')
   # plt.plot(dates, svr_lin.predict(dates), color= 'green', label ='linear model')
    #plt.plot(dates, svr_poly.predict(dates), color= 'blue', label ='polynomial model')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title('Support Vector Regresion')
    plt.show()
    
    return svr_rbf.predict(x)[0]
# ,svr_lin.predict(x)[0], svr_poly.predict(x)[0]

data_name = name + '.csv'
ques= input("would you like to predict the high/low/open/close ??    ")
if ques == "close":

    get_data(data_name)

elif ques =="high":
    get_high(data_name)
    
elif ques == "open":
    get_open(data_name)
    
elif ques == "low":
    get_low(data_name)
    
# 5 companies, 1jan16 to 31dec
#why difference betweenn closing n opening
#how to find accuracy?
# prepare table 1feb16 to 28feb16 


predicted_price =   predict_prices(dates, prices, 0)
print ("\n \n\n \n \n \n \n Precidected price is :", predicted_price, "\n \n")
#print ("hello")

