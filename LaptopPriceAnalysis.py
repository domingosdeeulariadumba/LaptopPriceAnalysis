# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:42:57 2023

@author: domingosdeeulariadumba
"""



""" IMPORTING LIBRARIES """


# EDA and Plotting

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as pp
pp.style.use('ggplot')


# Machine Learning Modules

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lreg
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor as xgbreg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as SS, normalize


# To save the ML model

import joblib as jbl


# For ignoring warnings

import warnings
warnings.filterwarnings('ignore')




"""" DATA CLEANING AND PREPARATION """


  
    '''
    Importing the dataset.
    '''
df = pd.read_csv("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/LaptopPrice_Dataset.csv")


    '''
    The main goal of this project is to help predicting the price of laptops
    based on their configurations. Regarding this point, it is noticed in the
    dataset entries lots of information, which will be solved with feature
    engineering on the following steps.
    '''


    '''
    Presenting the columns of the dataset.
    '''
df.columns


    '''
    Dropping the first column due to its irrelevance in our analysis.
    '''
df = df.drop(['Unnamed: 0'], axis = 1)


    '''
    Name column:
        - Here we split the 'Name' column for extracting the brand and 
          then the product.
    '''  
name = df['Name'].str.split(' ', n = 1, expand = True)

df['Brand'] = name[0]

df['Product'] = name[1].str.split('Dual Core| Hexa Core| Quad Core| Core',
                                  expand = True)[0]


    '''
    Changing the 'Processor' column:
        - We will first extract the most frequent processor types from the 
          Processor column to create a new one;  
        - then it will be created a new column with the core type;
        - endly, it will be created a new column informing the processors
          generation.
    '''
prc_type = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Intel Core i9',
            'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9']

df['ProcessorType'] = 'Other'

for k in range(len(df)):
    for i in prc_type:
        if i in str(df['Processor'][k]):
                    df['ProcessorType'][k] = i
                
core_type = ['Dual Core', 'Quad Core', 'Hexa Core', 'Octa Core']

df['ProcessorCore'] = 'Core'  

for j in range(len(df)):
    for k in core_type:
        if k in str(df['Processor'][j]):
                    df['ProcessorCore'][j] = k

df['ProcessorGen'] = 'Undefined'

for i in range(len(df)):
    for j in range(7,12):
        if str(j)+'th' in (df['Processor'][i]) or str(j)+'th' in (df['Name'][i]):
            df['ProcessorGen'][i] = str(j)+'th'


    '''
    Rearranging the 'RAM' column by extracting just the number in its entries.
    '''
df['RAM'] = df['RAM'].str.split(' ', expand = True)[0]

df['RAM'] = df['RAM'].astype(int)  

    '''
    The RAM of the 390th entry was not inserted in its equivalent column. This
    information, 8 GB, is seen in the 'Name' column.
    '''
df['RAM'][df['RAM'] == 'Upgradable'] = 8

df['RAM'] = df['RAM'].astype(int)


    '''
    Operating System Information:
        This attribute will be reduced to three: 'Windows', 'Mac' and 'Other
        OS'. The latter will be used in case the Operating System is not one
        of the formers, dicarding the rest of information of this column.
    '''      
df['OpSys'] = ''

for k in range(len(df)):
    if 'Windows' in str(df['Operating System'][k]):
        df['OpSys'][k] = 'windows'
    elif 'Mac' in str(df['Operating System'][k]):
       df['OpSys'][k] = 'Mac'
    else:
       df['OpSys'][k] = 'Other OS' 
       
       
    '''
    Organizing the 'Storage' column:
        - For this step The 'TB' unit will first be replaced by '1000 GB';
        - it will then be extracted just the numeric part of the storage
          information;
        - lastly, this column will be split to differ 'HDD' from 'SSD' storage.
    '''
df['Storage'] = df['Storage'].replace('1 TB', '1000 GB', regex = True)

Storage = df['Storage'].str.split('|', expand = True)

df['HDD Storage'] = ''

df['SSD Storage'] = ''

for j in range(len(Storage)):
    if 'HDD' in str(Storage[0][j]):
        df['HDD Storage'][j] = Storage[0][j]
        df['SSD Storage'][j] = Storage[1][j]
    else:
        df['SSD Storage'][j] = Storage[0][j]
        df['HDD Storage'][j] = Storage[1][j]

df['HDD Storage'] = df['HDD Storage'].str.split('GB', expand = True)[0]

df['HDD Storage'] = df['HDD Storage'].fillna(0).astype(int)

df['SSD Storage'] = df['SSD Storage'].str.split('GB', expand = True)[0]

df['SSD Storage'] = df['SSD Storage'].fillna(0).astype(int)
   
    '''
    The storage of the 201st entry was not properly iserted. The correct value,
    1 TB (1000 GB), was taken from the 'Name' column.
    '''
df['SSD Storage'][df['SSD Storage'] == 'M.2 Slot for SSD Upgrade'] = 1000

df['SSD Storage'] = df['SSD Storage'].fillna(0).astype(int)


  '''
   Display column:
        - We first replace the records with the Display lenght in 'inch'  to
          'cm';
        - the Display information not recorder in 'cm' is replaced by 'other';
        - ultimately, the rest of information is discarded, by splitting the
        Display column by 'cm'.
    '''
df['Display'] = df['Display'].replace('15.6 inches', '39.62 cm', regex = True)

for i in range(len(df)):
            if 'cm' not in str(df['Display'][i]):
                    df['Display'][i] = 'Other'
                    
df['Display'] = df['Display'].str.split('cm', expand = True)[0]


    '''
     Warranty column:
          - for this step, we simply split the term 'Year' and 'Months',
          discarding the rest of information.
      '''
df['Warranty'] = df['Warranty'].replace('One-year', '1 Year', regex = True)

for i in range(len(df)):
    if 'Months' in str(df['Warranty'][i][:9]):
        df['Warranty'][i] = str(float(str(df['Warranty'][i])[:2])/12)+ ' Year'
    elif 'Year' not in str(df['Warranty'][i][:7]):
        df['Warranty'][i] = 'Other'
        
df['Warranty'] = df['Warranty'].str.split('Year', expand = True)[0]

for i in range(len(df)):
    if df['Warranty'][i] != 'Other':
       df['Warranty'][i] =  df['Warranty'][i] + ' Year'


    '''
    Organizing the 'Price' column:
        - On the following code, we replace the '₹' currency symbol
          and, lastly, ',' to white space.
    '''  
df['Price'] = df['Price'].replace(['₹', ','], '', regex = True).astype(float)


    '''
    Concatenating the clean data in a new dataset.
    '''
df_ltp=pd.concat([df['Brand'], df['Product'], df['ProcessorType'],
                  df['ProcessorCore'], df['ProcessorGen'], df['RAM'],
                  df['OpSys'], df['HDD Storage'], df['SSD Storage'], 
                  df['Display'], df['Warranty'], df['Rating'], df['Price']],
                 axis = 1)




"""" EXPLORATORY DATA ANALYSIS """



    '''
    Main statistical parameters of the explanatories and target variables.
    '''
df_ltp.describe().iloc[:,:-1]

df_ltp.describe()['Price']


     '''
     Presenting the entries overall information, such as type and missing
     values. As can be noticed by running the code below, at this point,
     everything seems right.
     '''
df_ltp.info()


    '''
    Distribution plots.
    '''

numeric_vls = list(df_ltp.describe().columns)

for i in numeric_vls:
    sb.set(rc = {'figure.figsize':(20,10)}, font_scale = 1)
    sb.displot(df_ltp[i])
    pp.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/{0}_displot.png'.format(i))
    pp.close()

    '''
    KDE plots.
    '''
for j in numeric_vls:
    pp.figure()
    sb.set(rc = {'figure.figsize':(16,10)}, font_scale = 2)
    sb.kdeplot(df_ltp[j])
    pp.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/{0}_kdeplot.png'.format(j))
    pp.close()

   
    '''
    How the laptop price varies? To answer this question, will be adopted the
    following approach:
        - For all the attributes, excepting the 'Rating, it will be used the
          bar plot to ilustrate this variation;
        - for the 'Rating' entries will be used a strip plot. Prior to this
          step, for the other numeric attributes will be used the scatter plot.
    '''
brplt_vars = ['Brand']+list(df_ltp.columns)[2:-2]

for j in brplt_vars:
    pp.figure()
    sb.set(rc = {'figure.figsize':(20,10)}, font_scale = 2)
    sb.barplot(x = df_ltp[j], y = df_ltp['Price'])
    pp.xticks(rotation = 55)
    pp.savefig ('C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/{0}_barplot.png'.format(j))
    pp.close()
    
sb.pairplot(df_ltp, x_vars = numeric_vls[:3],
            y_vars=['Price'], height = 8, aspect = .8, kind = 'scatter')
pp.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/scatterplot.png")
pp.close()

sb.set(rc={'figure.figsize':(20,10)}, font_scale= 2)
sb.stripplot(x='Rating', y='Price', data=df_ltp)
pp.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/Rating_stripplot.png")
pp.close()


  '''
  Correlation heatmap.
  '''
sb.heatmap(df_ltp.corr(), annot=True)
pp.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/correlation_heatmap.png")
pp.close()




""" LAPTOP PRICE PREDICTION MODEL """




    '''
    To build the ML model it is vital to first hot-encode the categorical entries.
    Then, we'll normalize, scale and reduce the dimensionality of these
    attributes
    '''

X = pd.get_dummies(df_ltp.iloc[:,:-1], drop_first = True)

y = df_ltp.iloc[:,-1]

def NSR(X):
    
    # Normalization
    X_norm = normalize(X)
        
    # Escaling   
    X_scld = SS().fit_transform(X_norm)
        
    # Dimensionality reduction                       
    pcomp = PCA(n_components = X.shape[1]).fit(X_scld)
    pcomp_var = pcomp.explained_variance_ratio_
    pcomp_var_cumsum = np.cumsum(np.round(pcomp_var, decimals = 4)*100)
    print('Principal Components: ', np.unique(pcomp_var_cumsum).size)
    
    pp.plot(pcomp_var_cumsum)
    pp.savefig("C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/principalcomponents.png")
    pp.close()
                     
    final_comp = PCA(n_components = np.unique(pcomp_var_cumsum).size).fit_transform(X_scld)
       
    return pd.DataFrame(final_comp)

x = NSR(X)


    '''
    Runnig the code above, we notice a reduction of 19 components.
    Next, we split this data into train and test sets. To train this model,
    we'll use the regression approach of SciKit-Learn and XGBoost to then pick the
    one with the best performance.
    '''
x_train, x_test, y_train, y_test = tts(x,y, test_size = 0.2, random_state = 97)


    '''
    SciKit-Learn's regression model.
    '''
LinReg = lreg()
LinReg.fit(x_train, y_train)
y_pred = LinReg.predict(x_test)

print('Test set (RMSE):', mean_squared_error(y_test, y_pred, squared=False))
print('NRMSE:', (mean_squared_error(y_test, y_pred, squared=False)/(y.max()-y.min())))
print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))


    '''
    XGBoost's regression model.
    '''
XGBreg = xgbreg()
XGBreg.fit(x_train, y_train)
y_pred1 = XGBreg.predict(x_test)
 
print('Test set (RMSE):', mean_squared_error(y_test, y_pred1, squared=False))
print('NRMSE:', (mean_squared_error(y_test, y_pred1, squared=False)/(y.max()-y.min())))
print('Coefficient of determination (R^2):', r2_score(y_test, y_pred1))




        """
        ClOSING REMARKS:
        
        Comparing the model parameters by running the two algorithms, we have
        noticed the below outcomes:
            - The regression model from SciKit-Learn fails to predict the price
              of laptops, with a negative coefficient of determination;
            - the XGBoost regressor outdid the prior approach, with a R2 Score
              of about 63%. This model will then be saved using Joblib library.
        """
jbl.dump(XGBreg,"C:/Users/domingosdeeularia/Documents/notyourbusiness/CodingAndAnalytics/Projects/LaptopPriceAnalysis/LaptopPriceModel.sav")

_______________________________________________________________________end___________________________________________________________________
