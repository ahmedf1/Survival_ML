

# importing some libraries to do the math and graphing
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt


# load the data set using the python pandas library

#first define the names of the columns

col_names = [
    'Customer Name', 'Location', 'Age', 'Scaled Age', 'Credit Score', 'Scaled Credit Score', 'Income(Thousands)', 'Scaled Income', 'Household Size', 'Scaled HouseholdSize', 'Number of Reported Internet Slowdowns', 'Scaled Reported Internet Slowdowns', 'Number of Cable Outages', 'Scaled Number of Cable Outages', 'Number of Calls to CSR', 'CSR Calls Scaled', 'Number Of Call Drops', 'Call Drop Scaled', 'Fios Competitive Zone', 'Fios Scaled', 'Internet Package Level', 'Cable Package Level', 'Phone Package Level', 'Scaled Service Level', 'Months Subscribed', 'Retention Cost', 'Customer Lifetime Value', 'Customer State'
    ]

df = pd.read_csv('/Users/Farhad_Ahmed/Desktop/Altice/Cleaned Existing Customer Data - Sheet1.csv', header=None,delim_whitespace=False,skiprows=1,delimiter=',', names=col_names,na_values='?')

print(df)
print(df.columns.tolist())


df1=np.stack((df['Scaled Age'],df['Scaled Credit Score'],df['Scaled Income'], df['Scaled HouseholdSize'], df['Scaled Reported Internet Slowdowns'], df['Scaled Number of Cable Outages'], df['CSR Calls Scaled'], df['Call Drop Scaled'], df['Fios Scaled'], df['Scaled Service Level'], df['Months Subscribed'], df['Retention Cost'], df['Customer Lifetime Value'], df['Customer State'])).T
                                                                                                                
df2=(df1[~np.isnan(df1).any(axis=1)])
print(df2.shape)

print()
