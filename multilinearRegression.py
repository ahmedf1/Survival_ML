

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
new_col_names = col_names[:24]
#print(new_col_names)

# open file /
df = pd.read_csv('/Users/Farhad_Ahmed/Desktop/Altice/alticeDataPlatform/src/Template/Cleaned Existing Customer Data - Sheet1.csv', header=None,delim_whitespace=False,skiprows=1,delimiter=',', names=col_names,na_values='?')
newCustomerdf = pd.read_csv('/Users/Farhad_Ahmed/Desktop/Altice/Cleaned Data for NewCustomers - Sheet1.csv', header=None,delim_whitespace=False,skiprows=1,delimiter=',', names=new_col_names,na_values='?')


#print(df)
print(newCustomerdf.columns.tolist())


MS_df1=np.stack((df['Scaled Age'],df['Scaled Credit Score'],df['Scaled Income'], df['Scaled HouseholdSize'], df['Scaled Reported Internet Slowdowns'], df['Scaled Number of Cable Outages'], df['CSR Calls Scaled'], df['Call Drop Scaled'], df['Fios Scaled'], df['Scaled Service Level'], df['Months Subscribed'])).T


RC_df1=np.stack((df['Scaled Age'],df['Scaled Credit Score'],df['Scaled Income'], df['Scaled HouseholdSize'], df['Scaled Reported Internet Slowdowns'], df['Scaled Number of Cable Outages'], df['CSR Calls Scaled'], df['Call Drop Scaled'], df['Fios Scaled'], df['Scaled Service Level'], df['Retention Cost'])).T

CLV_df1=np.stack((df['Scaled Age'],df['Scaled Credit Score'],df['Scaled Income'], df['Scaled HouseholdSize'], df['Scaled Reported Internet Slowdowns'], df['Scaled Number of Cable Outages'], df['CSR Calls Scaled'], df['Call Drop Scaled'], df['Fios Scaled'], df['Scaled Service Level'], df['Customer Lifetime Value'])).T

CS_df1=np.stack((df['Scaled Age'],df['Scaled Credit Score'],df['Scaled Income'], df['Scaled HouseholdSize'], df['Scaled Reported Internet Slowdowns'], df['Scaled Number of Cable Outages'], df['CSR Calls Scaled'], df['Call Drop Scaled'], df['Fios Scaled'], df['Scaled Service Level'], df['Customer State'])).T
              #df['Retention Cost'], df['Customer Lifetime Value'], df['Customer State'])).T


MS_df1=np.stack((df['Scaled Age'],df['Scaled Credit Score'],df['Scaled Income'], df['Scaled HouseholdSize'], df['Scaled Reported Internet Slowdowns'], df['Scaled Number of Cable Outages'], df['CSR Calls Scaled'], df['Call Drop Scaled'], df['Fios Scaled'], df['Scaled Service Level'], df['Months Subscribed'])).T
                                                                                                                
MS_df2=(MS_df1[~np.isnan(MS_df1).any(axis=1)])
RC_df2=(RC_df1[~np.isnan(RC_df1).any(axis=1)])
CLV_df2=(CLV_df1[~np.isnan(CLV_df1).any(axis=1)])
CS_df2=(CS_df1[~np.isnan(CS_df1).any(axis=1)])

#print(df2.shape)

MS_df3 = MS_df2[:,:10]
RC_df3 = RC_df2[:,:10]
CLV_df3 = CLV_df2[:,:10]
CS_df3 = CS_df2[:,:10]
#print(MS_df3)

MS_x = np.array(MS_df3)
MS_y = np.array(MS_df2[:,10:])

RC_x = np.array(RC_df3)
RC_y = np.array(RC_df2[:,10:])

CLV_x = np.array(CLV_df3)
CLV_y = np.array(CLV_df2[:,10:])

CS_x = np.array(CS_df3)
CS_y = np.array(CS_df2[:,10:])

#print(MS_x.shape)
#print(MS_y.shape)

MS_n = MS_x.shape[0]

RC_n = RC_x.shape[0]

CLV_n = CLV_x.shape[0]

CS_n = CS_x.shape[0]
#print(MS_n)


MS_a = np.ones((MS_y.shape[0],1))
MS_x = np.hstack((MS_a , MS_x))

RC_a = np.ones((RC_y.shape[0],1))
RC_x = np.hstack((RC_a , RC_x))

CLV_a = np.ones((CLV_y.shape[0],1))
CLV_x = np.hstack((CLV_a , CLV_x))

CS_a = np.ones((CS_y.shape[0],1))
CS_x = np.hstack((CS_a , CS_x))
#print(x.shape)

def compute_cost(x, y, w, n):
    # Remember w is a vector here.
    cost=(1/(2*n)) * np.dot(np.ones(n).T, (np.dot(x,w)-y)**2)
    return cost

def gradient_descent(x , y , learning_rate , w , n , num_iters):
    # In place of None, write the updated value of w in temp
    for i in range(num_iters):
        # derivative vector is given by : X_train.Transpose *  (( X_train * w_vector)- y )
        temp =  w - (learning_rate/n)*np.dot(x.T,(np.dot(x,w)-y))
        w = temp
        
        if(i%100==0):
            # In place of None, call the cost you just coded above
            cost= compute_cost(x, y, w, n)
            print("Cost")
            print(cost)
    return w

#w_testcase=np.zeros((14,1))
#g=gradient_descent(x , y , 0.000049 , w_testcase , n , 100000)
#print(g[0])

def multiple_linear_reg_model_gda(x , y , n , learning_rate , num_iters):
    #initialize the values of parameter vector w. It should be a column vector of zeros of dimension(n,1)
    w = np.zeros((11,1))
    
    #calculate the initial cost by calling the function you coded above.
    initial_cost= compute_cost(x, y , w, n)
    print("Initial Cost")
    print(initial_cost)
    
    #calculate the optimized value of gradients by calling the gradient_descent function coded above
    
    w = gradient_descent(x , y , learning_rate , w , n , num_iters)
    
    #Calculate the cost with the optimized value of w by calling the cost function.
    
    final_cost = compute_cost(x, y , w, n)
    print("Final Cost")
    print(final_cost)
    return w


learning_rate = 0.00055
num_iters = 20000


#MS_w = multiple_linear_reg_model_gda(MS_x , MS_y , MS_n , learning_rate , num_iters)

#RC_w = multiple_linear_reg_model_gda(RC_x , RC_y , RC_n , learning_rate , num_iters)

#CLV_w = multiple_linear_reg_model_gda(CLV_x , CLV_y , CLV_n , learning_rate , num_iters)

CS_w = multiple_linear_reg_model_gda(CS_x , CS_y , CS_n , learning_rate , num_iters)
# The value of final cost should be 14.3470049896 or nearly this(depending on the values of learning_rate and num_itersations you choose.)

#print(w)

#x = [31,	5.16,	36.4,	12,	5,	0,	10,	20,	12,	27,	28]
def predict(x,w):
    predicted = np.dot(w.T, x.T)
    return predicted

print(predict(CS_x,CS_w))




