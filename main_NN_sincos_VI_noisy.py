# -*- coding: utf-8 -*-
"""
Use neural networks for state estimation.

Input of the model: hour and hour data in cyclical encoding (cos,sin); weekday/weekend, holiday; voltage and current measurement of a DPMU.

Output of the model: the voltage magnitude of the feeder's smart meters.

Both PMU and smart meter data contains noise. But when evaluating the testing performance, we should use noiseless smart meter data for reference.

Need to check the following before running:
    (1) load table path.
    (2) temp h5 file path and name.

@author: admin
"""



# Load libraries
# define the seed first
import numpy as np
seed=1 # seed that determines how the dataset will be randomly split in to training, validation, and testing datasets.
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed) # now tensoflow 2.0 use this function to set random seed

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


# define evlauation function of MAPE
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

# Set some paths and values
table_path='DataSet_folder/' # this is the path of the folder that contains the needed csv data files.
temp_file_path='temp_best_model.h5' # this is the path and file name of the saved temporary best model file during training.
save_MAPE_csv_table_path='MAPE_summary.csv'
epoch_num=2 # maximum number of epochs in the model training.
batch_size=10000 # number of samples in a mini-batch
patience=10 # early stopping patience, in terms of epochs.


# load DPMU locations
dataset=pd.read_csv(table_path+'uPMU_V_list.csv') # load the list of DPMU voltage measurement nodes.
uPMU_V_list = dataset.values
dataset=pd.read_csv(table_path+'Branch_list.csv') # load the list of DPMU current measurement branches.
Branch_list = dataset.values

# build a loop table of different DPMU locations (i.e., the pairs of a DPMU voltage measurement node and a current measurement branch, so that the node is connected to the branch)
branch_node_table=np.empty([Branch_list.shape[0]*2, 3])
load_IV_list=np.empty([Branch_list.shape[0]*2, 2]) # the index number inside it starts from 1
for idx in range(0,Branch_list.shape[0]):
    temp_idx1=idx*2
    temp_idx2=temp_idx1+1
    temp_idx3=np.where(uPMU_V_list[:,0]==Branch_list[idx,0])
    temp_idx4=np.where(uPMU_V_list[:,0]==Branch_list[idx,1])
    temp_idx3=int(temp_idx3[0]) # tuple to int
    temp_idx4=int(temp_idx4[0])
    load_IV_list[temp_idx1,0]=idx+1
    load_IV_list[temp_idx2,0]=idx+1
    load_IV_list[temp_idx1,1]=temp_idx3+1
    load_IV_list[temp_idx2,1]=temp_idx4+1
    
    branch_node_table[temp_idx1,0:2]=Branch_list[idx,:]
    branch_node_table[temp_idx2,0:2]=Branch_list[idx,:]
    branch_node_table[temp_idx1,2]=Branch_list[idx,0]
    branch_node_table[temp_idx2,2]=Branch_list[idx,1]


load_IV_list=load_IV_list.astype(int) # the loop table of different DPMU locations. It stores the indices of the nodes and branches.


for location_idx in range(0,load_IV_list.shape[0]):
    print('location_idx: %.d' % (location_idx))	# show progress of testing different DPMU locations.
    
    # load dataset-----------------------------------------------------
    #load uPMU |V|
    dataset = pd.read_csv(table_path+'uPMU_V_mag.csv') # load DPMU voltage magnitude data
    array = dataset.values
    PMU_V_select=load_IV_list[location_idx,1] 
    tmp_idx1=(PMU_V_select-1)*3
    tmp_idx2=tmp_idx1+3
    X_V_mag = array[:,tmp_idx1:tmp_idx2]
    #load uPMU rad
    dataset = pd.read_csv(table_path+'uPMU_V_rad.csv') # load DPMU voltage angle data
    array = dataset.values
    X_V_rad = array[:,tmp_idx1:tmp_idx2]
    
    # load uPMU current
    dataset = pd.read_csv(table_path+'uPMU_I_mag.csv') # load DPMU current magnitude data
    array = dataset.values
    PMU_I_select=load_IV_list[location_idx,0] 
    tmp_idx1=(PMU_I_select-1)*3
    tmp_idx2=tmp_idx1+3
    X_I_mag = array[:,tmp_idx1:tmp_idx2]
    #load uPMU rad
    dataset = pd.read_csv(table_path+'uPMU_I_rad.csv') # load DPMU current angle data
    array = dataset.values
    X_I_rad = array[:,tmp_idx1:tmp_idx2]
    
    
    #load weekend, holiday, hour, and month.
    dataset = pd.read_csv(table_path+'date_indicator.csv')
    array = dataset.values
    temp_v=array[:,1] # month
    temp_cos=np.cos(2*np.pi*temp_v/12.0) # cyclical encoding for cyclical variables
    temp_sin=np.sin(2*np.pi*temp_v/12.0)
    X_month = np.concatenate((temp_cos[:,np.newaxis], temp_sin[:,np.newaxis]),axis=1) 
    
    temp_v=array[:,3] # hour
    temp_v=np.ceil(temp_v/2)
    temp_cos=np.cos(2*np.pi*temp_v/24.0) # cyclical encoding for cyclical variables
    temp_sin=np.sin(2*np.pi*temp_v/24.0)
    X_hour = np.concatenate((temp_cos[:,np.newaxis], temp_sin[:,np.newaxis]),axis=1) 
    
    X_week_holiday=array[:,4:6] # weekend indicator and holiday indicator.
    
    #load target values, i.e., the voltage magnitude of smart meters.
    dataset = pd.read_csv(table_path+'smart_meter_volt.csv')
    smart_meter_list=dataset.columns.to_list()
    array = dataset.values
    Y = array[:,0:]
    _,num_meter=np.shape(Y)
    
    #load target values, i.e., the voltage magnitude of smart meters without noise.
    dataset = pd.read_csv(table_path+'smart_meter_volt_noiseless.csv')
    array = dataset.values
    Y_noiseless = array[:,0:]
    
    X_PMU=np.concatenate((X_V_mag, X_V_rad,X_I_mag, X_I_rad),axis=1) 
    X_PMU=preprocessing.scale(X_PMU) #standarize
    X=np.concatenate((X_PMU, X_week_holiday, X_month, X_hour),axis=1) # complete input set
    
    validation_size = 0.20 # the ratio of data to use for validation.
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) # split data into train and test
    X_train2, X_test2, Y_train_noiseless, Y_test_noiseless = model_selection.train_test_split(X, Y_noiseless, test_size=validation_size, random_state=seed)# split noiseless data for test.
    X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(X_train, Y_train, test_size=validation_size, random_state=seed) # split data for training and validation.
    
    
    # create model (4 layer)-----------------------------------
    model = Sequential()
    model.add(Dense(200, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu')) # one layer, 200 nodes, initialized with uniform distribution, activation function is 'relu'.
    model.add(Dense(200, kernel_initializer='uniform', activation='relu')) # another layer of 200 nodes.
    model.add(Dense(200, kernel_initializer='uniform', activation='relu')) # third layer.
    model.add(Dense(Y_train.shape[1], kernel_initializer='uniform', activation='linear')) # output layer, activation function is 'linear'.
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error']) # setup training algorithm as 'adam', loss function is mean squared error.
    
    # simple early stopping setup
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
#    temp_file_path='temp_save/best_model_sincos_VI_noisy.h5'
    mc = ModelCheckpoint(temp_file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # Fit the model and obtain the training result.
    history=model.fit(X_train, Y_train, epochs=epoch_num, batch_size=batch_size, validation_data=(X_valid,Y_valid),callbacks=[es,mc])
    
    # evaluate the model, showing the MSE at the end of training
    _, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    _, valid_acc = model.evaluate(X_valid, Y_valid, verbose=0)
    _, test_acc = model.evaluate(X_test, Y_test_noiseless, verbose=0)
    print('Train: %.10f, Valid: %.10f Test: %.10f' % (train_acc, valid_acc, test_acc))	
    
    saved_model = load_model(temp_file_path) # load the temporarily saved model that has the lowest validation error, show the corresponding MSE.
    _, train_acc = saved_model.evaluate(X_train, Y_train, verbose=0)
    _, valid_acc = saved_model.evaluate(X_valid, Y_valid, verbose=0)
    _, test_acc = saved_model.evaluate(X_test, Y_test_noiseless, verbose=0)
    print('Train: %.10f, Valid: %.10f Test: %.10f' % (train_acc, valid_acc, test_acc))	
    print('Best test MSE: %.10e' % test_acc)
    
    best_predict=saved_model.predict(X_test)
    
    # plot training history, can be commented out if figures are not needed.
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.show()
    plt.grid(linestyle='dotted')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss history')
    plt.yscale('log')
    
    
    
    # calcualte R2 score
    tmp_1=((best_predict-Y_test_noiseless)**2).mean(axis=0)
    tmp_2=np.var(Y_test_noiseless,axis=0)
    R2_vec=1-np.divide(tmp_1,tmp_2)
    R2_vec=np.expand_dims(R2_vec,0) #insert a dimension to make the (10,) vector to (1,10) matrix
    
    # calcualte MAPE
    temp_MAPE=mape(np.ravel(Y_test_noiseless), np.ravel(best_predict))
    temp_MAE=mean_absolute_error(np.ravel(Y_test_noiseless), np.ravel(best_predict))
    print('Overall MAPE: %.10f, Overall MAE: %.10f' % (temp_MAPE, temp_MAE))	
    
    
    MAPE_by_meter=np.empty([num_meter])
    MAE_by_meter=np.empty([num_meter])
    for m in range(num_meter):
        MAPE_by_meter[m]=mape(Y_test_noiseless[:,m],best_predict[:,m])
        MAE_by_meter[m]=mean_absolute_error(Y_test_noiseless[:,m],best_predict[:,m])
    
    MAPE_by_meter=MAPE_by_meter[np.newaxis,:]
    MAE_by_meter=MAE_by_meter[np.newaxis,:]
    
    # a temp result array for easier copy and paste.
    temp_result=np.concatenate((R2_vec, MAPE_by_meter, MAE_by_meter),axis=0) # complete input set
    temp_v=np.empty([3, 1])
    temp_v[0,0]=test_acc
    temp_v[1,0]=temp_MAPE
    temp_v[2,0]=temp_MAE
    temp_result=np.concatenate((temp_result,temp_v),axis=1)
    
    if location_idx==0:
        R2_vec_table=temp_result[0,:][np.newaxis,:] # table containing each meter's R2 score and average R2 score, and number of epochs
        MAPE_by_meter_table=temp_result[1,:][np.newaxis,:] # table containing each meter's MAPE and average MAPE
        MAE_by_meter_table=temp_result[2,:][np.newaxis,:] # table containing each meter's MAE and average MAE
        Num_epoch_vec=np.empty([1])
        Num_epoch_vec[0]=es.stopped_epoch+1 # the stopped_epoch is the last epoch minus 1 (I dont' know why)
    else:
        R2_vec_table=np.concatenate((R2_vec_table, temp_result[0,:][np.newaxis,:]),axis=0)
        MAPE_by_meter_table=np.concatenate((MAPE_by_meter_table, temp_result[1,:][np.newaxis,:]),axis=0)
        MAE_by_meter_table=np.concatenate((MAE_by_meter_table, temp_result[2,:][np.newaxis,:]),axis=0)
        Num_epoch_vec=np.append(Num_epoch_vec,es.stopped_epoch+1)
        
R2_vec_table=np.concatenate((R2_vec_table, Num_epoch_vec[:,np.newaxis]),axis=1) # add epoch number to this table


# save the MAPE result table to a csv vile
save_table_MAPE=np.concatenate((branch_node_table, MAE_by_meter_table),axis=1)
temp_list_1=['DPMU I branch node 1', 'DPMU I branch node 2', 'DPMU V node']
temp_list_2=['Overall MAPE (%)']
temp_header=temp_list_1+smart_meter_list+temp_list_2
df = pd.DataFrame(save_table_MAPE, columns=temp_header)
df.to_csv(save_MAPE_csv_table_path, header=True, index=False)