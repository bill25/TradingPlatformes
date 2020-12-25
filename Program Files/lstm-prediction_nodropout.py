import sys
import subprocess as sp, platform as pf

def cls():
    if pf.system() == 'Windows':
        sp.call('cls', shell=True)
    else:
        sp.call('clear', shell=True)

def CheckInternet():
    try:
        urq.urlopen('https://www.google.com')
        return True
    except:
        try:
            urq.urlopen('https://www.baidu.com/')
        except:
            return False

if CheckInternet():
    print ('Installing and Updating Modules\n')
    sp.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'matplotlib', 'pandas', 'guizero', 'sklearn', 'numpy', 'scipy', 'keras', 'tqdm', 'wxPython', '--user'])
    cls()
    print ('All Modules Updated')
print ('Importing Modules')

import time, datetime, random, wx, glob, gc, os
import numpy as np, math
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from scipy.stats import t
from PIL import Image
from guizero import App, PushButton, Text, Picture, Box
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from keras import Sequential, backend
from keras.layers import LSTM, Dense
from keras.callbacks import Callback, History, TerminateOnNaN, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils.vis_utils import model_to_dot
from keras.utils import  plot_model
from keras_tqdm import TQDMCallback

#For all Activation, view https://keras.io/activation/
#For all Optimizers, view https://keras.io/optimizers/
#For all Loss f(x)s, view https://keras.io/losses/
dataFile, scaler, _ = 'GBP_HKD Historical Data.csv', MinMaxScaler(), wx.App(False)
epochs, step_slice, validation_proportion, propagation_batch_size = 500, 50, 0.2, 64
neurons_per_layer = [60,120,150]
activation, optimizer, loss = 'relu', 'adam', 'mean_squared_error'
load_weights_point, checkPrediction = '----.hdf5', False
moving_average, ema_scaling_factor = False, 650
data = pd.read_csv(dataFile, date_parser = True)
neural_network_layers, run_time, screen_res = len(neurons_per_layer), str(datetime.datetime.today()), list(wx.GetDisplaySize())
print ('Import Complete\n')

class TimingCallback(Callback):
    def __init__(self):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.start=time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time.time()-self.start)

def DeScale(array, scaler=scaler, scaling_method=str(scaler).split('(')[0]):
    if scaling_method == 'MinMaxScaler':
        return ((array/scaler.scale_[0]) + scaler.data_min_[0]).tolist()
    elif scaling_method == 'StandardScaler':
        return ((array*(scaler.var_[0])**0.5) + scaler.mean_[0]).tolist()
    elif scaling_method == 'RobustScaler':
        return ((array/scaler.scale_[0]) + scaler.center_[0]).tolist()

def RMSE(observed, predicted):
    if len(observed) == len(predicted):
        return math.sqrt(sum([(predicted[i]-observed[i])**2 for i in range(len(predicted))])/len(predicted))

def FindModel(array_x, array_y, array_xnl, array_ynl, typ):
    constant_n = len(array_x)
    array_x2 = [math.pow(x,2) for x in array_x]
    array_xy = [array_x[i] * array_y[i] for i in range(constant_n)]
    array_y2 = [math.pow(y,2) for y in array_y]
    det = 1 / (constant_n*sum(array_x2) - math.pow(sum(array_x),2))
    beta0 = sum(array_x2) * sum(array_y) - sum(array_x) * sum(array_xy)
    beta1 = constant_n * sum(array_xy) - sum(array_x) * sum(array_y)
    r = ((constant_n*sum(array_xy) - sum(array_x)*sum(array_y))/math.sqrt((constant_n*sum(array_x2) - sum(array_x)**2)*(constant_n*sum(array_y2) - sum(array_y)**2)))

    if r > 1:
        r = 1
    elif r < -1:
        r = -1

    if typ == 'l':
        x_coeff = beta1*det
        y_intercept = beta0*det
    elif typ == 'p':
        x_coeff = beta1*det
        y_intercept = math.e ** (beta0*det)
    elif typ == 'e':
        x_coeff = math.e ** (beta1*det)
        y_intercept = math.e ** (beta0*det)
    return [x_coeff, y_intercept, r**2, r]

def ReturnLogValue(number):
    try:
        return math.log(number, math.e)
    except ValueError:
        return number

def closeGraphs():
    plt.close('all')

def displayPrediction(DY_realData, DY_prediction, DY_train, DY_train_initial, moving_average=moving_average, currency=dataFile):
    closeGraphs()
    DY_train_initial.append(DY_realData[0])
    title = currency.replace(' Historical Data.csv', '').replace('_', ' to ')
    plt.figure(1)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.plot([len(DY_train_initial)+i for i in range(len(DY_realData))], DY_realData, color = 'red', label = 'Real Forex Price')
    plt.plot([len(DY_train_initial)+i for i in range(len(DY_prediction))], DY_prediction, color = 'blue', label = 'Predicted Forex Price')
    plt.title(title+' Forex Price Prediction\nRMSE: '+str(rmse))
    plt.xlabel('Time')
    plt.ylabel(title+' Price')
    if moving_average:
        plt.plot([i for i in range(len(DY_train))], DY_train, color='black',label = 'Moving Average Training Data')
        plt.plot([i for i in range(len(DY_train_initial))], DY_train_initial, color='purple', label='Training Data')
    else:
        plt.plot([i for i in range(len(DY_train_initial))], DY_train_initial, color='black', label='Training Data')
    plt.legend()
    plt.show()

def displayLearning(loss, val_loss):
    closeGraphs()
    plt.figure(1)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.plot([i+1 for i in range(len(loss))], loss, color='black',label = 'Loss')
    plt.plot([i+1 for i in range(len(loss))], val_loss, color = 'red', label = 'Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def displayMultipleGraphs(history_loss, history_val, tlogs):
    closeGraphs()
    _, graphs_loc = plt.subplots(2,3)
    
    graphs_loc[0,0].plot([i+1 for i in range(len(history_loss))], history_loss, label='Train Loss')
    graphs_loc[1,0].plot([i+1 for i in range(len(history_val))], history_val, label='Test Loss')
    graphs_loc[0,1].plot(history_loss, tlogs, label='Time taken for each Epoch')
    graphs_loc[1,1].plot(history_val, tlogs, label='Train Loss')
    graphs_loc[0,2].plot([i+1 for i in range(len(tlogs))], tlogs, label='Test Loss')
    graphs_loc[1,2].plot(history_loss, history_val, label='Test Loss against Validation Loss')
    
    graphs_loc[0,0].set_title('Loss Values v Epoch')
    graphs_loc[1,0].set_title('Validation Loss Values v Epoch')
    graphs_loc[0,1].set_title('Time v Loss')
    graphs_loc[1,1].set_title('Time v Validation Loss')
    graphs_loc[0,2].set_title('Epoch v Time')
    graphs_loc[1,2].set_title('Validation Loss v Loss')

    graphs_loc[0,0].set(xlabel='Epoch', ylabel='Loss')
    graphs_loc[1,0].set(xlabel='Epoch', ylabel='Validation Loss')
    graphs_loc[0,1].set(xlabel='Loss', ylabel='Time')
    graphs_loc[1,1].set(xlabel='Validation Loss', ylabel='Time')
    graphs_loc[0,2].set(xlabel='Epoch', ylabel='Time')
    graphs_loc[1,2].set(xlabel='Loss', ylabel='Validation Loss')

    graphs_loc[0,0].grid(color='black', linestyle='-', linewidth=0.1)
    graphs_loc[1,0].grid(color='black', linestyle='-', linewidth=0.1)
    graphs_loc[0,1].grid(color='black', linestyle='-', linewidth=0.1)
    graphs_loc[1,1].grid(color='black', linestyle='-', linewidth=0.1)
    graphs_loc[0,2].grid(color='black', linestyle='-', linewidth=0.1)
    graphs_loc[1,2].grid(color='black', linestyle='-', linewidth=0.1)

    try:
        LvE = PlotBestFitLine([i+1 for i in range(len(history_loss))], history_loss, label='loss v Epoch Reg.Line',graphs_loc=graphs_loc, loc_x=0, loc_y=0)
        LvE_Textbox.value = LvE
    except OverflowError:
        LvE_Textbox.value = 'loss v Epoch Reg.Line\nInfo Cannot be displayed'

    try:
        VvE = PlotBestFitLine([i+1 for i in range(len(history_val))], history_val, label='val_loss v Epoch Reg.Line',graphs_loc=graphs_loc, loc_x=1, loc_y=0)
        VvE_Textbox.value = VvE
    except OverflowError:
        VvE_Textbox.value = 'val_loss v Epoch Reg.Line\nInfo Cannot be displayed'

    try:
        TvL = PlotBestFitLine(history_loss, tlogs, label='Time v loss Reg.Line', graphs_loc=graphs_loc, loc_x=0, loc_y=1)
        TvL_Textbox.value = TvL
    except OverflowError:
        TvL_Textbox.value = 'Time v loss Reg.Line\nInfo Cannot be displayed'

    try:
        TvV = PlotBestFitLine(history_val, tlogs, label='Time v val_loss Reg.Line', graphs_loc=graphs_loc, loc_x=1, loc_y=1)
        TvV_Textbox.value = TvV
    except OverflowError:
        TvV_Textbox.value = 'Time v val_loss Reg.Line\nInfo Cannot be displayed'

    try:
        EvT = PlotBestFitLine([i+1 for i in range(len(tlogs))], tlogs, label='Epoch v Time Reg.Line',graphs_loc=graphs_loc, loc_x=0, loc_y=2)
        EvT_Textbox.value = EvT
    except OverflowError:
        EvT_Textbox.value = 'Epoch v Time Reg.Line\nInfo Cannot be displayed'

    try:
        LvV = PlotBestFitLine(history_loss, history_val, label='val_loss v loss Reg.Line',graphs_loc=graphs_loc, loc_x=1, loc_y=2)
        LvV_Textbox.value = LvV
    except OverflowError:
        LvV_Textbox.value = 'val_loss v loss Reg.Line\nInfo Cannot be displayed'
    
    plt.show()

def displayonlyPred(Y_test, Y_pred, rmse, currency=dataFile):
    closeGraphs()
    title = currency.replace(' Historical Data.csv', '').replace('_', ' to ')
    plt.figure(1)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.plot(Y_test, color='black', label='Realistic Market Data')
    plt.plot(Y_pred, color='red', label='Predicted Market Data')
    plt.title(title+' Forex Price Prediction\nRMSE: '+str(rmse))
    plt.xlabel('Time')
    plt.ylabel(title+' Price')
    plt.legend()
    plt.show()

def PlotBestFitLine(array_x, array_y, label='', graphs_loc='', loc_x='', loc_y=''):
    landscape = np.linspace(min(array_x), max(array_x), 50000)
    array_xlog = [ReturnLogValue(i) for i in array_x]
    array_ylog = [ReturnLogValue(i) for i in array_y]
    linear_data_set = FindModel(array_x, array_y, array_x, array_y, 'l')
    exp_data_set = FindModel(array_x, array_ylog, array_x, array_y, 'e')
    powered_data_set = FindModel(array_xlog, array_ylog, array_x, array_y, 'p')
    logarithmic_data_set = FindModel(array_xlog, array_y, array_x, array_y, 'l')
    data_set = [linear_data_set, powered_data_set, exp_data_set, logarithmic_data_set]
    error_set = [float(data_set[0][2]), float(data_set[1][2]), float(data_set[2][2]), float(data_set[3][2])]
    pmcc = [float(data_set[0][3]), float(data_set[1][3]), float(data_set[2][3]), float(data_set[3][3])]
    
    if max(error_set) == error_set[0]:
        linear_y = data_set[0][0]*landscape + (data_set[0][1])
        if str(type(graphs_loc)) != "<class 'str'>":
            graphs_loc[loc_x, loc_y].plot(landscape, linear_y, linewidth = 2, label=label)
        else:
            plt.plot(landscape, linear_y, linewidth = 2, label=label)
        return (str(label)+'\nLinear\nPMCC: '+str(pmcc[0])+'\nCoefficient of Determination: '+str(error_set[0]))

    elif max(error_set) == error_set[1]:
        powered_y = data_set[1][1] * (landscape ** data_set[1][0])
        if str(type(graphs_loc)) != "<class 'str'>":
            graphs_loc[loc_x, loc_y].plot(landscape, powered_y, linewidth = 2, label=label)
        else:
            plt.plot(landscape, powered_y, linewidth = 2, label=label)
        return (str(label)+'\nPowered\nPMCC: '+str(pmcc[1])+'\nCoefficient of Determination: '+str(error_set[1]))
        
    elif max(error_set) == error_set[2]:
        exp_y = data_set[2][1] * (data_set[2][0] ** landscape)
        if str(type(graphs_loc)) != "<class 'str'>":
            graphs_loc[loc_x, loc_y].plot(landscape, exp_y, linewidth = 2, label=label)
        else:
            plt.plot(landscape, exp_y, linewidth = 2, label=label)
        return (str(label)+'\nExponential\nPMCC: '+str(pmcc[2])+'\nCoefficient of Determination: '+str(error_set[2]))
        
    elif max(error_set) == error_set[3]:
        y = [data_set[3][1] + data_set[3][0] * math.log(landscape[i], math.e) for i in range(len(landscape))]
        if str(type(graphs_loc)) != "<class 'str'>":
            graphs_loc[loc_x, loc_y].plot(landscape, y, linewidth = 2, label=label)
        else:
            plt.plot(landscape, y, linewidth = 2, label=label)
        return (str(label)+'\nLogarithmic\nPMCC: '+str(pmcc[3])+'\nCoefficient of Determination: '+str(error_set[3]))

def PlotGraph(x, y, label, title, xlabel, ylabel):
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.plot(x, y, label = label)
    PlotBestFitLine(x, y, label=label+' Reg.Line')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def disclaimer_mit():
    print ('DISCLAIMER')
    print (open('DISCLAIMER.txt','r').read())
    print ('\nMIT LICENSE')
    print (open('LICENSE.txt','r').read())

def createLog(run_time, restore, scaler, neural_network_layers,test_data_len, step_slice, validation_proportion,
              neurons_per_layer, activation, optimizer, loss, propagation_batch_size,
              tlogs, model_init_time, X_train, training_time, predict_time, rmse, moving_average):
    try:
        doubleClickOpen(str('Model_Summary '+str(str(run_time).replace(':','.'))+'.txt'))
    except:
        with open(str('Model_Summary '+str(str(run_time).replace(':','.'))+'.txt'), 'a+') as summary_f:
            summary_f.write('Backend Framework: '+str(backend.backend()).title()+'\n'+
                            'Inherited: '+str(restore)+'\n'+
                            'Scaling: '+str(scaler).split('(')[0]+'\n'+
                            'Run Time: '+str(datetime.datetime.today())+'\n'+
                            'Constants Configuration\n'+
                            'Total LSTM Layers: '+str(neural_network_layers)+'\n'+
                            'Number of Test Data: '+str(test_data_len)+'\n'+
                            'Step Slice: '+str(step_slice)+'\n'+
                            'Test Proportion while Running: '+str(validation_proportion)+'\n'+
                            'LSTM Layer Configuration: '+str(neurons_per_layer)+'\n'+
                            'Activation Function: '+str(activation)+'\n'+
                            'Optimizer: '+str(optimizer)+'\n'+
                            'Loss Function: '+str(loss)+'\n'+
                            'Batch Size: '+str(propagation_batch_size)+'\n'+
                            'Epochs: '+str(len(tlogs))+'\n'+
                            'Moving Average: '+str(moving_average)+'\n')
            LSTM_model.summary(print_fn=lambda x: summary_f.write(x + '\n'))
            summary_f.write('\nTiming Statistics\n'+
                            'Model Initilization Time: '+str(model_init_time)+'\n'+
                            'Training Samples: '+str(len(X_train))+'\n'+
                            'Training Time: '+str(training_time)+'\n'+
                            'Prediction Time: '+str(predict_time)+'\n'+
                            'RMSE: '+str(rmse))
            summary_f.close()
        doubleClickOpen(str('Model_Summary '+str(str(run_time).replace(':','.'))+'.txt'))
        
def writeRMSERecords(rmse, tlogs, step_slice=step_slice):
    try:
        doubleClickOpen('RMSE_Records.csv')
    except:
        with open('RMSE_Records.csv', 'a+') as records:
            records.write(str(step_slice)+','+str(rmse)+','+str(len(tlogs))+'\n')
        records.close()
        doubleClickOpen('RMSE_Records.csv')
    
def passFunction():
    pass

def createButton(box, def_text_size, def_colour, text='', command=passFunction, args=[], grid=[]):
    width = 15
    height = 1
    b1 = PushButton(box, text=text, command=command, args=args, grid=grid, width=width, height=height, align='left')
    b1.text_size = def_text_size
    b1.text_color = def_colour

def doubleClickOpen(file_name):
    if pf.system() == 'Darwin':
        sp.call(('open', file_name))
    elif pf.system() == 'Windows':
        os.startfile(str(file_name))
    else:
        sp.call(('xdg-open', file_name))

def removefiletype(file_type):
    for i in [i for i in glob.glob('*.'+file_type) if i not in glob.glob('*Historical Data*.csv') + ['DISCLAIMER.txt','LICENSE.txt']]:
        os.remove(i)

def FindSmoothingFactor(stp):
    return 2/(stp+1)

def EMA(ema_y, val_t, stp):
    return val_t * FindSmoothingFactor(stp) + ema_y * (1 - FindSmoothingFactor(stp))

def DEMA(ema_y, val_t, stp):
    return 2*EMA(ema_y, val_t,stp) - EMA(ema_y,EMA(ema_y, val_t, stp),stp)

def openDirectory():
    os.startfile(os.getcwd())

def calculateRc(n, p=0.05):
    if n == 2:
        return 1
    else:
        df, p = n-2, 1-p
        tc = t.ppf(p, df)
        return str((((tc**2)/df)/(((tc**2)/df)+1))**0.5)

def randomTest(model, Xtrain, Ytrain, valLoss, vp=validation_proportion):
    closeGraphs()
    index = random.randint(0,int((len(Xtrain)-1)*(1-vp)))
    train_pred = model.predict(np.array([Xtrain[index]]))[0]
    plt.plot(train_pred, color='red', label='Prediction')
    plt.plot(Ytrain[index], color='black', label='Actual Data')
    plt.xlabel('From Day 1 of the Random Set of Data')
    plt.ylabel('The prediction unscaled')
    plt.title('An Analysis of the Understanding of the data\nRMSE: '+str(RMSE(train_pred, Ytrain[index]))+'\nPrevious Epoch Loss (RMSE): '+str(math.sqrt(valLoss[-1])))
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.legend()
    plt.show()

gc.enable()
test_data_len = len(data)-step_slice*2
Rdata_test, data_train = data[test_data_len::].copy(), data[:test_data_len:].copy()
data_training = scaler.fit_transform(data_train.drop(['Date','Change %'], axis=1))
data_test = scaler.transform(Rdata_test.drop(['Date','Change %'], axis=1))
if moving_average:
    price = [data_training[i][0] for i in range(len(data_training))]
    open_price = [data_training[i][1] for i in range(len(data_training))]
    high_price = [data_training[i][2] for i in range(len(data_training))]
    low_price = [data_training[i][3] for i in range(len(data_training))]
    Y_train_initial = DeScale(price)
    price_ema = []
    open_price_ema = []
    high_price_ema = []
    low_price_ema = []
    
    for i in range(len(data_training)):
        if i == 0:
            price_ema.append(DEMA(price[i], price[i], ema_scaling_factor))
            open_price_ema.append(DEMA(open_price[i], open_price[i], ema_scaling_factor))
            high_price_ema.append(DEMA(high_price[i], high_price[i], ema_scaling_factor))
            low_price_ema.append(DEMA(low_price[i], low_price[i], ema_scaling_factor))
        else:
            price_ema.append(DEMA(price_ema[-1], price[i], ema_scaling_factor))
            open_price_ema.append(DEMA(open_price_ema[-1], open_price[i], ema_scaling_factor))
            high_price_ema.append(DEMA(high_price_ema[-1], high_price[i], ema_scaling_factor))
            low_price_ema.append(DEMA(low_price_ema[-1], low_price[i], ema_scaling_factor))

    data_training = np.array([[price_ema[i], open_price_ema[i], high_price_ema[i], low_price_ema[i]] for i in range(len(data_training))])
else:
    Y_train_initial = []

X_train, Y_train, output_price, X_test, Y_test = [], [], [i[0] for i in data_training], [data_test[:step_slice:]], data_test[step_slice::, 0]
for i in range(step_slice, len(data_training)-step_slice+1):
    X_train.append(data_training[i-step_slice:i])
    Y_train.append(data_training[i:i+step_slice, 0])

X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

#Step III: Neural Network Implementation
start_time = time.time()
LSTM_model = Sequential()
LSTM_model.add(LSTM(
    units = neurons_per_layer[0],
    activation = activation,
    return_sequences = True,
    input_shape = (step_slice, 4)))
for i in range(1, neural_network_layers-1, 1):
    LSTM_model.add(LSTM(
        units = neurons_per_layer[i],
        activation = activation,
        return_sequences = True))
LSTM_model.add(LSTM(
    units = neurons_per_layer[-1],
    activation = activation))
LSTM_model.add(Dense(units=step_slice))
LSTM_model.compile(optimizer=optimizer, loss=loss)
cls()

try:
    plot_model(LSTM_model, to_file='LSTM_Architecture_NoDropout.jpg',show_shapes=True)
except AssertionError:
    print ('Cannot Export Neural Network Graph; DPI too high')

try:
    LSTM_model.load_weights(load_weights_point)
    print ('Inherited Weights\n')
    restore = True
except OSError:
    print ('Did not inherit weights from Previous Training\n')
    restore = False
except ValueError:
    decision = input('Wrong Data Fitting, Continue?(Y/n) ').lower()
    if decision != 'y':
        quit()
    restore = False
    print ('Did not inherit weights from Previous Training\n')

model_init_time = str(time.time()-start_time)
start_time = time.time()
Timings = TimingCallback()
LSTM_model.summary()
print ('\n')
if not checkPrediction:
    history = LSTM_model.fit(
        X_train,
        Y_train,
        epochs = epochs,
        batch_size = propagation_batch_size,
        validation_split=validation_proportion,
        verbose=0,
        callbacks=[Timings,
                History(),
                TerminateOnNaN(),
                ModelCheckpoint('{epoch:03d}-val_loss {val_loss:.5f}-loss {loss:.5f}.hdf5', monitor='val_loss', save_best_only=True, mode='min'),
                CSVLogger('LSTM-Prediction.csv',append=True),
                TQDMCallback()])
#EarlyStopping(monitor='loss', min_delta=5e-6, patience=15, restore_best_weights=False),
training_time = str(time.time()-start_time)
start_time = time.time()
Y_pred = LSTM_model.predict(X_test)[0]
predict_time = str(time.time()-start_time)
rmse = RMSE(Y_test, Y_pred)
Y_training, Y_realData, Y_prediction = DeScale(output_price), DeScale(Y_test), DeScale(Y_pred)

gc.collect()
def_text_size = 9
def_colour = 'white'
border_size = 10
artifical_padding = ' '*2
graphApp = App('LSTM Forex Prediction - Graphical Representations', height=screen_res[1], width=screen_res[0], layout='grid', bg='black')
title = Text(graphApp, text='LSTM Forex Prediction Statistical Data Representations', size=18, color='white', grid=[0,0], width='fill')
Text(graphApp, text=' ',size=def_text_size-5, grid=[0,1])
if len(Timings.logs) > 2:
    summary_list = []
    LSTM_model.summary(print_fn=lambda x: summary_list.append(artifical_padding+x))
    model_summary = "\n".join(summary_list)

    box_ol = Box(graphApp, layout='grid',grid=[0,2])
    box_l1 = Box(box_ol, layout='grid', grid=[0,1], border=border_size, align='top', height='fill')
    box_l2 = Box(box_ol, layout='grid', grid=[1,1], border=border_size, align='top', height='fill')
    box_l3 = Box(box_ol, layout='grid', grid=[2,1], border=border_size, align='top', height='fill')
    box_l4 = Box(box_ol, layout='grid', grid=[3,1], border=border_size, align='top', height='fill')
    box_l5 = Box(box_ol, layout='grid', grid=[4,1], border=border_size, align='top', height='fill')

    Text(box_l1, text='Model Configuration', size=def_text_size+3, grid=[0,0], color=def_colour)
    Picture(box_l1, image='LSTM_Architecture_NoDropout.jpg', width=int(screen_res[1]*0.85*Image.open('LSTM_Architecture.jpg').size[0]/Image.open('LSTM_Architecture.jpg').size[1]), height=int(screen_res[1]*0.85), grid=[0,1])
    Text(box_l2, text=artifical_padding+'Completion Statistics', size=def_text_size+3, grid=[0,0], color=def_colour)
    Text(box_l2, text='\n'+artifical_padding+'LSTM Model Configuration', size=def_text_size+3, grid=[0,7], color=def_colour)
    Text(box_l2, text=artifical_padding+'Backend Framework: '+str(backend.backend()).title(), size=def_text_size, grid=[0,1], color=def_colour, align='left')
    Text(box_l2, text=artifical_padding+'Model Initilization Time: '+str(model_init_time), size=def_text_size, grid=[0,2], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Training Samples: '+str(len(X_train)), size=def_text_size, grid=[0,3], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Training Time: '+str(training_time), size=def_text_size, grid=[0,4], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Prediction Time: '+str(predict_time), size=def_text_size, grid=[0,5], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'RMSE: '+str(rmse), size=def_text_size, grid=[0,6], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Scaling: '+str(scaler).split('(')[0], size=def_text_size, grid=[0,8], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Total LSTM Layers: '+str(neural_network_layers), size=def_text_size, grid=[0,9], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Number of Training Data: '+str(test_data_len), size=def_text_size, grid=[0,10], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Step Slice: '+str(step_slice), size=def_text_size, grid=[0,11], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Test Proportion while Running: '+str(validation_proportion), size=def_text_size, grid=[0,12], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'LSTM Layer Configuration: '+str(neurons_per_layer), size=def_text_size, grid=[0,13], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Activation Function: '+str(activation), size=def_text_size, grid=[0,15], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Optimizer: '+str(optimizer), size=def_text_size, grid=[0,16], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Loss Function: '+str(loss), size=def_text_size, grid=[0,17], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Batch Size: '+str(propagation_batch_size), size=def_text_size, grid=[0,18], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Epochs: '+str(len(Timings.logs)), size=def_text_size, grid=[0,19], align='left', color=def_colour)
    Text(box_l2, text=artifical_padding+'Moving Average: '+str(moving_average), size=def_text_size, grid=[0,20], align='left', color=def_colour)

    Text(box_l3, text=artifical_padding+'In Depth Model Summary', size=def_text_size+3, grid=[0,0], color=def_colour)
    Text(box_l3, text=model_summary, size=def_text_size, grid=[0,1], align='left', color=def_colour)

    Text(box_l4, text='Settings', grid=[0,0], color='white', size=def_text_size+3)
    createButton(box_l4, def_text_size, def_colour, text='Save Configurations', command=createLog, args=[run_time, restore, scaler, neural_network_layers,test_data_len, step_slice, validation_proportion, neurons_per_layer, activation, optimizer, loss, propagation_batch_size, Timings.logs, model_init_time, X_train, training_time, predict_time, rmse, moving_average], grid=[0,1])
    createButton(box_l4, def_text_size, def_colour, text='Save RMSE Values', command=writeRMSERecords, args=[rmse, Timings.logs], grid=[0,2])
    createButton(box_l4, def_text_size, def_colour, text='Remove all\nCheckpoints', command=removefiletype, args=['hdf5'], grid=[0,3])
    createButton(box_l4, def_text_size, def_colour, text='Remove all\nText File Logs', command=removefiletype, args=['txt'], grid=[0,4])
    createButton(box_l4, def_text_size, def_colour, text='Remove all\nCSV Logs', command=removefiletype, args=['csv'], grid=[0,5])
    createButton(box_l4, def_text_size, def_colour, text='Open Current\nWorking Directory', command=openDirectory, grid=[0,6])

    Text(box_l4, text='Graphs', grid=[1,0], color='white', size=def_text_size+3)
    createButton(box_l4, def_text_size, def_colour, text='Entire Dataset', command=displayPrediction, args=[Y_realData, Y_prediction, Y_train_initial, Y_training], grid=[1,1])
    createButton(box_l4, def_text_size, def_colour, text='Prediction Graph', command=displayonlyPred, args=[Y_realData, Y_prediction, rmse], grid=[1,2])
    createButton(box_l4, def_text_size, def_colour, text='Random Test', command=randomTest, args=[LSTM_model, X_train, Y_train, history.history['loss']], grid=[1,3])
    createButton(box_l4, def_text_size, def_colour, text='Loss & val_loss v Epoch', command=displayLearning, args=[history.history['loss'], history.history['val_loss']], grid=[1,4])
    createButton(box_l4, def_text_size, def_colour, text='All Graphs', command=displayMultipleGraphs, args=[history.history['loss'], history.history['val_loss'], Timings.logs], grid=[1,5])
    createButton(box_l4, def_text_size, def_colour, text='Loss v Epoch', command=PlotGraph, args=[[i+1 for i in range(len(history.history['loss']))], history.history['loss'], 'Train Loss', 'Loss Values while Propagation', 'Epoch', 'loss'], grid=[1,6])
    createButton(box_l4, def_text_size, def_colour, text='val_loss v Epoch', command=PlotGraph, args=[[i+1 for i in range(len(history.history['val_loss']))], history.history['val_loss'], 'Validation Test Loss', 'Validation loss Values while Propagation', 'Epoch', 'val_loss'], grid=[1,7])
    createButton(box_l4, def_text_size, def_colour, text='Epoch v Time', command=PlotGraph, args=[[i+1 for i in range(len(Timings.logs))], Timings.logs, 'Time taken for each Epoch', 'Epoch against Time', 'Epoch', 'Time'], grid=[1,8])
    createButton(box_l4, def_text_size, def_colour, text='val_loss v Time', command=PlotGraph, args=[history.history['val_loss'], Timings.logs, 'Time taken each val_loss', 'val_loss against Time', 'val_loss', 'Time'], grid=[1,9])
    createButton(box_l4, def_text_size, def_colour, text='Loss v Time', command=PlotGraph, args=[history.history['loss'], Timings.logs, 'Time taken each loss', 'Epoch against Time', 'Loss', 'Time'], grid=[1,10])
    createButton(box_l4, def_text_size, def_colour, text='Loss v val_loss', command=PlotGraph, args=[history.history['loss'], history.history['val_loss'], 'Loss and val_loss', 'val_loss against loss', 'Loss', 'val_loss'], grid=[1,11])

    Text(box_l5, size=def_text_size+3, grid=[0,0], color=def_colour, text='Statistical Data')
    Text(box_l5, size=def_text_size, grid=[0,1], color=def_colour, text='One-tail Test Critical Value: '+str(calculateRc(len(Timings.logs))))
    LvE_Textbox = Text(box_l5, size=def_text_size, grid=[0,2], color=def_colour)
    VvE_Textbox = Text(box_l5, size=def_text_size, grid=[0,3], color=def_colour)
    EvT_Textbox = Text(box_l5, size=def_text_size, grid=[0,4], color=def_colour)
    TvV_Textbox = Text(box_l5, size=def_text_size, grid=[0,5], color=def_colour)
    TvL_Textbox = Text(box_l5, size=def_text_size, grid=[0,6], color=def_colour)
    LvV_Textbox = Text(box_l5, size=def_text_size, grid=[0,7], color=def_colour)
else:
    Text(graphApp, text='Insufficient Data to Plot Regression Graphs Data', size=11, align='left', color='white', grid=[0,1])
    createButton(graphApp, def_text_size, def_colour, text='Prediction Graph', command=displayonlyPred, args=[Y_realData, Y_prediction, rmse], grid=[0,2])

cls()
disclaimer_mit()
graphApp.display()
gc.collect()
cls()
