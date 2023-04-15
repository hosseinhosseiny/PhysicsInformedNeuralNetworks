#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 20:38:35 2021

@author: hosseinhosseiny
"""
import numpy as np
import pandas as pd
from sciann import Variable, Functional, SciModel
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sciann as sn


data = pd.read_excel('results_Q1211.xlsx')
#data = data_[data_['Depth'] != 0 ]
data_ = data

ind = data_['Depth'] == 0
data_['WaterSurfaceElevation'][ind]=0

data_['h_categ'] = pd.cut(data_["Elevation"],
                          bins = [0, 4, 8, 12, 16, np.inf],
                          labels = [2, 6, 10, 14, 18 ])
#-------------- scaling the variables
max_x = max(data_['X'])
min_x = min(data_['X'])
max_y = max(data_['Y'])
min_y = min(data_['Y'])
max_z = max(data_['Elevation'])
min_z = min(data_['Elevation'])
max_u = max(data_['VelocityX'])
min_u = min(data_['VelocityX'])
max_v = max(data_['VelocityY'])
min_v = min(data_['VelocityY'])
max_h = max(data_['Depth'])
min_h = min(data_['Depth'])
max_E = max(data_['WaterSurfaceElevation'])
min_E = min(data_['WaterSurfaceElevation'])
max_tawx = max( data_['ShearStressX'])
min_tawx = min( data_['ShearStressX'])
max_tawy = max(data_['ShearStressY'])
min_tawy = min(data_['ShearStressY'])

data_['x_sc'] = (data_['X'] - min_x) / (max_x - min_x)
data_['y_sc'] = (data_['Y'] - min_y) / (max_y - min_y)
data_['z_sc'] = (data_['Elevation'] - min_z) / (max_z - min_z)
data_['u_sc'] = (data_['VelocityX'] - min_u) / (max_u - min_u)
data_['v_sc'] = (data_['VelocityY'] - min_v) / (max_v - min_v)
data_['h_sc'] = (data_['Depth'] - min_h) / (max_h - min_h)
data_['E_sc'] = (data_['WaterSurfaceElevation'] - min_E) / (max_E - min_E)
data_['tawx_sc'] = (data_['ShearStressX'] - min_tawx) / (max_tawx - min_tawx)
data_['tawy_sc'] = (data_['ShearStressY'] - min_tawy) / (max_tawy - min_tawy)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit (n_splits = 1, test_size = 0.2, random_state = 5)

for train_index, test_index in split.split (data_,data_['h_categ'] ):

  strat_train_set = data_.iloc[train_index]
  strat_test_set = data_.iloc[test_index]

data_tr = data_.loc[train_index]
data_test = data_.loc[test_index]

x_tr_sc = data_tr['x_sc']
y_tr_sc = data_tr['y_sc']
u_tr_sc = data_tr['u_sc']
v_tr_sc = data_tr['v_sc']
h_tr_sc = data_tr['h_sc']
E_tr_sc = data_tr['E_sc']
tawx_tr_sc = data_tr['tawx_sc']
tawy_tr_sc = data_tr['tawy_sc']

x_test_sc = data_test['x_sc']
y_test_sc = data_test['y_sc']
u_test_sc = data_test['u_sc']
v_test_sc = data_test['v_sc']
h_test_sc = data_test['h_sc']
E_test_sc = data_test['E_sc']
tawx_test_sc = data_test['tawx_sc']
tawy_test_sc = data_test['tawy_sc']
#-------------PINN/sciann implementation

#--- define variables
x = sn.Variable('x', dtype='float64')
y = sn.Variable('y', dtype='float64')
#
u = sn.Functional('u', [x,y], 5*[150], "relu")
v = sn.Functional('v', [x,y], 5*[150], "relu")
h = sn.Functional('h', [x,y], 5*[150], "relu")
E = sn.Functional('E', [x,y], 5*[150], "relu")
tawx = sn.Functional('tawx', [x,y], 5*[150], "relu")
tawy = sn.Functional('tawy', [x,y], 5*[150], "relu")

#------ doing the differential
uh_x, vh_y = sn.math.diff((u*h), x) , sn.math.diff((v*h), y)
u2h_x, v2h_y = sn.math.diff((u**2*h), x) , sn.math.diff((v**2*h), y)
E_x, E_y = sn.math.diff(E, x) , sn.math.diff(E, y)
uvh_x , uvh_y = sn.math.diff((u*v*h), x) , sn.math.diff((u*v*h), y)
tawx_x, tawy_y = sn.math.diff(tawx, x) , sn.math.diff(tawy, y)

#----define the equations/constrains
L1 = uh_x + vh_y
L2 = u2h_x + uvh_y + (9.81 * h) * (E_x) - (0.001 * tawx_x) # ro was assumed 1000 kg/m3
L3 = uvh_x + v2h_y + (9.81 * h) * (E_y) - (0.001 * tawy_y)
L4 = u
L5 = v
L6 = h
L7 = E
L8 = tawx
L9 = tawy


#----- building the sciann optimization model and tarining it

m = sn.SciModel([x,y], [L1, L2, L3, L4,
                        L5, L6, L7, L8, L9], "mse", "Adadelta")


history = m.train([x_tr_sc.values.flatten(), y_tr_sc.values.flatten()],
        [0, 0, 0,u_tr_sc.values.flatten(),
         v_tr_sc.values.flatten(), h_tr_sc.values.flatten(),
         E_tr_sc.values.flatten(), tawx_tr_sc.values.flatten(),
         tawy_tr_sc.values.flatten() ],
        batch_size=2000,
        epochs=100,
        shuffle=True,
        learning_rate=([0, 100, 500, 1000, 10000], 
                       [0.005, 0.001, 0.0005, 0.0001, 0.00001]),
        #validation_data = [x_val_sc,y_val_sc,x_val_sc,y_val_sc,x_val_sc,y_val_sc,x_val_sc],
        reduce_lr_after=100,
        stop_loss_value=1e-8,
        verbose=1)
m.save_weights('trained-rans_tanh.hdf5')

plt.semilogy(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('Flood_PIN_tanh.eps', dpi = 300, bbox_inches = "tight")


aaa = x_test_sc.values.flatten()
bbb = y_test_sc.values.flatten()
ccc = m.predict([aaa, bbb])
ccc_u = u.eval(m,[aaa, bbb])
h_pred_sc = ccc[5]
h_pred = h_pred_sc *(max_h - min_h) + min_h

h_dif = h_pred - np.array(data_test['Depth'])

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(np.array(data_test['X']), np.array(data_test['Y']),  np.array(h_pred),
             c=np.array(h_pred), s=5,
             cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')


















