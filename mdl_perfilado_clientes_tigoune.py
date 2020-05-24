# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:47:44 2020

@author: Juan Carlos Agudelo
"""
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import csv

## Lectura de datos     
base_sur = pd.read_csv(r'C:\Users\pc\Downloads\BASE_SUR.txt', sep=r'¡', engine='python')
base_sur.head()


## Elimine variables: 
base_sur2 = base_sur[['CUCONUSE', 'SESUSERV', 'SERVDESC', 'SESUSUSC', 'SESUFEIN',
        'DEPADESC', 'SESUCUSA', 'SESUSAPE','SESUCICL', 'SUSCNITC', 
        #'VECTOR','SESUCATE', 'SESUSUCA','SESULOCA','SESUSAAN',
        #'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
        #'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
        'TIPIFICACION_CLIENTE', 'SUMA','CANTIDAD', 
       'CALIFICACION_VECTOR_SERVICIO',
       #'VECTOR_CUALITATIVO_CONTRATO', 
       #'BANCO_1', 'DIA_PAGO_1', 'DIA_PAGO_2',
       #'DIA_PAGO_3', 'FECHA_SUSPENSION', 
       'SEGMENTACION', 'REGIONAL','ESTRATO_AGRUPADO', 
       #'RANGO_EDAD', 'GENERO', 
       'DEPARTAMENTO_AGR','ANTIGUEDAD_DIAS', 
       #'PROM_SUSC', 'CANAL_ENVIO'
       'CANAL_PAGO']]

base_sur3 = base_sur2.dropna()

## Group by por nit para calificacion
b4 = base_sur3[['SUSCNITC','CALIFICACION_VECTOR_SERVICIO']]
group_nit = b4.groupby(['SUSCNITC'])['CALIFICACION_VECTOR_SERVICIO'].mean().reset_index()


## Join de la nueva calificacion
base_sur4 = pd.merge(base_sur3, group_nit, how='left', on=['SUSCNITC'])

## Prueba promedio
b5 = base_sur4[['SUSCNITC','CALIFICACION_VECTOR_SERVICIO_x','CALIFICACION_VECTOR_SERVICIO_y']]

### Creacion de la etiqueta de clasificacion
base_sur4['y'] = pd.cut(x=base_sur4['CALIFICACION_VECTOR_SERVICIO_y'],
                            bins=[-1,50,76,100],
                            labels=['No pago','Pago inoportuno','Pago'])





# X -> features, y -> label 
base_sur.columns[[30,31,36,47,   15,16,17,18,19,20,21,22,23,24,25,26]]
X = base_sur.drop(base_sur.columns[[30,31,36,47]], axis=1)
y = base_sur['y']
  
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 
  
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions) 














