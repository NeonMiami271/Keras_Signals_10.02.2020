#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Скрипт для подбора гиперпараметров(количество слоев, количество нейронов в слое
# количество эпох обучения, обучающей выборки и тд)

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
import numpy
from datetime import datetime
import time
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
import random

start_time = datetime.now()
numpy.random.seed(2)

# Загрузка данных обучения и данных проверки
dataset_train_x = numpy.loadtxt("/Dataset/Dataset/Dataset_Full_Full.csv", delimiter=",")
dataset_train_y = numpy.loadtxt("/Dataset/Answer/Answers_Full_Full.csv", delimiter=",")
x_train = dataset_train_x[:,0:1024]
y_train = dataset_train_y

dataset_test_x = numpy.loadtxt("/Dataset/Dataset/Dataset_Full_Test.csv", delimiter=",")
dataset_test_y = numpy.loadtxt("/Dataset/Answer/Answers_Full_Test.csv", delimiter=",")
x_test = dataset_test_x[:,0:1024]
y_test = dataset_test_y

# Страндартизация критериев
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

# Создание модели нейронной сети с возможностью варьирования гиперпараметров
def build_model(hp):
    model = keras.Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(layers.Dense(units=hp.Int('units_input_layer',
                                   min_value=512,
                                   max_value=2048,
                                   step=10),
                                   input_dim=1024,
                                   activation=activation_choice))

    model.add(layers.Dense(units=hp.Int('units_layer_1',
                                    min_value=6,
                                    max_value=512,
                                    step=2),
                                    activation=activation_choice))

    model.add(layers.Dense(units=hp.Int('units_layer_2',
                                    min_value=6,
                                    max_value=512,
                                    step=2),
                                    activation=activation_choice))

    model.add(layers.Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae'])
    return model

# Запуск функции подбора гиперпараметров
tuner = BayesianOptimization(build_model, objective='val_mae', max_trials=100 ,directory='directory_tuner')
tuner.search_space_summary()

# Обучение сети с учетом гиперпараметров (можно указать, чтобы подбор количества
# эпох и мини-выборки осуществлялся функцией подбора)
tuner.search(x_train,
             y_train,
             batch_size=2000,
             epochs=25,
             validation_split=0.2,
             )
tuner.results_summary()

# Вывод двух лучших моделей
models = tuner.get_best_models(num_models=2)

# Вывод предсказанных результатов двумя лучшими моделями
for model in models:
  model.summary()
  model.evaluate(x_test, y_test)
  print()

print("Stydy time: " + str (datetime.now() - start_time))
