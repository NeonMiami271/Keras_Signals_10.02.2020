#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Скрип для предсказания значения единичного сигнала (один сигнал в файле)
# уже обученной и сохраненной нейронной сети

from tensorflow.keras.models import load_model
import numpy

# Загрузка критериев стандартизации и данных для тестирования
mean = numpy.loadtxt("mean.csv", delimiter=",")
std = numpy.loadtxt("std.csv", delimiter=",")
x_test = numpy.array([numpy.loadtxt("/Dataset/test.csv", delimiter=",")])
x_test -= mean
x_test /= std

# Загрузка модели нейронной сети
model = load_model('my_model.h5')
print("Model download!")

# Сравнение реального результата с результатам, предсказанным нейронной сетью
prediction = model.predict(x_test)
print("%.0f" % prediction[0][0])
