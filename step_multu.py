#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Скрип для предсказания группы данных (несколько сигналов в одном файле)
# значений уже обученной и сохраненной нейронной сети

from tensorflow.keras.models import load_model
import numpy

# Загрузка критериев стандартизации и данных для тестирования
mean = numpy.loadtxt("mean.csv", delimiter=",")
std = numpy.loadtxt("std.csv", delimiter=",")
x_test = numpy.loadtxt("/home/vladlen/Рабочий стол/deep_learning/find_signals/07_02_2020/Dataset/Data_Sets_Time/Dataset/Dataset_Full_Test.csv", delimiter=",")
y_test = numpy.loadtxt("/home/vladlen/Рабочий стол/deep_learning/find_signals/07_02_2020/Dataset/Data_Sets_Time/Answer/Answers_Full_Test.csv", delimiter=",")
x_test -= mean
x_test /= std

# Загрузка модели нейронной сети
model = load_model('my_model.h5')
print("Model download!")

# Сравнение реальных результатов с результатами, предсказанными нейронной сетью
print("Predict Data:          Real Data:")
for i in range(len(y_test)):
    print("   %.0f                  %.0f" % (model.predict(x_test)[i], y_test[i]))
