#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Скрипт для обучения рекуррентной нейронной сети

from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime
import time
from keras.models import model_from_json
from tensorflow.keras.models import load_model
import csv

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

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Функция остановки обучения при переобучении (patience - параметр "терпимости" к переобучению)
early_stopping_callback = EarlyStopping(monitor='val_mae', patience = 2)

# Функция, начинающая процесс обучения
history = model.fit(x_train, y_train, epochs=20, batch_size=4000, validation_split = 0.2, verbose=1, callbacks = [early_stopping_callback])

# Оценка качества обучения нейронной сети на проверочном наборе данных
mse, mae = model.evaluate(x_test, y_test, verbose=0)

# Сравнение реальных результатов с результатами, предсказанными нейронной сетью
pred = model.predict(x_test)
print("Predict amplitude      Actual amplitude       Different         %")
for i in range(45):
    per = ((y_test[i] * 100) / pred[i][0]) - 100
    if (per < 0): per = per * (-1)
    diff = pred[i][0] - y_test[i]
    if (diff < 0): diff = diff * (-1)
    print("      %.0f               %.0f            %.0f          %.0f" % (pred[i][0], y_test[i], diff, per))
print("MAE: ", mae)

# Сохранение модели нейронной сети
model.save("my_model.h5")

# Запись в отдельный файл данных стандартизации
# для последующего использования в скриптах с загрузкой модели
with open("mean.csv", "wb") as file:
    writer = csv.writer(file)
    writer.writerow(mean)

with open("std.csv", "wb") as file:
    writer = csv.writer(file)
    writer.writerow(std)

print("Model saved!")

# Вывод структуры нейронной сети
model.summary()

print("Study break in epoch:", early_stopping_callback.stopped_epoch)
print("Stydy time: " + str (datetime.now() - start_time))

# График процесса обучения
plt.plot(history.history['mae'], label='mae in training set')
plt.plot(history.history['val_mae'], label='mae in testing set')
plt.xlabel('Study epoch')
plt.ylabel('mae')
plt.legend()
plt.show()
