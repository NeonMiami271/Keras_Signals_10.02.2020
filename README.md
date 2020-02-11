1. Устанавливаем интерпретатор python и загружаем необходимые библиотеки:

                                sudo apt-get install -y python python-dev
                                sudo apt-get install python-pip
                                pip install -r requirements
                                pip install -U keras-tuner

2. main.py - основной скрипт для обучения и сохранения обученной рекуррентной нейронной сети 
   
   main_tuner.py - скрипт для подбора гиперпараметров нейронной сети
   
   step_multu.py - скрипт для предсказания результатов данных (синалов), записанных в один файл
   
   step_single.py - скрипт для предсказания результата одного сигнала

3. Папка Dataset с данными для обучения
