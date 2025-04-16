# Импорт необходимых библиотек
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование изображений в одномерные массивы
num_pixels = X_train.shape[1] * X_train.shape[2]  # 28 * 28 = 784 пикселя
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')  # Преобразуем в 2D-массив
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')  # Преобразуем в 2D-массив

# Нормализация данных (приведение значений пикселей к диапазону [0, 1])
X_train = X_train / 255
X_test = X_test / 255

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train)  # Например, 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_test = to_categorical(y_test)

# Создание модели
model = Sequential()  # Создаем последовательную модель

# Добавляем скрытый слой с 784 нейронами и функцией активации ReLU
model.add(Dense(784, input_dim=num_pixels, activation='relu'))

# Добавляем выходной слой с 10 нейронами (по одному для каждого класса) и функцией активации softmax
model.add(Dense(10, activation='softmax'))

# Компиляция модели
model.compile(optimizer='adam',  	       # Используем оптимизатор Adam
              loss='categorical_crossentropy',  # Функция потерь для многоклассовой классификации
              metrics=['accuracy'])  	# Метрика — точность (accuracy)

# Обучение модели
history = model.fit(X_train, y_train,  # Обучающие данные
                    validation_data=(X_test, y_test),  # Тестовые данные для валидации
                    epochs=10,  # Количество эпох
                    batch_size=200,  # Размер батча
                    verbose=2)  # Вывод информации о процессе обучения

# Оценка качества модели на тестовых данных
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность на тестовых данных: {scores[1] * 100:.2f}%")


