# Функция активации: пороговая функция
import random


def step_function(x):
    """
    Пороговая функция активации.
    Возвращает 1, если входное значение x >= 0, иначе возвращает 0.
    """
    return 1 if x >= 0 else 0


# Класс формального нейрона
class FormalNeuron:
    def __init__(self, weights, bias, activation_func):
        """
        Конструктор класса FormalNeuron.
        :param weights: список весов нейрона
        :param bias: смещение (bias) нейрона
        :param activation_func: функция активации нейрона
        """
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func

    def predict(self, x):
        """
        Метод для предсказания выхода нейрона.
        :param x: входной вектор (список значений)
        :return: результат работы нейрона после применения функции активации
        """
        z = sum(x_i * w_i for x_i, w_i in zip(x, self.weights)) + self.bias
        return self.activation_func(z)


# Класс персептрона
class Perceptron:
    def __init__(self, n_inputs, n_outputs, activation_func=step_function, learning_rate=0.1):
        """
        Конструктор класса Perceptron.
        :param n_inputs: количество входных параметров (например, количество пикселей в изображении)
        :param n_outputs: количество выходных нейронов (например, количество классов)
        :param activation_func: функция активации (по умолчанию step_function)
        :param learning_rate: скорость обучения (по умолчанию 0.1)
        """
        self.learning_rate = learning_rate
        self.neurons = []
        for _ in range(n_outputs):
            weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
            bias = random.uniform(-1, 1)
            neuron = FormalNeuron(weights, bias, activation_func)
            self.neurons.append(neuron)
            print(f"Создан нейрон {i + 1} с весами: {weights} и смещением: {bias:.2f}")

    def predict(self, x):
        return [neuron.predict(x) for neuron in self.neurons]


def train(self, X, y, epochs=10):
    """
    Метод для обучения персептрона.
    :param X: список входных векторов (обучающие данные)
    :param y: список целевых векторов (метки классов в one-hot encoding)
    :param epochs: количество эпох обучения (по умолчанию 10)
    :return: список суммарных ошибок для каждой эпохи
    """
    errors = []
    for epoch in range(epochs):
        epoch_error = 0
        for inputs, targets in zip(X, y):
            outputs = self.predict(inputs)
            for i, neuron in enumerate(self.neurons):
                error = targets[i] - outputs[i]
                epoch_error += abs(error)
                for j in range(len(neuron.weights)):
                    neuron.weights[j] += self.learning_rate * error * inputs[j]
                neuron.bias += self.learning_rate * error
        errors.append(epoch_error)
        print(f"Эпоха {epoch + 1}/{epochs}, суммарная ошибка: {epoch_error}")
    return errors


# Квадрат
square = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
# Треугольник
triangle = [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Круг
circle = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0]

# Данные для обучения
X = [
    square,  # Квадрат
    triangle,  # Треугольник
    circle,  # Круг
    square,  # Квадрат
    triangle,  # Треугольник
    circle,  # Круг
]
# Метки классов (one-hot encoding):
y = [
    [1, 0, 0],  # Квадрат
    [0, 1, 0],  # Треугольник
    [0, 0, 1],  # Круг
    [1, 0, 0],  # Квадрат
    [0, 1, 0],  # Треугольник
    [0, 0, 1],  # Круг
]

# Создание и обучение персептрона
n_inputs = len(X[0])
n_outputs = len(y[0])
perceptron = Perceptron(n_inputs, n_outputs, learning_rate=0.1)

# Обучение сети
epochs = 20
errors = perceptron.train(X, y, epochs)

# Тестирование сети
test_data = [
    square,  # Квадрат
    triangle,  # Треугольник
    circle,  # Круг
    # Искаженные данные
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],  # Квадрат с небольшим искажением
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Треугольник с небольшим искажением
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],  # Круг с небольшим искажением
]

# Тестируем сеть на новых данных
for i, data in enumerate(test_data):
    output = perceptron.predict(data)
    predicted_class = output.index(max(output))
    if predicted_class == 0:
        print(f"Тестовый пример {i + 1}: Предсказано: Квадрат")
    elif predicted_class == 1:
        print(f"Тестовый пример {i + 1}: Предсказано: Треугольник")
    elif predicted_class == 2:
        print(f"Тестовый пример {i + 1}: Предсказано: Круг")

