import numpy as np


# Сигмоидная функция активации и ее производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Класс нейронной сети
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):

        """
        Параметры:
            input_size (int): Количество входных признаков
            hidden_size (int): Количество нейронов в скрытом слое
            output_size (int): Количество нейронов в выходном слое
        """

        self.weights_hidden = np.random.rand(input_size, hidden_size)  # Веса для скрытого слоя
        self.bias_hidden = np.random.rand(1, hidden_size)  # Смещения для скрытого слоя
        self.weights_output = np.random.rand(hidden_size, output_size)  # Веса для выходного слоя
        self.bias_output = np.random.rand(1, output_size)  # Смещения для выходного слоя

    def forward(self, X):

        """
       Прямое распространение (forward propagation)

        Параметры:
            X (numpy.ndarray): Входные данные (матрица размера [n_samples, input_size]).

        Возвращает:
            numpy.ndarray: Выходные данные сети (предсказания).
        """

        self.hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)  # Активация скрытого слоя
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)  # Активация выходного слоя
        return self.output

    def backward(self, X, y, learning_rate):

        """
        Обратное распространение ошибки (backpropagation)

        Параметры:
            X (numpy.ndarray): Входные данные
            y (numpy.ndarray): Целевые значения
            learning_rate (float): Скорость обучения
        """

        # Ошибка на выходном слое
        error_output = y - self.output
        d_output = error_output * sigmoid_derivative(self.output)

        # Ошибка на скрытом слое
        error_hidden = d_output.dot(self.weights_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_layer_output)

        # Обновление весов и смещений
        self.weights_output += self.hidden_layer_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, learning_rate, epochs):
        # Обучение сети
        for epoch in range(epochs):
            output = self.forward(X)  # Прямое распространение
            self.backward(X, y, learning_rate)  # Обратное распространение

            # Вывод ошибки каждые 100 эпох
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))  # Средняя квадратичная ошибка (MSE)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        # Предсказание на новых данных
        output = self.forward(X)
        return np.round(output)  # Округляем выходные значения до 0 или 1


# Алиса: 54 кг 170 cм Ж; Борис: 65 кг 183 см М; Иван: 62 кг 175 см М; Диана: 49 кг 152 см Ж
# Вес -55, рост -165

# Инициализация данных
X = np.array([[-1, 5], [10, 18], [7, 10], [-6, -13]])  # Входные данные (рост и вес)
y = np.array([[0], [1], [1], [0]])  # Метки (пол: 0 - женский, 1 - мужской)

# Создание и обучение нейронной сети
input_size = 2  # Количество входных признаков (рост и вес)
hidden_size = 2  # Количество нейронов в скрытом слое
output_size = 1  # Количество выходных нейронов (пол)

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y, learning_rate=0.1, epochs=5000)

pol = ['Женщина', 'Мужчина']  # Женский и мужской пол

lena = np.array([0, 4])  # Лена: 55 кг, 169 см (сдвинутые данные)
fedya = np.array([13, 15])  # Федя: 68 кг, 174 см (сдвинутые данные)

# Предсказание для тестовых данных
print("Лена: ", pol[int(nn.predict(lena))])  # Предсказанный пол для Лены
print("Федя: ", pol[int(nn.predict(fedya))])  # Предсказанный пол для Феди
