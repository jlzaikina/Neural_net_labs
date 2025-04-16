# Импорт библиотек
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Загрузка и подготовка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(), # Преобразование изображения в тензоры PyTorch
    transforms.Normalize((0.5,), (0.5,)) # Нормализация данных (среднее=0.5, стандартное отклонение=0.5)
])

# Загрузка тренировочных и тестовых данных
train_set = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Создание DataLoader
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # Перемешиваем данные
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False) # Не перемешиваем тестовые данные

# Визуализация нескольких примеров
def show_examples(loader):
    data_iter = iter(loader) # Итератор для тренировочных данных
    images, labels = next(data_iter) # Берем первый батч

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()

show_examples(train_loader)
# Определение модели
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) # Полносвязный слой (вход: 28x28 пикселей, выход: 128 нейронов)
        self.relu = nn.ReLU() # Функция активации ReLU
        self.fc2 = nn.Linear(128, 10) # Выходной слой (10 классов - цифры от 0 до 9)

    def forward(self, x):
        # Прямой проход (forward pass)
        x = x.view(-1, 28 * 28)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
model = MNISTClassifier()
criterion = nn.CrossEntropyLoss() # Функция потерь для задачи классификации
optimizer = optim.Adam(model.parameters(), lr=0.001) # Оптимизатор Adam с шагом обучения 0.001

# Функция обучения
def train(model, loader, criterion, optimizer, epochs=10):
    model.train() # Устанавливаем модель в режим обучения
    for epoch in range(epochs): # Проходимся по количеству эпох
        running_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images) # Прямой проход (forward pass)
            loss = criterion(outputs, labels) # Вычисляем потери
            loss.backward()  # Обратный проход (backward pass): вычисляем градиенты
            optimizer.step() # Шаг оптимизации: обновляем параметры модели
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader):.4f}')

# Функция тестирования
def test(model, loader):
    model.eval()  # Устанавливаем модель в режим тестирования
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images) # Прямой проход (forward pass)
            _, predicted = torch.max(outputs.data, 1) # Получаем индексы максимальных значений (предсказанные классы)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Обучение и тестирование модели
print("Starting training...")
train(model, train_loader, criterion, optimizer, epochs=10)

print("\nTesting model...")
test(model, test_loader)

# Визуализация предсказаний
def show_predictions(loader):
    data_iter = iter(loader)
    images, labels = next(data_iter)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}')
        plt.axis('off')
    plt.show()

show_predictions(test_loader)

