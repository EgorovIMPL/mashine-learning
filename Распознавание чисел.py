from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Создание и обучение нейронной сети
def create_model():
    model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

# Обучение модели на MNIST
def train_model(model):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    model.save("model.h5")

model = create_model()
train_model(model)


def preprocess_image(image_path):
    # Загружаем изображение и инвертируем цвета
    binary_image = cv2.threshold(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), 128, 255, cv2.THRESH_BINARY_INV)[1]

    # Убираем шумы и находим контуры
    binary_image = cv2.medianBlur(binary_image, 3)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Получаем bounding boxes и Y-координаты
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    y_coords = np.array([[y] for _, y, _, _ in bounding_boxes])

    # Кластеризация по Y-координате для определения строк
    clustering = DBSCAN(eps=40, min_samples=1).fit(y_coords)

    digit_images = []
    row_clusters = {}

    # Группируем и обрабатываем цифры по строкам
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        label = clustering.labels_[i]
        digit = binary_image[y:y+h, x:x+w]

        # Масштабируем и центрируем в 28x28
        scale = 20 / max(h, w)
        resized = cv2.resize(digit, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        padded = np.zeros((28, 28), dtype=np.uint8)
        padded[(28 - resized.shape[0]) // 2:(28 - resized.shape[0]) // 2 + resized.shape[0], 
               (28 - resized.shape[1]) // 2:(28 - resized.shape[1]) // 2 + resized.shape[1]] = resized

        row_clusters.setdefault(label, []).append((x, padded))  # Используем setdefault для упрощения

    # Сортируем строки и цифры
    for label in sorted(row_clusters.keys(), key=lambda l: min(bounding_boxes[i][1] for i in range(len(bounding_boxes)) if clustering.labels_[i] == l)):
        digit_images.extend(img for _, img in sorted(row_clusters[label], key=lambda item: item[0]))

    return digit_images

def recognize_digits(model, digit):
    # Нормализация изображения
    digit_normalized = np.expand_dims(digit.astype('float32') / 255.0, axis=(-1, 0))
    return np.argmax(model.predict(digit_normalized))

def display_digits(digits, model):
    for digit in digits:
        recognized_digit = recognize_digits(model, cv2.resize(digit, (28, 28)))
        plt.imshow(digit.squeeze(), cmap='gray')
        plt.title(f'Цифра: {recognized_digit}')
        plt.axis('off')
        plt.show()

# Запуск
model = load_model("model.h5")
image_path = "/content/Снимок экрана 2025-02-01 135702.png"
digits = preprocess_image(image_path)

display_digits(digits, model)
