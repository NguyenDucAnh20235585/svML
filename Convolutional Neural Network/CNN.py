import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from sklearn.model_selection import train_test_split
x_train, x_train_2, y_train, y_train_2 = train_test_split(
    x_train, y_train, test_size=0.4, random_state=42
)

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train_2 = x_train_2 / 255.0
y_train = to_categorical(y_train, 10)
y_train_2 = to_categorical(y_train_2, 10)
y_test = to_categorical(y_test, 10)

train_datagen = ImageDataGenerator(
    rotation_range = 40,      # Xoay ngẫu nhiên từ 0 đến 40 độ qua trái hoặc phải
    width_shift_range = 0.2,  # Dịch hình ảnh qua trái hoặc phải từ 0 đến 20%
    height_shift_range = 0.2, # Dịch hình ảnh lên hoặc xuống từ 0 đến 20%
    shear_range = 0.2,        # Cắt nghiên hình ảnh từ 0 đến 20%
    zoom_range = 0.2,         # Phóng to, thu nhỏ hình ảnh từ 0 đến 20%
    horizontal_flip = True,   # Lật ngang hình ảnh
    fill_mode = 'nearest'     # Các pixel mới được tạo ra dựa trên các pixel gần nhất
)

train_images_augmented = train_datagen.flow(
    x_train_2,
    y_train_2,
    batch_size=32,
)

x_train_full = np.concatenate([x_train, x_train_2], axis=0)
y_train_full = np.concatenate([y_train, y_train_2], axis=0)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    verbose=1,
    restore_best_weights=True
)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 5))
predictions = model.predict(x_test[:21])
for i in range(18):
    plt.subplot(3, 6, i+1)
    plt.imshow(x_test[i].reshape(32, 32, 3), cmap='gray')
    plt.title(f"Label: {label[predictions[i].argmax()]}")
    plt.axis('off')
plt.show()