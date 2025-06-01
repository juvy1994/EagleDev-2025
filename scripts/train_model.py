import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Definir la ruta de las imágenes procesadas
train_dir = "../images/processed_images"

# Definir el generador de imágenes con aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest'
)

# Cargar las imágenes de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    color_mode='grayscale',  # 👈 imagen en blanco y negro
    batch_size=32,
    class_mode='categorical'
)

# Calcular dinámicamente steps_per_epoch para evitar errores por datos insuficientes
steps_per_epoch = train_generator.samples // train_generator.batch_size
print(f"📸 Total de imágenes: {train_generator.samples}")
print(f"📦 Batch size: {train_generator.batch_size}")
print(f"🔁 Steps por epoch: {steps_per_epoch}")

# Definir la arquitectura del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Ajusta a la cantidad de clases
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    verbose=2
)

# Guardar el modelo entrenado
model.save('covid_diagnosis_model.h5')
print("✅ Modelo guardado como 'covid_diagnosis_model.h5'")
