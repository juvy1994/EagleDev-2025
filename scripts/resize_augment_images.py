import cv2
import os
import numpy as np
import albumentations as A
from pathlib import Path

# Definir la ruta de las imágenes originales y las imágenes procesadas
input_dir = Path("../images/raw_images")
output_dir = Path("../images/processed_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Definir las transformaciones de aumento de datos
transform = A.Compose([
    A.Resize(150, 150),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=0.5),
    A.RandomGamma(p=0.2)
])

print("Contenido de raw_images:")
print(os.listdir(input_dir))

for class_name in ["COVID", "VIRAL PNEUMONIA", "NORMAL"]:
    class_path = input_dir / class_name
    if class_path.exists():
        print(f"Existe carpeta: {class_path}")
    else:
        print(f"NO existe carpeta: {class_path}")

# Función para procesar y guardar las imágenes
def process_images(class_name):
    class_dir = input_dir / class_name / "images"
    print(f"Procesando imágenes en: {class_dir}")

    if not class_dir.exists():
        print(f"❌ La carpeta no existe: {class_dir}")
        return

    output_class_dir = output_dir / class_name
    if output_class_dir.exists() and any(output_class_dir.iterdir()):
        print(f"⏭️ Carpeta de salida ya existe y contiene imágenes, se salta procesamiento: {output_class_dir}")
        return

    output_class_dir.mkdir(parents=True, exist_ok=True)
    print(f"Carpeta de salida creada: {output_class_dir}")

    for img_name in os.listdir(class_dir):
        img_path = class_dir / img_name
        if img_path.is_file():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ No se pudo leer la imagen: {img_path}")
                continue

            img = np.expand_dims(img, axis=-1)

            augmented = transform(image=img)
            augmented_img = augmented["image"].astype(np.uint8)

            output_img_path = output_class_dir / img_name
            cv2.imwrite(str(output_img_path), augmented_img)
            print(f"Imagen procesada y guardada en: {output_img_path}")

# Procesar todas las clases
for class_name in ["COVID", "VIRAL PNEUMONIA", "NORMAL"]:
    process_images(class_name)
