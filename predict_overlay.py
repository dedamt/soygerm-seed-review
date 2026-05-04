import os
import cv2
import numpy as np
from ultralytics import YOLO

# caminho do modelo treinado
model = YOLO(r"C:\soygerm\runs\segment\teste_yolo11n_seg_soja-2\weights\best.pt")

# pasta com imagens
input_folder = r"C:\soygerm\data\images\train"

# pasta de saída
output_folder = r"C:\soygerm\resultado_overlay"
os.makedirs(output_folder, exist_ok=True)

# cor verde em BGR
mask_color = (0, 255, 0)

# transparência da máscara
alpha = 0.45

# extensões aceitas
valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(valid_ext):
        continue

    image_path = os.path.join(input_folder, file_name)

    # predição
    results = model.predict(
        source=image_path,
        imgsz=1024,
        conf=0.25,
        device="cpu",
        save=False
    )

    # lê imagem original
    image = cv2.imread(image_path)
    overlay = image.copy()

    for result in results:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []

            for i, mask in enumerate(masks):
                # se quiser somente a classe "germinadas", filtre aqui
                # assumindo que 0 = germinadas
                if len(classes) > i and int(classes[i]) != 0:
                    continue

                # redimensiona máscara para o tamanho da imagem
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                # aplica cor verde na região segmentada
                overlay[mask_resized == 1] = mask_color

    # mistura overlay com imagem original
    result_img = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, result_img)

print(f"Imagens salvas em: {output_folder}")