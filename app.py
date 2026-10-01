import gradio as gr
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import os
import datetime

# Настройка окружения
device = torch.device("cpu")
save_dir = "/home/user/app/saved_predictions"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("📁 Папка создана:", save_dir)
os.makedirs(save_dir, exist_ok=True)

# Загрузка модели
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_dr_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Grad-CAM
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# Предобработка изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Предсказание и сохранение результата
def predict_retinopathy(image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img = image.convert("RGB").resize((224, 224))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "ДР" if pred == 0 else "Норма"

    # Grad-CAM
    rgb_img_np = np.array(img).astype(np.float32) / 255.0
    rgb_img_np = np.ascontiguousarray(rgb_img_np)
    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred)])[0]
    cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
    cam_pil = Image.fromarray(cam_image)

    # Сохранение изображения с меткой и уверенностью
    filename = f"{timestamp}_{pred}_{confidence:.2f}.png"
    cam_pil.save(os.path.join(save_dir, filename))

    return cam_pil, f"{label} (Уверенность: {confidence:.2%})"

# Интерфейс Gradio
gr.Interface(
    fn=predict_retinopathy,
    inputs=gr.Image(type="pil", label="Снимок сетчатки"),
    outputs=[
        gr.Image(type="pil", label="Тепловая карта Grad-CAM"),
        gr.Text(label="Результат анализа")
    ],
    title="🔎 Распознавание диабетической ретинопатии",
    description=(
        "Загрузите изображение глазного дна (сетчатки), и модель глубокого обучения "
        "определит признаки диабетической ретинопатии (ДР). "
        "Тепловая карта Grad-CAM показывает области снимка, на которые модель обратила наибольшее внимание.\n\n"
        "⚠️ Приложение носит демонстрационный характер и не заменяет консультацию врача-офтальмолога."
    ),
    article="Модель: ResNet-50 · Объяснение предсказаний: Grad-CAM · <a href='https://huggingface.co/spaces/aimedica/AIMedica_Deep_Learning_for_Ophthalmologist' target='_blank'>Исходный Space на Hugging Face</a>"
).launch()
