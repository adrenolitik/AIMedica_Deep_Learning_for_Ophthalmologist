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
import sqlite3

# === Setup paths and model ===
device = torch.device("cpu")
ADMIN_KEY = "Diabetes_Detection"
image_folder = "collected_images"
os.makedirs(image_folder, exist_ok=True)

# === Load model ===
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_dr_classifier.pth", map_location=device))
model.to(device)
model.eval()

# === Grad-CAM setup ===
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === SQLite setup ===
def init_db():
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            prediction TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def log_to_db(timestamp, filename, prediction, confidence):
    conn = sqlite3.connect("logs.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (timestamp, filename, prediction, confidence) VALUES (?, ?, ?, ?)",
                   (timestamp, filename, prediction, confidence))
    conn.commit()
    conn.close()

init_db()  # ✅ Initialize table

# === Prediction Function ===
def predict_retinopathy(image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img = image.convert("RGB").resize((224, 224))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "Diabetic Retinopathy (DR)" if pred == 0 else "No DR"

    # Grad-CAM
    rgb_img_np = np.array(img).astype(np.float32) / 255.0
    rgb_img_np = np.ascontiguousarray(rgb_img_np)
    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred)])[0]
    cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
    cam_pil = Image.fromarray(cam_image)

    # Save image and log
    filename = f"{timestamp}_{label.replace(' ', '_')}.png"
    image_path = os.path.join(image_folder, filename)
    image.save(image_path)

    log_to_db(timestamp, image_path, label, confidence)

    return cam_pil, f"{label} (Confidence: {confidence:.2f})"

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 DR Detection with Grad-CAM + SQLite Logging")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Retinal Image")
        cam_output = gr.Image(type="pil", label="Grad-CAM")

    prediction_output = gr.Text(label="Prediction")
    run_button = gr.Button("Submit")

    run_button.click(
        fn=predict_retinopathy,
        inputs=image_input,
        outputs=[cam_output, prediction_output]
    )

demo.launch()
