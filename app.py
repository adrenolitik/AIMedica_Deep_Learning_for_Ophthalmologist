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
import csv
import datetime
import zipfile

# ✅ Admin key (hidden until typed)
ADMIN_KEY = "Diabetes_Detection"

# Set device
device = torch.device("cpu")

# Load model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_dr_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Grad-CAM setup
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Data storage
image_folder = "collected_images"
os.makedirs(image_folder, exist_ok=True)

csv_log_path = "prediction_logs.csv"
if not os.path.exists(csv_log_path):
    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image_filename", "prediction", "confidence"])

# Prediction function
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

    # Save image & log
    image_filename = f"{timestamp}_{label.replace(' ', '_')}.png"
    image_path = os.path.join(image_folder, image_filename)
    image.save(image_path)

    with open(csv_log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, image_filename, label, f"{confidence:.4f}"])

    return cam_pil, f"{label} (Confidence: {confidence:.2f})"

# Admin unlock
def unlock_admin(key_input):
    if key_input == ADMIN_KEY:
        return gr.update(visible=True)
    return gr.update(visible=False)

# Download functions
def download_csv():
    return csv_log_path

def download_dataset_zip():
    zip_filename = "dataset_bundle.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(csv_log_path, arcname="prediction_logs.csv")
        for fname in os.listdir(image_folder):
            fpath = os.path.join(image_folder, fname)
            zipf.write(fpath, arcname=os.path.join("images", fname))
    return zip_filename

# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Diabetic Retinopathy Detection with Grad-CAM")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Retinal Image")
        cam_output = gr.Image(type="pil", label="Grad-CAM Output")

    prediction_output = gr.Text(label="Prediction")
    run_button = gr.Button("Submit")

    run_button.click(
        fn=predict_retinopathy,
        inputs=image_input,
        outputs=[cam_output, prediction_output]
    )

    gr.Markdown("### 🔐 Admin Access (Rodiyah only)")

    admin_key_input = gr.Text(label="Enter Admin Key", type="password", placeholder="Only Rodiyah knows this!")
    unlock_button = gr.Button("Unlock Downloads")

    with gr.Column(visible=False) as admin_panel:
        gr.Markdown("### ✅ Download Panel (Private Access)")
        with gr.Row():
            download_csv_btn = gr.Button("📄 Download CSV Log")
            download_zip_btn = gr.Button("📦 Download Full Dataset")
        csv_file = gr.File()
        zip_file = gr.File()

    unlock_button.click(
        fn=unlock_admin,
        inputs=admin_key_input,
        outputs=admin_panel
    )

    download_csv_btn.click(fn=download_csv, inputs=[], outputs=csv_file)
    download_zip_btn.click(fn=download_dataset_zip, inputs=[], outputs=zip_file)

demo.launch()
