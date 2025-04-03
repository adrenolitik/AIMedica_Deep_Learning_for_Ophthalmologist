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
from gradio.routes import Request

# 🔐 Secret key
ADMIN_KEY = "Diabetes_Detection"

# Device
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

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Folders
image_folder = "collected_images"
os.makedirs(image_folder, exist_ok=True)

csv_log_path = "prediction_logs.csv"
if not os.path.exists(csv_log_path):
    with open(csv_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "image_filename", "prediction", "confidence"])

# 🔍 Prediction
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

    # Save image + log
    image_filename = f"{timestamp}_{label.replace(' ', '_')}.png"
    image_path = os.path.join(image_folder, image_filename)
    image.save(image_path)

    with open(csv_log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, image_filename, label, f"{confidence:.4f}"])

    return cam_pil, f"{label} (Confidence: {confidence:.2f})"

# 📁 Admin downloads
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

# ✅ Admin check (query param)
def is_admin(request: Request):
    return request and request.query_params.get("admin") == ADMIN_KEY

# 🌐 App
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Diabetic Retinopathy Detection with Grad-CAM")

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

    # 🔒 Hidden admin section
    with gr.Column(visible=False) as admin_section:
        gr.Markdown("### 🔐 Private Downloads (Rodiyah Only)")
        with gr.Row():
            download_csv_btn = gr.Button("📄 Download CSV Log")
            download_zip_btn = gr.Button("📦 Download Dataset ZIP")
        csv_file = gr.File()
        zip_file = gr.File()

    # ✅ Reveal only if correct ?admin=Diabetes_Detection in URL
    demo.load(
        fn=lambda req: gr.update(visible=True) if is_admin(req) else gr.update(visible=False),
        inputs=[],
        outputs=admin_section,
        queue=False,
        api_name=False,
        request=True  # ✅ Required to pass HTTP request into lambda
    )

    download_csv_btn.click(fn=download_csv, inputs=[], outputs=csv_file)
    download_zip_btn.click(fn=download_dataset_zip, inputs=[], outputs=zip_file)

demo.launch()
