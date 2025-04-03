import gradio as gr
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import io
import csv
import datetime

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

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# In-memory log list
prediction_log = [["timestamp", "image_name", "prediction", "confidence"]]

# Logging function
def log_prediction(filename, prediction, confidence):
    timestamp = datetime.datetime.now().isoformat()
    row = [timestamp, filename, prediction, f"{confidence:.4f}"]
    prediction_log.append(row)
    print("⏺ Logged:", row)

# Prediction function
def predict_retinopathy(image):
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

    # Log it
    filename = getattr(image, "filename", "uploaded_image")
    log_prediction(filename, label, confidence)

    cam_pil = Image.fromarray(cam_image)
    return cam_pil, f"{label} (Confidence: {confidence:.2f})"

# CSV download function
def download_logs():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(prediction_log)
    output.seek(0)

    # Save as a temporary file for download
    with open("prediction_logs.csv", "w", newline="") as f:
        f.write(output.getvalue())

    return "prediction_logs.csv"


# Build the UI with Gradio Blocks
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Diabetic Retinopathy Detection with Grad-CAM & Logging")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Retinal Image")
        cam_output = gr.Image(type="pil", label="Grad-CAM")

    prediction_output = gr.Text(label="Prediction")

    with gr.Row():
        run_button = gr.Button("Submit")
        download_button = gr.Button("📥 Download Logs")
        download_file = gr.File(label="Your Log File", interactive=False)

    run_button.click(
        fn=predict_retinopathy,
        inputs=image_input,
        outputs=[cam_output, prediction_output]
    )

    download_button.click(
        fn=download_logs,
        inputs=[],
        outputs=download_file
    )

demo.launch()
