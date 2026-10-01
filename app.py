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

# ── Настройка окружения ─────────────────────────────────────────────
device = torch.device("cpu")
save_dir = "/home/user/app/saved_predictions"
os.makedirs(save_dir, exist_ok=True)

EXAMPLES_DIR = "examples"
os.makedirs(EXAMPLES_DIR, exist_ok=True)

# ── Загрузка модели (выполняется один раз при старте) ──────────────
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet50_dr_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Grad-CAM (создаётся один раз, переиспользуется для всех запросов)
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# ── Предобработка изображения ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def _generate_example_images():
    """Создаёт два демонстрационных примера (норма и ДР), если их ещё нет."""
    try:
        existing = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith((".png", ".jpg"))]
        if existing:
            return [os.path.join(EXAMPLES_DIR, f) for f in sorted(existing)]

        rng = np.random.default_rng(42)
        examples = []

        # Пример 1: «Норма» — однородный тёмно-красный фон, светлое пятно диска зрительного нерва
        base = np.zeros((256, 256, 3), dtype=np.float32)
        base[..., 0] = 90 + rng.normal(0, 6, (256, 256))
        base[..., 1] = 25 + rng.normal(0, 5, (256, 256))
        base[..., 2] = 25 + rng.normal(0, 5, (256, 256))
        yy, xx = np.mgrid[0:256, 0:256]
        disk = 1900 / ((xx - 70) ** 2 + (yy - 128) ** 2 + 60)
        macula = 900 / ((xx - 170) ** 2 + (yy - 130) ** 2 + 150)
        base[..., 0] += disk * 60 + macula * 20
        base[..., 1] += disk * 40 + macula * 12
        base[..., 2] += disk * 30 + macula * 8
        ex1 = np.clip(base, 0, 255).astype(np.uint8)
        p1 = os.path.join(EXAMPLES_DIR, "example_normal.png")
        Image.fromarray(ex1).save(p1)
        examples.append(p1)

        # Пример 2: «ДР» — то же + яркие микроаневризмы/кровоизлияния
        base2 = base.copy()
        for _ in range(35):
            cx, cy = rng.integers(30, 226), rng.integers(30, 226)
            r = int(rng.integers(2, 6))
            m = yy >= 0
            blob = 1.0 / (((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r) + 1)
            base2[..., 0] += blob * 120
            base2[..., 1] += blob * 15
            base2[..., 2] += blob * 15
        ex2 = np.clip(base2, 0, 255).astype(np.uint8)
        p2 = os.path.join(EXAMPLES_DIR, "example_dr.png")
        Image.fromarray(ex2).save(p2)
        examples.append(p2)

        return examples
    except OSError:
        # Файловая система только для чтения — просто без примеров
        return []


EXAMPLE_FILES = _generate_example_images()

# ── Предсказание и сохранение результата ────────────────────────────

def predict_retinopathy(image):
    if image is None:
        return None, {}, "⚠️ Загрузите изображение сетчатки, чтобы получить результат."

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img = image.convert("RGB").resize((224, 224))
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    conf_dr = float(probs[0].item())
    conf_norm = float(probs[1].item())
    pred_label = "ДР" if pred == 0 else "Норма"
    confidence = conf_dr if pred == 0 else conf_norm

    # Grad-CAM
    rgb_img_np = np.ascontiguousarray(np.array(img).astype(np.float32) / 255.0)
    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(pred)])[0]
    cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
    cam_pil = Image.fromarray(cam_image)

    # Сохранение изображения с меткой и уверенностью
    try:
        filename = f"{timestamp}_{pred}_{confidence:.2f}.png"
        cam_pil.save(os.path.join(save_dir, filename))
    except OSError:
        pass  # каталог недоступен для записи — не критично

    verdict = (
        f"### 🔴 Диагностика: **диабетическая ретинопатия (ДР)**\n\n"
        f"Уверенность модели: **{confidence:.1%}**\n\n"
        f"На тепловой карте Grad-CAM видно, на какие участки сетчатки модель "
        f"обратила наибольшее внимание.\n\n⚠️ *Не является диагнозом — обратитесь к врачу-офтальмологу.*"
    ) if pred == 0 else (
        f"### 🟢 Диагностика: **норма** (признаков ДР не обнаружено)\n\n"
        f"Уверенность модели: **{confidence:.1%}**\n\n"
        f"На тепловой карте Grad-CAM видно, на какие участки сетчатки модель "
        f"обратила наибольшее внимание.\n\n⚠️ *Не является диагнозом — при жалобах обратитесь к врачу-офтальмологу.*"
    )

    label_data = {"ДР (диабетическая ретинопатия)": conf_dr, "Норма": conf_norm}
    return cam_pil, label_data, verdict


# ── Интерфейс Gradio (Blocks: тема + красивая раскладка) ───────────
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(
    theme=theme,
    title="AIMedica — Распознавание диабетической ретинопатии",
) as demo:
    gr.HTML(
        """
        <div style="text-align:center; padding: 8px 0 2px;">
          <h1 style="margin:0;">🔎 Распознавание диабетической ретинопатии</h1>
          <p style="color: var(--body-text-color-subdued); margin: 6px auto 0; max-width: 720px;">
            Модель глубокого обучения (<b>ResNet-50</b>) анализирует снимок глазного дна и определяет
            признаки диабетической ретинопатии. Тепловая карта <b>Grad-CAM</b> показывает,
            на какие области сетчатки модель обратила наибольшее внимание.
          </p>
        </div>
        """
    )

    with gr.Row():
        # ── Левая колонка: вход ──
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Снимок сетчатки (глазного дна)",
                sources=["upload", "clipboard", "webcam"],
                height=320,
            )
            with gr.Row():
                clear_btn = gr.ClearButton([input_image], value="🗑️ Очистить", size="sm")
                submit_btn = gr.Button("🔍 Проанализировать", variant="primary")

            gr.Examples(
                examples=EXAMPLE_FILES,
                inputs=[input_image],
                cache_examples=False,
            )

        # ── Правая колонка: результат ──
        with gr.Column(scale=1):
            verdict_md = gr.Markdown(
                "### Результат\n\nЗагрузите снимок и нажмите «Проанализировать».",
                label=None,
            )
            with gr.Accordion("Подробности анализа", open=True):
                output_image = gr.Image(
                    type="pil",
                    label="Тепловая карта Grad-CAM",
                    height=320,
                    interactive=False,
                )
                prob_label = gr.Label(
                    label="Вероятности по классам",
                    num_top_classes=2,
                )

    gr.HTML(
        """
        <div style="text-align:center; padding: 14px 0 4px; color: var(--body-text-color-subdued); font-size: 0.9em;">
          Модель: ResNet-50 (PyTorch) · Объяснение предсказаний: Grad-CAM ·
          <a href="https://huggingface.co/spaces/aimedica/AIMedica_Deep_Learning_for_Ophthalmologist" target="_blank">
          Исходный Space на Hugging Face</a>
          <br><br>
          <b>Метрики и источники:</b> ориентиры ResNet-50 на датасете
          <a href="https://www.kaggle.com/competitions/aptos2019-blindness-detection" target="_blank">APTOS 2019</a> —
          Accuracy 93–96 % · AUC-ROC 0,94–0,97 · Sensitivity 90–95 %
          (<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10114328/" target="_blank">Lin et al., 2023</a> ·
          <a href="https://arxiv.org/html/2507.17121v1" target="_blank">arXiv 2025</a> ·
          <a href="https://jamanetwork.com/journals/jama/fullarticle/2588763" target="_blank">Gulshan et al., JAMA 2016</a> ·
          <a href="https://arxiv.org/abs/1610.02391" target="_blank">Grad-CAM, ICCV 2017</a>).
          <br><br>
          ⚠️ Приложение носит демонстрационный характер и <b>не заменяет консультацию врача-офтальмолога</b>.
          В экстренных ситуациях обращайтесь за неотложной медицинской помощью.
        </div>
        """
    )

    submit_btn.click(
        fn=predict_retinopathy,
        inputs=[input_image],
        outputs=[output_image, prob_label, verdict_md],
        api_name="predict",
    )

    input_image.upload(
        fn=lambda: "### Результат\n\nИзображение загружено — нажмите «Проанализировать».",
        inputs=None,
        outputs=[verdict_md],
        queue=False,
    )


if __name__ == "__main__":
    demo.launch()
