# 🚀 Prompt to Image Generator with prompt engineering comparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?style=for-the-badge&logo=huggingface)
![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-v1.5-orange?style=for-the-badge)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-green?style=for-the-badge&logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

> A hands-on AI project that demonstrates how **prompt engineering dramatically affects image generation quality** using Stable Diffusion v1.5 — comparing basic, detailed, and negative-prompt strategies with full inference logging.

---


## 🧠 About the Project

This project explores how **text prompt construction influences AI-generated image quality** using the `runwayml/stable-diffusion-v1-5` model from Hugging Face. The core idea is simple but powerful — the same model, the same seed, and the same hyperparameters can produce drastically different images based purely on how the prompt is written.

Three prompt variants are compared side by side:

| Variant | Strategy |
|---|---|
| **Basic** | Minimal subject description only |
| **Detailed** | Rich keywords + weighted tokens + composition control |
| **Detailed + Negative** | All of the above + strong negative prompt suppression |

Every run is logged with inference time, parameters, and saved outputs — making this a complete end-to-end AI experimentation pipeline.

---

## 🖼️ Demo Output

The project generates a **vertical comparison image** showing all three outputs stacked top to bottom, with:
- Inference time badge on each image
- Prompt preview below each image
- Negative prompt highlighted in red (where applicable)

```
generated_images/
├── basic_result.png          ← Basic prompt output
├── detailed_result.png       ← Detailed prompt output
├── negative_result.png       ← Detailed + negative prompt output
├── comparison_vertical.png   ← Side-by-side visual comparison
└── inference_log.json        ← Full parameter & timing log
```

---

## ✅ What This Project Demonstrates

- ✔️ **Diffusion model usage** via Hugging Face Diffusers pipeline
- ✔️ **Prompt conditioning** — how text guides the denoising process
- ✔️ **Prompt weighting** using `(( ))` syntax and `:weight` multipliers
- ✔️ **Negative prompt engineering** to suppress unwanted elements
- ✔️ **Composition control** — centering subjects, controlling framing
- ✔️ **Inference logging** — tracking time, parameters, and outputs
- ✔️ **Reproducibility** — fixed seed for fair comparison across variants
- ✔️ **API integration** — Hugging Face model hub + Diffusers library
- ✔️ **Experimentation mindset** — isolating one variable (the prompt) to evaluate impact

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.8+` | Core language |
| `Hugging Face Diffusers` | Stable Diffusion pipeline |
| `runwayml/stable-diffusion-v1-5` | Pre-trained text-to-image model |
| `PyTorch` | Deep learning backend |
| `xformers` | Memory-efficient attention (VRAM optimization) |
| `Matplotlib` | Image visualization & comparison layout |
| `Pillow (PIL)` | Image saving and handling |
| `Google Colab (T4 GPU)` | Free cloud GPU environment |
| `JSON` | Inference log storage |



---

## 🚀 Getting Started

### Prerequisites

- A **Hugging Face account** (free) — [Sign up here](https://huggingface.co/join)
- A **Hugging Face access token** — [Get it here](https://huggingface.co/settings/tokens)
- Access to **Google Colab** (free tier works, T4 GPU recommended)

---

### Run on Google Colab

This is the **recommended and easiest** way to run this project.

**Step 1** — Open Google Colab: [colab.research.google.com](https://colab.research.google.com)

**Step 2** — Enable GPU:
```
Runtime → Change runtime type → Hardware Accelerator → T4 GPU → Save
```

**Step 3** — Upload or copy the notebook `notebook.ipynb`

**Step 4** — Run all cells top to bottom. When prompted by `notebook_login()`, paste your Hugging Face token.

**Step 5** — After all cells complete, download results:
```python
import shutil
from google.colab import files
shutil.make_archive("sd_results", "zip", "generated_images")
files.download("sd_results.zip")
```

> ⚠️ **First run note:** The model (~4GB) downloads automatically on first run. This takes 3–5 minutes. Subsequent runs use the cached version.

---

**requirements.txt**
```
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
torch>=2.0.0
torchvision
xformers
safetensors
Pillow
matplotlib
huggingface_hub
```

> ⚠️ **Local GPU requirement:** At least **6GB VRAM** is recommended. For CPU-only, change `torch_dtype` to `torch.float32` and expect 5–15 minutes per image.

---

## ⚙️ How It Works

### Step 1 — Install Dependencies

```python
!pip install diffusers transformers accelerate torch torchvision xformers safetensors Pillow matplotlib
```

### Step 2 — Load the Model

### Step 3 — Define Prompt Variants

Three prompt configurations are defined with increasing levels of detail and control. All share the same base subject: `"a lone astronaut on Mars"`.

### Step 4 — Generate & Compare Images

Each prompt variant is passed through the same pipeline with **identical settings** (seed=42, steps=50, guidance=7.5) to ensure only the prompt is the variable being tested.

### Step 5 — Inference Log

All parameters and timing data are saved automatically to `inference_log.json` after each run.

---

## 🎯 Prompt Engineering Strategy

### 1. Basic Prompt

```
a lone astronaut on Mars, ((full body shot)), ((perfectly centered))
```

A minimal prompt with just the subject and basic composition hints. Gives the model maximum creative freedom. Results can be unpredictable — the model may choose unusual angles, distances, or compositions.

---

### 2. Detailed Prompt

```
((a lone astronaut on Mars:1.6)), ((full body shot:1.3)), ((astronaut centered in frame:1.4)),
red dusty terrain, dramatic sunset sky, ultra-realistic, cinematic lighting,
8k resolution, photorealistic, detailed spacesuit, vast landscape,
award-winning photography, golden hour
```

Adds rich descriptive keywords and **weighted tokens** using `(( :weight))` syntax to force the model to prioritize the astronaut as the main subject. Cinematic and quality keywords push the style toward photorealism.

---

### 3. Detailed + Negative Prompt

**Positive:**
```
((a lone astronaut on Mars:1.4)), ((full body shot:1.3)), ((center of frame:1.3)),
red dusty terrain, dramatic sunset sky, ultra-realistic, cinematic lighting, photorealistic
```

**Negative:**
```
wide landscape, extreme wide angle, fisheye, tiny subject, cropped astronaut,
partial body, cut off figure, blurry, low quality, cartoon, anime, painting,
distorted, extra limbs, bad anatomy, watermark, text, deformed,
subject off-center, empty landscape, no person, no astronaut
```

The negative prompt explicitly tells the model what to **avoid** — poor framing, quality issues, and subject absence. This produces the most controlled and refined output of the three.

---

## 📊 Inference Log Results

```
================================================================
📋  INFERENCE LOG
================================================================

🔹 1. Basic Prompt (Improved Composition)
   Prompt          : a lone astronaut on Mars, ((full body shot)), ((perfectly centered))...
   Negative Prompt : wide angle, fisheye, extreme perspective, cropped, cut off...
   Steps           : 50
   Guidance Scale  : 7.5
   Seed            : 42
   Inference Time  : 12.08 seconds
   Saved To        : generated_images/basic_result.png

🔹 2. Detailed Prompt (Astronaut Forced Visible)
   Prompt          : ((a lone astronaut on Mars:1.6)), ((full body shot:1.3))...
   Negative Prompt : empty landscape, no person, no astronaut, tiny subject...
   Steps           : 50
   Guidance Scale  : 7.5
   Seed            : 42
   Inference Time  : 12.86 seconds
   Saved To        : generated_images/detailed_result.png

🔹 3. Detailed + Strong Negative (Highly Controlled)
   Prompt          : ((a lone astronaut on Mars:1.4)), ((full body shot:1.3))...
   Negative Prompt : wide landscape, extreme wide angle, fisheye, tiny subject...
   Steps           : 50
   Guidance Scale  : 7.5
   Seed            : 42
   Inference Time  : 13.59 seconds
   Saved To        : generated_images/negative_result.png
================================================================
```

---

## 🔧 Key Parameters Explained

| Parameter | Value Used | What It Controls |
|---|---|---|
| `num_inference_steps` | 50 | Denoising steps. More = better quality, slower speed |
| `guidance_scale` | 7.5 | How strictly the model follows the prompt (7–12 ideal) |
| `seed` | 42 | Random seed for reproducibility. Same seed = same layout |
| `height / width` | 512 x 512 | Output image resolution |
| `torch_dtype` | float16 | Half precision — faster inference, less VRAM |
| `negative_prompt` | (varies) | Concepts to suppress during generation |

**Guidance Scale Quick Reference:**

| Value | Effect |
|---|---|
| 5 – 7 | More creative, less literal |
| 7.5 | Balanced — recommended default |
| 8 – 12 | Strictly follows the prompt |
| 12+ | Can over-saturate or distort |

**Inference Steps Quick Reference:**

| Steps | Use Case |
|---|---|
| 20 – 30 | Fast preview / drafting |
| 50 | Balanced quality (used here) |
| 75 – 100 | Maximum quality, slow |

---

## 📝 Conclusion & Findings

This project successfully demonstrated the profound impact of **prompt engineering on Stable Diffusion image generation**.

**Key Takeaways:**

**Prompt complexity directly correlates with output quality.** The basic prompt produced a loosely interpreted, unpredictable composition (aerial view, small subject), while the detailed prompt generated a cinematic ground-level perspective with proper subject framing.

**Prompt weighting gives surgical control.** Using `((subject:1.6))` syntax forces the model to prioritize certain concepts during the denoising process, dramatically improving subject visibility and composition.

**Negative prompts are as important as positive ones.** The third variant — which added strong negative prompt suppression — produced the cleanest, most refined result, eliminating cropping, poor quality, and framing issues entirely.

**Inference time scales minimally with prompt complexity.** The difference between the simplest and most complex prompt was only ~1.5 seconds (12.08s → 13.59s), meaning the quality gains from detailed prompting come at virtually no performance cost.

**Prompt engineering is a technical discipline.** This project proves that the same model with the same seed and hyperparameters can produce vastly different outputs based purely on prompt construction — making prompt engineering as important as understanding the model itself.

---

## 🔮 Future Improvements

- [ ] Add a **Gradio web UI** for interactive prompt testing
- [ ] Support **SDXL (Stable Diffusion XL)** for higher resolution outputs
- [ ] Add **ControlNet integration** for pose and depth control
- [ ] Implement **batch seed testing** (auto-run multiple seeds per prompt)
- [ ] Add **CLIP score evaluation** to quantitatively measure prompt-image alignment
- [ ] Add **img2img** pipeline comparison alongside txt2img
- [ ] Export results as a **PDF report** with images + log table


---

> ⭐ If you found this project helpful or interesting, please consider giving it a star on GitHub — it helps others discover it!

---

*Built with ❤️ using Hugging Face Diffusers and Google Colab*
