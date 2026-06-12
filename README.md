---
title: OCR Studio
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
license: mit
short_description: OCR, vision-LLM, and text-LLM cleanup. No paid APIs.
---

<div align="center">

# 📄 OCR Studio

**Free OCR + AI extraction — no paid APIs.**

**🔗 [https://insane-11-ocr-studio.hf.space](https://insane-11-ocr-studio.hf.space)**

[![Live Demo](https://img.shields.io/badge/Try_on_Spaces-📄-brightgreen?style=for-the-badge&logo=huggingface)](https://insane-11-ocr-studio.hf.space)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)]()
[![Gradio 4.44](https://img.shields.io/badge/Gradio-4.44.1-indigo?logo=python&logoColor=white)]()
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-red?logo=tesseract&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Docker](https://img.shields.io/badge/SDK-Docker-2496ED?logo=docker&logoColor=white)]()

</div>

> ⚠️ Free Spaces sleep after inactivity. First visit may take ~30s to wake up.

---

## ✨ Features

### 🔹 Tab 1 — Quick OCR (Tesseract)
| Feature | Detail |
|---------|--------|
| **Input** | Images (PNG, JPG, BMP, TIFF, WebP) & multi-page PDFs |
| **Languages** | 12 supported — EN, HI, ES, FR, DE, PT, IT, ZH, JA, KO, AR, RU |
| **Controls** | DPI slider (150–600) & PSM mode (0–13) for fine-tuning |
| **Output** | Copyable text, live stats (pages/chars/words), `.txt` download |
| **Token?** | ❌ Not required — works fully offline |

### 🔹 Tab 2 — Smart Extract (Vision LLM)
| Feature | Detail |
|---------|--------|
| **Models** | [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) · [Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) |
| **Input** | Images & PDFs (first page only) |
| **Custom prompt** | Optional — replace the default extraction instructions |
| **Output** | Copyable text, stats, `.txt` download |
| **Token?** | ✅ `HF_TOKEN` (free, read scope) |

### 🔹 Tab 3 — Cleanup & Structure (Text LLM)
| Feature | Detail |
|---------|--------|
| **Models** | [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) · [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) · [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| **Cleanup mode** | Fixes OCR typos, normalizes whitespace, preserves paragraphs |
| **Structure mode** | Extracts JSON — supports **Summary**, **Invoice**, **Resume**, **Generic** schemas |
| **Output** | Cleaned text or pretty-printed JSON, stats, download |
| **Token?** | ✅ `HF_TOKEN` (free, read scope) |

### 🧩 Bonus
- 🔗 **Cross-tab linking** — Vision LLM output can be sent directly to Cleanup & Structure
- 📊 **Live stats** — character count, word count, model used, page count
- 📥 **One-click download** — `.txt` or `.json` with auto-detected suffix
- 🎨 **Clean UI** — Gradio Soft theme with indigo accent

---

## 📁 Project Structure

```
ocr-studio/
├── app.py              # Main Gradio app — UI layout, 6 processing functions, monkey-patches
├── requirements.txt    # Python deps (gradio, pytesseract, pdf2image, huggingface-hub, etc.)
├── Dockerfile          # Python 3.12-slim + Tesseract (12 langs) + Poppler
├── README.md           # This file
├── packages.txt        # Tesseract lang packages reference (ignored in Docker mode)
└── .gitignore
```

---

## 🔄 OCR Workflow

```
                      ┌──────────────────────────┐
                      │   Upload File             │
                      │  (image or PDF)           │
                      └────────────┬─────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
        ┌──────────────────┐ ┌──────────┐ ┌──────────────┐
        │  Tab 1           │ │ Tab 2    │ │ Tab 3        │
        │  Quick OCR       │ │ Smart    │ │ Cleanup &    │
        │  (Tesseract)     │ │ Extract  │ │ Structure    │
        │                  │ │ (Vision  │ │ (Text LLM)   │
        │  12 langs        │ │  LLM)    │ │              │
        │  DPI / PSM       │ │          │ │ ┌────────┐  │
        │  Multi-page PDF  │ │ Custom   │ │ │ Clean  │  │
        └────────┬─────────┘ │ prompt   │ │ │ mode   │  │
                 ▼           │          │ │ └────────┘  │
        ┌────────────────┐  │ Page 1   │ │ ┌────────┐  │
        │ Text output    │  │ only     │ │ │Struct. │  │
        │ + stats + .txt │  │          │ │ │ mode   │  │
        └────────────────┘  │ No token │ │ └────────┘  │
                            └────┬─────┘ │ 4 schemas   │
                                 ▼        │            │
                        ┌──────────────┐ │ JSON valid. │
                        │ Text output  │ └──────┬───────┘
                        │ + stats+.txt │        ▼
                        └──────┬───────┘ ┌──────────────┐
                               │ send    │ JSON / text  │
                               └────────►│ + stats+downl│
                                          └──────────────┘
```

> 💡 **Tab 2 → Tab 3 pipeline:** Click "Send to Cleanup & Structure →" to pipe vision LLM output straight into Tab 3 for polishing or JSON extraction.

## 🏗️ Stack

```
Gradio 4.44.1   →  Web interface
Tesseract       →  Offline OCR engine (12 language packs)
pdf2image       →  PDF rasterization (Poppler)
huggingface-hub →  Free Inference API for LLMs
Docker          →  Deployment (Python 3.12-slim)
```

---

## 🖥️ Run Locally

### 1. System dependencies

| OS | Command |
|----|---------|
| **Windows** | Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) + [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) |
| **macOS** | `brew install tesseract poppler` |
| **Linux** | `sudo apt install tesseract-ocr poppler-utils` |

> For 12-language Tesseract support on Linux: `sudo apt install tesseract-ocr-all`

### 2. Python setup

```bash
git clone https://github.com/Insane-11/ocr-studio.git
cd ocr-studio
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` in your browser.

### 3. Token setup (optional — needed for Tabs 2 & 3)

```bash
# Windows (PowerShell)
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# macOS / Linux
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read scope is enough).

---

## ☁️ Deploy to Hugging Face Spaces

One-click deploy — no credit card required.

1. **Create a Space** at [huggingface.co/new-space](https://huggingface.co/new-space) — set SDK to **Docker**, Hardware to **CPU basic**

2. **Push the code:**

   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ocr-studio
   git push space main
   ```

3. **Add your token:** Space → **Settings** → **Variables and secrets** → New secret:
   - Name: `HF_TOKEN`
   - Value: `hf_xxxxxxxxxxxxxxxxxxxx`
   - Visibility: can be hidden

4. **Done!** Visit `https://huggingface.co/spaces/YOUR_USERNAME/ocr-studio`

---

## ⚠️ Notes & Limitations

- **Vision LLM + PDFs** — only page 1 is processed (token cost & inference time)
- **Free Inference API** — has rate limits; suitable for demos, not production traffic
- **Privacy** — all processing runs on HF Spaces servers; uploaded files are not stored
- **Cold start** — free Spaces sleep after ~48h of inactivity; first visit wakes it (~30s)
- **Compute hours** — free tier has a monthly cap; if exceeded, the Space sleeps until next month

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.
