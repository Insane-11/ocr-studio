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
---

# OCR Studio

Free OCR + AI extraction in a single Gradio app. Three modes: Tesseract, vision-LLM, and LLM cleanup + structured output. No paid APIs.

## Features

- **Quick OCR** — Tesseract for images and multi-page PDFs, 12 languages, adjustable DPI/PSM
- **Smart Extract** — Vision LLM (Qwen2-VL / Llama-3.2-Vision) via free Hugging Face Inference API
- **Cleanup & Structure** — Text LLM (Qwen2.5 / Mistral / Llama) for OCR cleanup and JSON extraction (summary, invoice, resume, generic)
- One-click `.txt` / `.json` download, live stats, copy buttons, clean Soft-themed UI

## Stack

Gradio · Tesseract (`pytesseract`) · `pdf2image` (Poppler) · `huggingface-hub` InferenceClient

## Run locally

System deps (Tesseract + Poppler) — see `https://github.com/UB-Mannheim/tesseract/wiki` (Windows) or `brew install tesseract poppler` (macOS) or `apt install tesseract-ocr poppler-utils` (Linux).

```bash
git clone https://github.com/<you>/ocr-studio.git
cd ocr-studio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Tab 1 works without any token. For Tabs 2 and 3, set `HF_TOKEN` in your shell:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Get a free token at `https://huggingface.co/settings/tokens` (Read scope is enough).

## Deploy to Hugging Face Spaces (free)

1. Create a Space at `https://huggingface.co/new-space` — SDK: **Docker**, Hardware: **CPU basic**
2. Push:

```bash
git remote add space https://huggingface.co/spaces/<you>/ocr-studio
git push space main
```

3. Add your HF token: Space → **Settings** → **Variables and secrets** → New secret:
   - Name: `HF_TOKEN`
   - Value: `hf_xxxxxxxxxxxxxxxxxxxx`
   - Visibility: can be hidden

4. Live URL: `https://huggingface.co/spaces/<you>/ocr-studio`

## Notes

- Vision LLM tab uses page 1 of PDFs only (token cost + inference time).
- Free Inference API has rate limits — fine for demos, not for production traffic.
- All processing is server-side on HF Spaces; uploaded files are not stored.

## License

MIT
