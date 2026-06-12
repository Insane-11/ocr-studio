---
sdk: docker
---

# OCR Studio

Extract text from images and PDFs using Tesseract, vision-LLMs, or text-LLM cleanup and structured JSON.

| | |
|---|---|
| **Live app** | [insane-11-ocr-studio.hf.space](https://insane-11-ocr-studio.hf.space) |
| **SDK** | Docker (Python 3.12, Gradio 4.44, Tesseract 12 langs, Poppler) |
| **License** | MIT |

## Tabs

- **Quick OCR** — Tesseract OCR for images and multi-page PDFs. 12 languages, adjustable DPI/PSM. No token needed.
- **Smart Extract** — Vision LLM (Qwen2-VL / Llama-3.2-Vision) via Hugging Face Inference API. Images and PDFs (page 1). Token required.
- **Cleanup & Structure** — Text LLM (Qwen2.5 / Mistral / Llama) for OCR cleanup or JSON extraction (summary, invoice, resume, generic). Token required.

## Run Locally

**System deps:** Tesseract + Poppler ([Windows](https://github.com/UB-Mannheim/tesseract/wiki) / `brew install tesseract poppler` / `sudo apt install tesseract-ocr poppler-utils`)

```bash
git clone https://github.com/Insane-11/ocr-studio.git
cd ocr-studio
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
python app.py
```

Set `HF_TOKEN` for Tabs 2 & 3 ([free token](https://huggingface.co/settings/tokens), read scope).

## Deploy to HF Spaces

1. Create a [new Space](https://huggingface.co/new-space) — SDK: **Docker**, Hardware: **CPU basic**
2. Push: `git remote add space https://huggingface.co/spaces/YOU/ocr-studio && git push space main`
3. Add `HF_TOKEN` secret in Space settings

## License

MIT
