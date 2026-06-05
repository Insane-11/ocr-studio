import os
import io
import re
import json
import base64
import tempfile
from pathlib import Path

import gradio as gr
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from huggingface_hub import InferenceClient


HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

VISION_MODELS = {
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Llama-3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}
TEXT_MODELS = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
DEFAULT_VISION = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_TEXT = "Qwen/Qwen2.5-7B-Instruct"

LANG_MAP = {
    "English": "eng",
    "Hindi": "hin",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Portuguese": "por",
    "Italian": "ita",
    "Chinese (Simplified)": "chi_sim",
    "Japanese": "jpn",
    "Korean": "kor",
    "Arabic": "ara",
    "Russian": "rus",
}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXT = {".pdf"}


def client():
    return InferenceClient(token=HF_TOKEN or None)


def file_kind(path):
    ext = Path(path).suffix.lower()
    if ext in IMG_EXT:
        return "image"
    if ext in PDF_EXT:
        return "pdf"
    return "unknown"


def image_b64(path):
    ext = Path(path).suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext if ext != 'jpg' else 'jpeg'}"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def pil_b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def write_temp(text, suffix=".txt"):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return tmp.name


def stats_block(text, pages=None, model=None, extra=""):
    parts = []
    if pages is not None:
        parts.append(f"**Pages:** {pages}")
    parts.append(f"**Characters:** {len(text):,}")
    parts.append(f"**Words:** {len(text.split()):,}")
    if model:
        parts.append(f"**Model:** `{model}`")
    if extra:
        parts.append(extra)
    return "  \n".join(parts)


def tesseract_run(file_obj, language, dpi, psm, progress=gr.Progress()):
    if file_obj is None:
        return "", "**Status:** Upload a file first.", None
    path = str(file_obj)
    kind = file_kind(path)
    if kind == "unknown":
        return "", f"**Status:** Unsupported file type.", None
    lang = LANG_MAP.get(language, "eng")
    progress(0.05, desc="Reading file...")
    pages_text = []
    try:
        if kind == "pdf":
            progress(0.1, desc="Rasterizing PDF...")
            images = convert_from_path(path, dpi=int(dpi))
            total = len(images)
            for i, img in enumerate(images, 1):
                progress(0.1 + 0.85 * (i - 1) / max(total, 1), desc=f"OCR page {i}/{total}")
                chunk = pytesseract.image_to_string(img, lang=lang, config=f"--psm {int(psm)}").strip()
                pages_text.append((i, chunk))
        else:
            img = Image.open(path)
            progress(0.5, desc="OCR...")
            chunk = pytesseract.image_to_string(img, lang=lang, config=f"--psm {int(psm)}").strip()
            pages_text.append((1, chunk))
    except pytesseract.TesseractNotFoundError:
        return "", "**Status:** Tesseract binary not found. Install it locally or deploy to HF Spaces.", None
    except Exception as e:
        return "", f"**Status:** Error — `{e}`", None

    body = "\n\n".join(
        f"--- Page {i} ---\n{t}" for i, t in pages_text if t
    ).strip()

    if not body:
        return "", "**Status:** No text detected. Try a higher DPI or a different language.", None

    progress(0.98, desc="Writing file...")
    return body, stats_block(body, pages=len(pages_text), model="Tesseract"), write_temp(body)


def vision_run(file_obj, model_label, custom_prompt, progress=gr.Progress()):
    if file_obj is None:
        return "", "**Status:** Upload a file first.", None
    if not HF_TOKEN:
        return (
            "",
            "**Status:** `HF_TOKEN` is not set. Add it as a Space secret in **Settings → Variables**.",
            None,
        )

    path = str(file_obj)
    kind = file_kind(path)
    if kind == "unknown":
        return "", "**Status:** Unsupported file type.", None

    model = VISION_MODELS.get(model_label, DEFAULT_VISION)
    progress(0.15, desc="Encoding image...")
    try:
        if kind == "pdf":
            images = convert_from_path(path, dpi=200, first_page=1, last_page=1)
            if not images:
                return "", "**Status:** PDF has no pages.", None
            data_url = pil_b64(images[0])
        else:
            data_url = image_b64(path)
    except Exception as e:
        return "", f"**Status:** Image prep failed — `{e}`", None

    prompt = (custom_prompt or "").strip() or (
        "Extract all text from this image verbatim. Preserve the original layout, line breaks, "
        "and structure. Output only the extracted text, no commentary."
    )

    progress(0.35, desc=f"Calling {model}...")
    try:
        completion = client().chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=2048,
            temperature=0.0,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return "", f"**Status:** Inference failed — `{e}`", None

    if not text:
        return "", "**Status:** Empty response from model.", None

    progress(0.95, desc="Done")
    suffix = "_vision.txt" if kind == "pdf" else "_vision.txt"
    return text, stats_block(text, model=model, extra="(first page only for PDFs)"), write_temp(text, suffix)


def llm_clean(raw_text, model_label, custom_instr, progress=gr.Progress()):
    if not raw_text or not raw_text.strip():
        return "", "**Status:** Paste some OCR text first.", None
    if not HF_TOKEN:
        return "", "**Status:** `HF_TOKEN` is not set.", None

    model = TEXT_MODELS.get(model_label, DEFAULT_TEXT)
    instr = (custom_instr or "").strip() or (
        "Clean this OCR output: fix obvious character errors, normalize whitespace, "
        "preserve paragraph breaks, do not summarize or rephrase. Return only the cleaned text."
    )
    snippet = raw_text[:7000]

    progress(0.25, desc=f"Calling {model}...")
    try:
        completion = client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You clean noisy OCR text. Be faithful to the original."},
                {"role": "user", "content": f"{instr}\n\n---\n{snippet}"},
            ],
            max_tokens=2500,
            temperature=0.1,
        )
        cleaned = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return "", f"**Status:** Inference failed — `{e}`", None

    if not cleaned:
        return "", "**Status:** Empty response.", None

    progress(0.95, desc="Done")
    return (
        cleaned,
        stats_block(cleaned, model=model, extra=f"In: {len(raw_text):,} chars"),
        write_temp(cleaned, "_clean.txt"),
    )


STRUCTURE_SCHEMAS = {
    "Summary + Key Points": (
        "Return a JSON object with: title (string|null), summary (string, 2-3 sentences), "
        "key_points (array of 3-7 short strings), entities (object with arrays: people, "
        "organizations, locations, dates)."
    ),
    "Invoice / Receipt": (
        "Return a JSON object with: vendor (string), date (string), total (string), "
        "currency (string|null), line_items (array of {description, quantity, price}), "
        "notes (string)."
    ),
    "Resume / CV": (
        "Return a JSON object with: name (string), email (string|null), phone (string|null), "
        "location (string|null), summary (string), skills (array of strings), "
        "experience (array of {company, role, dates, description}), "
        "education (array of {institution, degree, dates})."
    ),
    "Generic JSON": (
        "Return a JSON object that best captures the structure of this text. Always include "
        "a `raw_summary` field with a one-paragraph summary. Use null for missing fields."
    ),
}


def extract_json_block(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def llm_structure(raw_text, schema_label, model_label, progress=gr.Progress()):
    if not raw_text or not raw_text.strip():
        return "", "**Status:** Paste text first.", None
    if not HF_TOKEN:
        return "", "**Status:** `HF_TOKEN` is not set.", None

    model = TEXT_MODELS.get(model_label, DEFAULT_TEXT)
    schema_instr = STRUCTURE_SCHEMAS.get(schema_label, STRUCTURE_SCHEMAS["Summary + Key Points"])
    snippet = raw_text[:6500]

    progress(0.25, desc=f"Extracting structure with {model}...")
    try:
        completion = client().chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured data from text. Respond with valid JSON only — no markdown fences, no commentary.",
                },
                {
                    "role": "user",
                    "content": f"{schema_instr}\n\n---\n{snippet}",
                },
            ],
            max_tokens=1500,
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return "", f"**Status:** Inference failed — `{e}`", None

    candidate = extract_json_block(raw)
    progress(0.9, desc="Validating JSON...")
    try:
        parsed = json.loads(candidate)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        ok = True
    except json.JSONDecodeError as e:
        pretty = raw
        ok = False

    status = stats_block(pretty, model=model, extra=f"Schema: {schema_label} | JSON valid: {ok}")
    suffix = "_structured.json" if ok else "_structured.txt"
    return pretty, status, write_temp(pretty, suffix)


CUSTOM_CSS = """
.footer { text-align: center; opacity: 0.7; margin-top: 1rem; font-size: 0.9rem; }
.token-banner { padding: 0.75rem 1rem; border-radius: 0.5rem; background: #fff5e0; border: 1px solid #f0c674; }
.token-banner.ok { background: #e8f5e9; border-color: #81c784; }
"""

with gr.Blocks(title="OCR Studio", theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS) as demo:
    gr.Markdown("# OCR Studio")
    gr.Markdown(
        "Free OCR + AI extraction. Three modes: fast Tesseract, vision-LLM for tricky layouts, "
        "and LLM cleanup & structured output. No paid APIs."
    )

    token_banner = gr.Markdown(
        visible=bool(HF_TOKEN),
        elem_classes="token-banner ok" if HF_TOKEN else "token-banner",
    )
    if HF_TOKEN:
        token_banner.value = "**HF_TOKEN detected** — Vision LLM and Text LLM tabs are enabled."
    else:
        token_banner.value = (
            "**HF_TOKEN not set** — Tab 1 (Tesseract) works without it. "
            "Add `HF_TOKEN` in **Space Settings → Variables and secrets** to enable Tabs 2 and 3."
        )

    with gr.Tabs():
        with gr.Tab("Quick OCR · Tesseract"):
            with gr.Row():
                with gr.Column(scale=1):
                    f1 = gr.File(
                        label="Upload image or PDF",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"],
                        type="filepath",
                    )
                    lang1 = gr.Dropdown(label="Language", choices=list(LANG_MAP.keys()), value="English")
                    with gr.Row():
                        dpi1 = gr.Slider(label="PDF DPI", minimum=150, maximum=600, step=50, value=300)
                        psm1 = gr.Slider(label="PSM", minimum=0, maximum=13, step=1, value=3)
                    with gr.Row():
                        run1 = gr.Button("Extract", variant="primary", scale=2)
                        clr1 = gr.Button("Clear", scale=1)
                with gr.Column(scale=2):
                    out1 = gr.Textbox(label="Extracted text", lines=20, show_copy_button=True, autoscroll=True)
                    stat1 = gr.Markdown("**Status:** Ready")
                    dl1 = gr.File(label="Download .txt", interactive=False)

            run1.click(tesseract_run, [f1, lang1, dpi1, psm1], [out1, stat1, dl1])
            clr1.click(lambda: ("", "**Status:** Ready", None, None), outputs=[out1, stat1, f1, dl1])

        with gr.Tab("Smart Extract · Vision LLM"):
            with gr.Row():
                with gr.Column(scale=1):
                    f2 = gr.File(
                        label="Upload image or PDF (first page used)",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"],
                        type="filepath",
                    )
                    mdl2 = gr.Dropdown(
                        label="Vision model",
                        choices=list(VISION_MODELS.keys()),
                        value=list(VISION_MODELS.keys())[0],
                    )
                    pr2 = gr.Textbox(
                        label="Custom prompt (optional)",
                        lines=3,
                        placeholder="Leave empty for default verbatim extraction.",
                    )
                    with gr.Row():
                        run2 = gr.Button("Extract", variant="primary", scale=2)
                        clr2 = gr.Button("Clear", scale=1)
                with gr.Column(scale=2):
                    out2 = gr.Textbox(label="Extracted text", lines=20, show_copy_button=True, autoscroll=True)
                    stat2 = gr.Markdown("**Status:** Ready")
                    dl2 = gr.File(label="Download .txt", interactive=False)
                    send2 = gr.Button("Send to Cleanup & Structure →")

            run2.click(vision_run, [f2, mdl2, pr2], [out2, stat2, dl2])
            clr2.click(lambda: ("", "**Status:** Ready", None, None, None), outputs=[out2, stat2, f2, dl2])

            send2.click(lambda txt: txt or "", inputs=[out2], outputs=[t_in3])

        with gr.Tab("Cleanup & Structure · Text LLM"):
            with gr.Row():
                with gr.Column(scale=1):
                    t_in3 = gr.Textbox(
                        label="Input text (paste or send from another tab)",
                        lines=14,
                        placeholder="Paste OCR text here...",
                    )
                    mdl3 = gr.Dropdown(
                        label="Text model",
                        choices=list(TEXT_MODELS.keys()),
                        value=list(TEXT_MODELS.keys())[0],
                    )
                    schema3 = gr.Dropdown(
                        label="Schema (for Structure mode)",
                        choices=list(STRUCTURE_SCHEMAS.keys()),
                        value="Summary + Key Points",
                    )
                    instr3 = gr.Textbox(
                        label="Custom instruction (optional, for Cleanup mode)",
                        lines=2,
                    )
                    with gr.Row():
                        run3a = gr.Button("Clean up", variant="primary")
                        run3b = gr.Button("Extract structure", variant="primary")
                        clr3 = gr.Button("Clear", scale=1)
                with gr.Column(scale=2):
                    out3 = gr.Textbox(label="Output", lines=22, show_copy_button=True, autoscroll=True)
                    stat3 = gr.Markdown("**Status:** Ready")
                    dl3 = gr.File(label="Download", interactive=False)

            run3a.click(llm_clean, [t_in3, mdl3, instr3], [out3, stat3, dl3])
            run3b.click(llm_structure, [t_in3, schema3, mdl3], [out3, stat3, dl3])
            clr3.click(lambda: ("", "**Status:** Ready", "", None), outputs=[t_in3, stat3, out3, dl3])

    gr.Markdown(
        "<div class='footer'>Gradio · Tesseract · Hugging Face Inference API · "
        "Deployable free on Hugging Face Spaces</div>"
    )


if __name__ == "__main__":
    demo.launch()
