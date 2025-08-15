import os
import io
import json
import math
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import List, Dict, Any


from table_extractor import extract_tables_from_image, extract_tables_from_pdf
from chart_extractor import extract_images_with_ocr, analyze_chart

# ensure output dir exists
OUT_DIR = "extracted_imgs"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------- Helpers: OCR + preprocessing -----------------
def _preprocess_for_ocr_pil(pil_img: Image.Image, target_min_dim: int = 1000) -> Image.Image:
    """Basic preprocessing for OCR: resize small images, denoise, binarize."""
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = 1
    if max(h, w) < target_min_dim:
        scale = int(math.ceil(target_min_dim / float(max(h, w))))
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    # denoise and threshold
    gray = cv2.medianBlur(gray, 3)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(th)


def ocr_image_pil(pil_img: Image.Image, lang: str = "eng") -> str:
    """Run pytesseract OCR with some default config tuned for documents."""
    try:
        prep = _preprocess_for_ocr_pil(pil_img)
     
        text = pytesseract.image_to_string(prep, lang=lang, config="--psm 6")
        return text.strip()
    except Exception:
        # fallback to raw OCR
        return pytesseract.image_to_string(pil_img).strip()


# ----------------- Helpers: layout / figure detection -----------------
def detect_figures_on_image(pil_img: Image.Image, min_area_ratio: float = 0.02) -> List[Dict[str, Any]]:
    """
    Detect large visual blocks (figures) on a page image using morphology.
    Returns list of {"bbox":(x,y,w,h), "crop":PIL.Image}
    """
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th

    # close gaps to form solid blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 40), max(8, h // 40)))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    figs = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < (w * h * min_area_ratio):
            continue
        # skip full-page capture
        if ww > 0.9 * w and hh > 0.9 * h:
            continue
        crop = Image.fromarray(cv2.cvtColor(img[y:y + hh, x:x + ww], cv2.COLOR_BGR2RGB))
        figs.append({"bbox": (x, y, ww, hh), "crop": crop})
    return figs


def correlate_figure_with_text(page_pil: Image.Image, fig_bbox: tuple, max_lines: int = 6) -> str:
    """
    Attempt to find caption text near a figure bbox using pytesseract's word boxes.
    Searches below and above the figure within a reasonable vertical window.
    Returns joined caption string (may be empty).
    """
    img_arr = np.array(page_pil.convert("RGB"))
    o = pytesseract.image_to_data(Image.fromarray(img_arr), output_type=pytesseract.Output.DICT)
    x, y, w, h = fig_bbox

    candidates = []
    n = len(o["text"])
    for i in range(n):
        txt = o["text"][i].strip()
        if not txt:
            continue
        lx = o["left"][i]
        ly = o["top"][i]
        lh = o["height"][i]
        # below figure (within 50% of figure height) or above figure
        if (ly > y + h and ly < y + h + max(50, int(0.5 * h))) or (ly + lh < y and ly + lh > y - max(50, int(0.5 * h))):
            candidates.append((ly, txt))
    candidates = sorted(candidates, key=lambda x: x[0])
    joined = " ".join([t for _, t in candidates[:max_lines]])
    return joined.strip()


# ----------------- Helpers: safe metadata for ChromaDB -----------------
def _make_safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure metadata values are simple types accepted by ChromaDB.
    Complex lists/dicts are JSON-stringified.
    """
    safe = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe[k] = v
        else:
            try:
                safe[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                safe[k] = str(v)
    return safe


# ----------------- Main extraction function -----------------
def extract_text_chunks(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text/table/chart/layout/image-correlation chunks from a file (PDF or image).
    Returns: list of {"text": <str>, "meta": <dict with simple values>}
    """
    ext = os.path.splitext(file_path)[1].lower()
    chunks: List[Dict[str, Any]] = []

    # ---------------- Image (PNG/JPG) ----------------
    if ext in [".png", ".jpg", ".jpeg"]:
        pil = Image.open(file_path).convert("RGB")

        # 1) OCR whole image 
        ocr_text = ocr_image_pil(pil)
        if ocr_text:
            for piece in (split_for_rag := lambda t, m=1200, o=100: [t[i:i+m] for i in range(0, max(1, len(t)), m - o)] )(ocr_text):
                chunks.append({
                    "text": piece,
                    "meta": _make_safe_meta({"type": "text", "page": 1})
                })

        # 2) Table extraction from image
        try:
            img_tables = extract_tables_from_image(file_path)
            for t in img_tables:
                md = t["df"].to_markdown(index=False)
                chunks.append({
                    "text": md,
                    "meta": _make_safe_meta({"type": "table", "page": t.get("page", 1)})
                })
        except Exception as e:
            # safe fallback
            print(f"[process] image table extraction failed: {e}")

        # 3) Chart extraction from image
        try:
            chart_items = extract_images_with_ocr(file_path, out_dir=OUT_DIR)
        
            for c in chart_items:
                
                summary_lines = []
                if c.get("ocr_text"):
                    summary_lines.append(c["ocr_text"].strip())
                if c.get("chart_type"):
                    summary_lines.append(f"chart_type: {c.get('chart_type')}")
                if c.get("data"):
                    summary_lines.append(f"chart_data: {json.dumps(c.get('data'), ensure_ascii=False)}")
                summary = "\n".join(summary_lines).strip() or f"Chart on page {c.get('page',1)}"

                meta = {
                    "type": "chart",
                    "page": c.get("page", 1),
                    "image_path": c.get("image_path"),
                    "chart_type": c.get("chart_type", "unknown"),
                    # store small flag and JSON string for complex data
                    "has_data": bool(c.get("data")),
                    "data_json": json.dumps(c.get("data") or [])
                }

                chunks.append({
                    "text": summary,
                    "meta": _make_safe_meta(meta)
                })
        except Exception as e:
            print(f"[process] chart extraction failed: {e}")

        # 4) Layout analysis:  visual figures and correlate nearby text
        try:
            figs = detect_figures_on_image(pil)
            for idx, fig in enumerate(figs, start=1):
                caption = correlate_figure_with_text(pil, fig["bbox"])
                save_name = f"img_page1_fig{idx}.png"
                save_path = os.path.join(OUT_DIR, save_name)
                fig["crop"].save(save_path)
                meta = {"type": "image_correlation", "page": 1, "image_path": save_path, "context": caption or ""}
                chunks.append({
                    "text": f"Figure (image) on page 1 — context: {caption or '(no caption)'}",
                    "meta": _make_safe_meta(meta)
                })
        except Exception as e:
            print(f"[process] figure detection failed: {e}")

    # ---------------- PDF ----------------
    elif ext == ".pdf":
        # open once
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise RuntimeError(f"Could not open PDF: {e}")

        # 1) Native text per page (if any) -> chunked
        for pno, page in enumerate(doc, start=1):
            try:
                txt = page.get_text().strip()
            except Exception:
                txt = ""
            # fallback to OCR if no native text
            if not txt:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    pil_page = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
                    txt = ocr_image_pil(pil_page)
                except Exception:
                    txt = ""
            if txt:
                # split into rag chunks
                maxc, overlap = 1200, 100
                i = 0
                n = len(txt)
                while i < n:
                    piece = txt[i:i + maxc]
                    chunks.append({
                        "text": piece,
                        "meta": _make_safe_meta({"type": "text", "page": pno})
                    })
                    i += (maxc - overlap)

        # 2) PDF tables via extract_tables_from_pdf (
        try:
            pdf_tables = extract_tables_from_pdf(file_path)
        except Exception as e:
            print(f"[process] extract_tables_from_pdf failed: {e}")
            pdf_tables = []
        for t in pdf_tables:
            md = t["df"].to_markdown(index=False)
            chunks.append({
                "text": md,
                "meta": _make_safe_meta({"type": "table", "page": t.get("page", 1)})
            })

        # 3) Also render pages and run image-table detection 
        for pno, page in enumerate(doc, start=1):
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                pil_page = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
                tmp_path = os.path.join(OUT_DIR, f"temp_page_{pno}.png")
                pil_page.save(tmp_path)

                # image-based table detection on rendered page
                try:
                    image_tables = extract_tables_from_image(tmp_path)
                    for t in image_tables:
                        md = t["df"].to_markdown(index=False)
                        chunks.append({
                            "text": md,
                            "meta": _make_safe_meta({"type": "table", "page": pno})
                        })
                except Exception as e:
                    print(f"[process] image table extraction on page {pno} failed: {e}")

                # 4) Chart extraction from the page image(s)
                try:
                    chart_items = extract_images_with_ocr(tmp_path, out_dir=OUT_DIR)
                    for c in chart_items:
                        meta = {
                            "type": "chart",
                            "page": c.get("page", pno),
                            "image_path": c.get("image_path"),
                            "chart_type": c.get("chart_type", "unknown"),
                            "has_data": bool(c.get("data")),
                            "data_json": json.dumps(c.get("data") or [])
                        }
                        # build readable text summary
                        summary_parts = []
                        if c.get("ocr_text"):
                            summary_parts.append(c["ocr_text"].strip())
                        if c.get("chart_type"):
                            summary_parts.append(f"chart_type: {c.get('chart_type')}")
                        if c.get("data"):
                            summary_parts.append(f"chart_data: {json.dumps(c.get('data'), ensure_ascii=False)}")
                        summary = "\n".join(summary_parts).strip() or f"Chart on page {meta['page']}"
                        chunks.append({"text": summary, "meta": _make_safe_meta(meta)})
                except Exception as e:
                    print(f"[process] chart extraction on page {pno} failed: {e}")

                # 5) Layout / figure detection and correlation
                try:
                    figs = detect_figures_on_image(pil_page)
                    for idx, fig in enumerate(figs, start=1):
                        caption = correlate_figure_with_text(pil_page, fig["bbox"])
                        save_name = f"page{pno}_fig{idx}.png"
                        save_path = os.path.join(OUT_DIR, save_name)
                        fig["crop"].save(save_path)
                        meta = {"type": "image_correlation", "page": pno, "image_path": save_path, "context": caption or ""}
                        chunks.append({
                            "text": f"Figure on page {pno} — context: {caption or '(no caption)'}",
                            "meta": _make_safe_meta(meta)
                        })
                except Exception as e:
                    print(f"[process] figure detection on page {pno} failed: {e}")

                # cleanup tmp image
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

            except Exception as e:
                print(f"[process] render page {pno} failed: {e}")

    else:
        raise ValueError("Unsupported file type. Please upload PDF, PNG, or JPG.")

    return chunks

