import fitz
import io, os, cv2, numpy as np, re
from PIL import Image
import pytesseract
from typing import List, Dict, Any

def _preproc_for_chart(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

def _detect_bar_chart_data(img_path: str) -> Dict[str, Any]:
    """
    Simple bar-chart digitizer:
    - Finds vertical bars by morphology
    - Sorts bars left->right
    - Reads labels from bottom strip via OCR
    - Uses left strip numbers (y-axis) to rescale if found
    """
    img = cv2.imread(img_path)
    if img is None:
        return {"chart_type": "unknown", "data": [], "ocr_text": ""}

    gray = _preproc_for_chart(img)
    h_img, w_img = gray.shape

    
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(10, h_img//50)))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < max(3, w_img*0.01) or h < h_img*0.15:
            continue
        aspect = h / (w + 1e-9)
        if aspect < 1.2:
            continue
        bars.append((x, y, w, h))

    if not bars:
        return {"chart_type": "unknown", "data": [], "ocr_text": ""}

    bars = sorted(bars, key=lambda b: b[0])  # left->right

    # OCR the whole image and bottom strip
    pil = Image.open(img_path).convert("RGB")
    ocr_text = pytesseract.image_to_string(pil)

    # try to read y-axis numbers on left strip
    left_strip = pil.crop((0, 0, int(w_img*0.15), h_img))
    left_txt = pytesseract.image_to_string(left_strip)
    y_numbers = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', left_txt)] if left_txt else []

    # bottom labels
    bottom_strip = pil.crop((0, int(h_img*0.80), w_img, h_img))
    label_text = pytesseract.image_to_string(bottom_strip)
    labels = re.findall(r'[A-Za-z0-9%&\- ]{1,}', label_text)
    labels = [l.strip() for l in labels if l.strip()]

    heights = [b[3] for b in bars]
    max_h = max(heights) if heights else 1
    raw_vals = [round(h/max_h*100, 1) for h in heights]

    # if y_numbers present try to rescale - assume top number maps to max_h
    if y_numbers:
        try:
            top_num = max(y_numbers)
            vals = [round(h/max_h*top_num, 2) for h in heights]
        except Exception:
            vals = raw_vals
    else:
        vals = raw_vals

    # align labels
    if len(labels) >= len(bars):
        labels = labels[:len(bars)]
    else:
        labels += [f"Bar {i+1}" for i in range(len(labels), len(bars))]

    data = [{"label": L, "value": V} for L, V in zip(labels, vals)]

    return {"chart_type": "bar", "data": data, "ocr_text": ocr_text}

def extract_images_with_ocr(file_path: str, out_dir: str = "extracted_imgs") -> List[Dict[str, Any]]:
    """
    Extract images from PDF or use single image; run OCR and chart parse on each image.
    Returns a list of dicts:
      {"page": int, "image_path": str, "ocr_text": str, "chart_type": str, "data": list}
    """
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    ext = os.path.splitext(file_path)[1].lower()

    def _append_result(page_no: int, pil_img: Image.Image, save_name: str):
        save_path = os.path.join(out_dir, save_name)
        pil_img.save(save_path)
        ocr_text = pytesseract.image_to_string(pil_img).strip()
        parsed = _detect_bar_chart_data(save_path)
        entry = {
            "page": page_no,
            "image_path": save_path,
            "ocr_text": ocr_text,
            "caption": (ocr_text[:120] if ocr_text else ""),
            "chart_type": parsed.get("chart_type", "unknown"),
            "data": parsed.get("data", []),
        }
        results.append(entry)

    if ext in [".png", ".jpg", ".jpeg"]:
        pil = Image.open(file_path).convert("RGB")
        _append_result(1, pil, os.path.basename(file_path))

    elif ext == ".pdf":
        doc = fitz.open(file_path)
        for pno in range(len(doc)):
            page = doc[pno]
            # extract embedded images from PDF page
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pil = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
                _append_result(pno + 1, pil, f"page{pno+1}_img{img_index}.png")
            # image for charts embedded 
            page_img = page.get_pixmap(matrix=fitz.Matrix(2,2))
            pil_page = Image.open(io.BytesIO(page_img.tobytes())).convert("RGB")
            # save page render too
            _append_result(pno + 1, pil_page, f"page{pno+1}_full.png")
    else:
        raise ValueError("Unsupported file type. Use PDF, PNG, or JPG.")
    return results

def analyze_chart(file_path: str) -> List[Dict[str,Any]]:
    """Compatibility wrapper: returns only charts (non-'unknown')."""
    try:
        results = extract_images_with_ocr(file_path)
        charts = [r for r in results if r.get("chart_type") != "unknown"]
        return charts
    except Exception as e:
        print(f"[chart_extractor] error: {e}")
        return []
