import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
import os
import io
from typing import List, Dict, Any
import pdfplumber

def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Return list of dicts: {page:int, df:DataFrame} using pdfplumber (keeps your previous behavior)."""
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for p_idx, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for t in tables or []:
                df = pd.DataFrame(t)
                if not df.empty:
                    if df.shape[0] > 1 and df.iloc[0].isna().sum() < len(df.columns) // 2:
                        df.columns = df.iloc[0].astype(str)
                        df = df[1:].reset_index(drop=True)
                out.append({"page": p_idx, "df": df})
    return out

def _preprocess_for_table(img):
    """Grayscale + adaptive threshold + invert for better line detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # increase resolution for small images
    h, w = gray.shape
    scale = max(1, int(1000.0 / max(h, w)))
    if scale > 1:
        gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 9)
    inv = 255 - th
    return inv

def extract_tables_from_image(img_path: str) -> List[Dict[str, Any]]:
    """
    Detect table regions and extract cell texts.
    Returns list of {"page":1, "df": pd.DataFrame} (page kept for compatibility).
    If robust cell detection fails, returns a single entry with OCR text (as one-column table).
    """
    img = cv2.imread(img_path)
    if img is None:
        return []

    processed = _preprocess_for_table(img)
    h, w = processed.shape

    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//30), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//30)))

    horizontal_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(processed, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    table_mask = cv2.add(horizontal_lines, vertical_lines)

    # find contours on table mask
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_areas = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)

        if ww < w * 0.2 or hh < h * 0.05:
            continue
        table_areas.append((x, y, ww, hh))

    results = []
    if not table_areas:
        # fallback: OCR entire image and return as single text block
        pil = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(pil)
        df = pd.DataFrame({"text":[text]})
        results.append({"page": 1, "df": df})
        return results

    # For detected table region
    for (x, y, ww, hh) in table_areas:
        table_roi = processed[y:y+hh, x:x+ww]
        # get horizontal and vertical lines
        hor = cv2.morphologyEx(table_roi, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        ver = cv2.morphologyEx(table_roi, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        inter = cv2.bitwise_and(hor, ver)
        # find intersection points
        points = cv2.findNonZero(inter)
        # if not enough intersections, fallback to OCR text chunk forr  ROI
        pil_roi = Image.fromarray(cv2.cvtColor(img[y:y+hh, x:x+ww], cv2.COLOR_BGR2RGB))
        if points is None or len(points) < 4:
            text = pytesseract.image_to_string(pil_roi)
            df = pd.DataFrame({"text":[text]})
            results.append({"page":1, "df":df})
            continue

        # build grid by clustering unique x and y from intersections
        pts = [(int(p[0][0]), int(p[0][1])) for p in points]
        xs = sorted(set([px for px,py in pts]))
        ys = sorted(set([py for px,py in pts]))

        # cluster xs/ys 
        def cluster_coords(coords, thresh=10):
            clusters = []
            coords = sorted(coords)
            cur = [coords[0]]
            for v in coords[1:]:
                if v - cur[-1] <= thresh:
                    cur.append(v)
                else:
                    clusters.append(int(sum(cur)/len(cur)))
                    cur = [v]
            clusters.append(int(sum(cur)/len(cur)))
            return clusters

        col_lines = cluster_coords(xs, thresh=max(3, ww//100))
        row_lines = cluster_coords(ys, thresh=max(3, hh//100))

        # use grid to crop cells
        cells = []
        for r in range(len(row_lines)-1):
            row = []
            for c in range(len(col_lines)-1):
                x1 = x + col_lines[c]
                x2 = x + col_lines[c+1]
                y1 = y + row_lines[r]
                y2 = y + row_lines[r+1]
                if x2 <= x1 or y2 <= y1:
                    row.append("")
                    continue
                crop = img[y1:y2, x1:x2]
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                cell_text = pytesseract.image_to_string(pil_crop).strip()
                row.append(cell_text)
            cells.append(row)

        # convert to DataFrame
        try:
            df = pd.DataFrame(cells)
            # if first row contains many non-empty texts, set as header
            if df.shape[0] > 1 and df.iloc[0].count() >= max(1, df.shape[1]//2):
                df.columns = df.iloc[0].fillna("").astype(str)
                df = df[1:].reset_index(drop=True)
        except Exception:
            df = pd.DataFrame({"text":[pytesseract.image_to_string(pil_roi)]})
        results.append({"page": 1, "df": df})

    return results
