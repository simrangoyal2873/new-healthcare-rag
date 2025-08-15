# image_table_extractor.py
from typing import List, Dict, Any
import pandas as pd
from paddleocr import PPStructure
from PIL import Image

def extract_tables_from_image(img_path: str) -> List[Dict[str, Any]]:
    """
    Detect tables in a PNG/JPG image and return DataFrames.
    [{ "page":1, "df": pd.DataFrame }]
    """
    out: List[Dict[str, Any]] = []
    # PP-Structure does table detection + structure reconstruction
    engine = PPStructure(show_log=False, lang='en')

    # You can pass file path directly
    results = engine(img_path)

  
    for item in results:
        if item.get("type") == "table":
            html = ""
            res = item.get("res", {})
            # Different paddleocr versions store html under slightly different keys
            if isinstance(res, dict):
                html = res.get("html", "") or res.get("structure_html", "")
            if html:
                try:
                    dfs = pd.read_html(html)
                    for df in dfs:
                        out.append({"page": 1, "df": df})
                except Exception:
                    # Fallback: create a single-cell dataframe with raw html
                    out.append({"page": 1, "df": pd.DataFrame({"raw_html":[html]})})

    return out

def df_to_markdown(df: "pd.DataFrame") -> str:
    return df.to_markdown(index=False)
