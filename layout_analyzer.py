import layoutparser as lp
from PIL import Image

def get_layout_from_pdf(pdf_path):
    """Analyzes PDF layout to find figures and their surrounding text."""
    try:
        # Load the layout model
        model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                                             label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                                             extra_config_path=None,
                                             enforce_cpu=False)

        layout = model.detect(Image.open('temp_file.png'))

        # Extract figures and text blocks
        figure_blocks = [b for b in layout if b.type == "Figure"]
        text_blocks = [b for b in layout if b.type == "Text"]

        correlations = []
        for fig_block in figure_blocks:
            # Simple logic: find the closest text block below the figure
            potential_captions = [b for b in text_blocks if b.block.y_1 > fig_block.block.y_2]
            if potential_captions:
                closest_caption = min(potential_captions, key=lambda b: b.block.y_1)

                # You would need to OCR the caption block here
                caption_text = "This is a placeholder caption text."

                correlations.append({
                    "image_bbox": fig_block.coordinates,
                    "caption": caption_text,
                    "context": "Here you would put surrounding text from other blocks"
                })
        return correlations
    except Exception as e:
        print(f"Error during layout analysis: {e}")
        return None