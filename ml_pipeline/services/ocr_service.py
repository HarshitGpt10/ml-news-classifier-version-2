
import easyocr
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import os

from ml_pipeline.services.translation_service import translate_to_english

pytesseract.pytesseract.tesseract_cmd = r"N:\Projects\teseract\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"N:\Projects\teseract\tessdata"

OCR_CONFIG = {
    "en": {"engine": "easyocr", "langs": ['en']},
    "hi": {"engine": "easyocr", "langs": ['en', 'hi']},
    "ta": {"engine": "tesseract", "code": "tam"},
    "bn": {"engine": "tesseract", "code": "ben"},
    "te": {"engine": "tesseract", "code": "tel"},
}

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("L")
    img = img.resize((img.width * 2, img.height * 2))
    img = ImageEnhance.Contrast(img).enhance(2.0)
    return img

def run_ocr(image_path, lang):
    img = preprocess_image(image_path)
    img_array = np.array(img)

    config = OCR_CONFIG.get(lang, OCR_CONFIG["en"])

    if config["engine"] == "easyocr":
        reader = easyocr.Reader(config["langs"], gpu=False)
        results = reader.readtext(img_array, detail=0, paragraph=True)
        return " ".join(results).strip()

    elif config["engine"] == "tesseract":
        return pytesseract.image_to_string(
            img,
            lang=config["code"],
            config="--oem 3 --psm 6"
        ).strip()

    return ""

def extract_text_from_image(image_path: str, language: str) -> str:
    raw_text = run_ocr(image_path, language)

    if not raw_text:
        return ""

    if language != "en":
        result = translate_to_english(raw_text, source_lang=language)
        return result["translated_text"]

    return raw_text