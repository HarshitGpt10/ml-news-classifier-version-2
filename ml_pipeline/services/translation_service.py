# """
# translation_service.py
# Reliable translation for Indian languages using IndicTrans2 (HuggingFace)
# """
#
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
#
# # =========================
# # CONFIG
# # =========================
# MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"  # ✅ PUBLIC MODEL
#
# print("🚀 Loading IndicTrans2 model... (first time will take time)")
#
# # =========================
# # LOAD MODEL
# # =========================
# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_NAME,
#     trust_remote_code=True
# )
#
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     MODEL_NAME,
#     trust_remote_code=True
# )
#
# # Device setup
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
#
# # Improve performance
# model.eval()
# torch.set_grad_enabled(False)
#
# print(f"✅ Model loaded on {device}")
#
#
# # =========================
# # TRANSLATION FUNCTION
# # =========================
# def translate_to_english(text: str, src_lang: str = "hin_Deva") -> str:
#     """
#     Translate Indian language text to English.
#
#     src_lang options:
#         Hindi      : "hin_Deva"
#         Marathi    : "mar_Deva"
#         Bengali    : "ben_Beng"
#         Tamil      : "tam_Taml"
#         Telugu     : "tel_Telu"
#         Gujarati   : "guj_Gujr"
#         Kannada    : "kan_Knda"
#         Malayalam  : "mal_Mlym"
#     """
#
#     if not text or text.strip() == "":
#         return text
#
#     try:
#         tgt_lang = "eng_Latn"
#
#         # ✅ FINAL CORRECT FORMAT (IMPORTANT)
#         input_text = f"{src_lang} {text.strip()} {tgt_lang}"
#
#         # Tokenize
#         inputs = tokenizer(
#             input_text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         ).to(device)
#
#         # Generate translation
#         outputs = model.generate(
#             **inputs,
#             max_length=256,
#             num_beams=4
#         )
#
#         # Decode
#         translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#         return translated.strip()
#
#     except Exception as e:
#         print(f"⚠️ Translation failed: {e}")
#         return text
#
#
# # =========================
# # QUICK TEST
# # =========================
# if __name__ == "__main__":
#     test = "भारत ने विश्व कप फाइनल में ऑस्ट्रेलिया को हराया। यह बहुत खुशी की बात है।"
#
#     print("\n📝 Original  :", test)
#     print("🌍 Translated:", translate_to_english(test, src_lang="hin_Deva"))

# MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
# MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"



"""
translation_service.py
───────────────────────
Detects the language of extracted news text and translates to English.

Supported Indian languages:
    Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam,
    Punjabi, Odia, Urdu, Assamese, Sanskrit (and 100+ more via Google)

Dependencies:
    pip install deep-translator langdetect

Usage:
    from ml_pipeline.services.translation_service import translate_to_english, detect_language
    result = translate_to_english("मुंबई में बाढ़ से लाखों प्रभावित")
    # {"text": "Millions affected by flood in Mumbai", "source_lang": "hi", ...}
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass

# ── Language metadata ──────────────────────────────────────────────────────────

@dataclass
class LanguageInfo:
    code:        str   # ISO 639-1
    name:        str
    native_name: str
    script:      str


SUPPORTED_LANGUAGES: list[LanguageInfo] = [
    LanguageInfo("hi", "Hindi",     "हिन्दी",    "Devanagari"),
    LanguageInfo("bn", "Bengali",   "বাংলা",      "Bengali"),
    LanguageInfo("ta", "Tamil",     "தமிழ்",      "Tamil"),
    LanguageInfo("te", "Telugu",    "తెలుగు",     "Telugu"),
    LanguageInfo("mr", "Marathi",   "मराठी",      "Devanagari"),
    LanguageInfo("gu", "Gujarati",  "ગુજરાતી",    "Gujarati"),
    LanguageInfo("kn", "Kannada",   "ಕನ್ನಡ",      "Kannada"),
    LanguageInfo("ml", "Malayalam", "മലയാളം",     "Malayalam"),
    LanguageInfo("pa", "Punjabi",   "ਪੰਜਾਬੀ",    "Gurmukhi"),
    LanguageInfo("or", "Odia",      "ଓଡ଼ିଆ",      "Odia"),
    LanguageInfo("ur", "Urdu",      "اردو",        "Nastaliq"),
    LanguageInfo("as", "Assamese",  "অসমীয়া",    "Bengali"),
    LanguageInfo("en", "English",   "English",     "Latin"),
]

LANG_BY_CODE = {l.code: l for l in SUPPORTED_LANGUAGES}
INDIAN_CODES = {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "ur", "as"}


# ── Language detection ─────────────────────────────────────────────────────────

def detect_language(text: str) -> dict:
    """
    Detect the language of a text string.

    Returns:
        {"code": "hi", "name": "Hindi", "confidence": 0.99, "is_indian": True}
    """
    # Script-based fast detection (Unicode block heuristics)
    script_result = _detect_by_script(text)
    if script_result:
        lang = LANG_BY_CODE.get(script_result)
        return {
            "code":      script_result,
            "name":      lang.name if lang else script_result,
            "confidence": 0.95,
            "is_indian": script_result in INDIAN_CODES,
            "method":    "script",
        }

    # Fallback: langdetect library
    try:
        from langdetect import detect, DetectorFactory, detect_langs
        DetectorFactory.seed = 42   # deterministic
        langs  = detect_langs(text)
        top    = langs[0]
        lang   = LANG_BY_CODE.get(top.lang)
        return {
            "code":      top.lang,
            "name":      lang.name if lang else top.lang,
            "confidence": round(top.prob, 3),
            "is_indian": top.lang in INDIAN_CODES,
            "method":    "langdetect",
        }
    except Exception:
        pass

    # Default: assume English
    return {"code": "en", "name": "English", "confidence": 0.5,
            "is_indian": False, "method": "fallback"}


def _detect_by_script(text: str) -> str | None:
    """Unicode block detection — very fast, no library needed."""
    sample = text[:200]
    counts = {
        "hi": len(re.findall(r"[\u0900-\u097F]", sample)),   # Devanagari (Hindi/Marathi/Sanskrit)
        "bn": len(re.findall(r"[\u0980-\u09FF]", sample)),   # Bengali
        "gu": len(re.findall(r"[\u0A80-\u0AFF]", sample)),   # Gujarati
        "pa": len(re.findall(r"[\u0A00-\u0A7F]", sample)),   # Gurmukhi (Punjabi)
        "or": len(re.findall(r"[\u0B00-\u0B7F]", sample)),   # Odia
        "ta": len(re.findall(r"[\u0B80-\u0BFF]", sample)),   # Tamil
        "te": len(re.findall(r"[\u0C00-\u0C7F]", sample)),   # Telugu
        "kn": len(re.findall(r"[\u0C80-\u0CFF]", sample)),   # Kannada
        "ml": len(re.findall(r"[\u0D00-\u0D7F]", sample)),   # Malayalam
        "ur": len(re.findall(r"[\u0600-\u06FF]", sample)),   # Arabic/Urdu
    }
    best = max(counts, key=counts.get)
    return best if counts[best] > 5 else None


# ── Translation ────────────────────────────────────────────────────────────────

def translate_to_english(text: str, source_lang: str = "auto") -> dict:
    """
    Translate text to English using deep-translator (Google Translate backend).

    Args:
        text        : Input text (any language)
        source_lang : ISO code or "auto"

    Returns:
        {
            "original_text"  : str,
            "translated_text": str,
            "source_lang"    : {"code", "name", "is_indian"},
            "was_translated" : bool,
            "engine"         : str,
        }
    """
    if not text or not text.strip():
        return _empty_result(text)

    # Detect language
    if source_lang == "auto":
        lang_info = detect_language(text)
    else:
        info = LANG_BY_CODE.get(source_lang)
        lang_info = {
            "code": source_lang,
            "name": info.name if info else source_lang,
            "is_indian": source_lang in INDIAN_CODES,
        }

    # If already English, skip translation
    if lang_info["code"] == "en":
        return {
            "original_text":   text,
            "translated_text": text,
            "source_lang":     lang_info,
            "was_translated":  False,
            "engine":          "none",
        }

    # Translate via deep-translator
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(
            source=lang_info["code"],
            target="en",
        )
        # Google Translate has a ~5000 char limit; chunk if needed
        translated = _translate_in_chunks(translator, text)
        return {
            "original_text":   text,
            "translated_text": translated,
            "source_lang":     lang_info,
            "was_translated":  True,
            "engine":          "google",
        }
    except ImportError:
        return {
            "original_text":   text,
            "translated_text": text,
            "source_lang":     lang_info,
            "was_translated":  False,
            "engine":          "none",
            "error":           "deep-translator not installed: pip install deep-translator",
        }
    except Exception as e:
        return {
            "original_text":   text,
            "translated_text": text,
            "source_lang":     lang_info,
            "was_translated":  False,
            "engine":          "failed",
            "error":           str(e),
        }


def _translate_in_chunks(translator, text: str, chunk_size: int = 4500) -> str:
    """Split long text into chunks (Google limit ~5000 chars)."""
    if len(text) <= chunk_size:
        return translator.translate(text)
    sentences = text.split(". ")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current:
        chunks.append(current.strip())
    return " ".join(translator.translate(c) for c in chunks if c)


def _empty_result(text: str) -> dict:
    return {
        "original_text":   text,
        "translated_text": text,
        "source_lang":     {"code": "en", "name": "English", "is_indian": False},
        "was_translated":  False,
        "engine":          "none",
    }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_texts = [
        ("Hindi",   "मुंबई में भारी बारिश के कारण लाखों लोग प्रभावित हुए।"),
        ("Bengali", "কলকাতায় আজ বিশেষ ক্রিকেট ম্যাচ অনুষ্ঠিত হবে।"),
        ("Tamil",   "சென்னையில் புதிய தொழில்நுட்ப மையம் திறக்கப்பட்டது."),
        ("English", "Apple reports record quarterly earnings."),
    ]
    for lang, text in test_texts:
        result = translate_to_english(text)
        print(f"\n[{lang}]")
        print(f"  Original   : {result['original_text'][:60]}")
        print(f"  Translated : {result['translated_text'][:60]}")
        print(f"  Was translated: {result['was_translated']}")
