"""
main.py — Advanced FastAPI backend for ML News Classifier v2
"""

import sys
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parents[2]))

from ml_pipeline.data.categories import CATEGORIES
from ml_pipeline.models.classifier import get_classifier
from ml_pipeline.services.ocr_service import extract_text_from_image   # ← NEW
from ml_pipeline.services.session_manager import SessionManager
from ml_pipeline.services.chatbot_service import get_chatbot

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML News Classifier API v2",
    description="8-category news classifier with OCR, translation, chatbot, and sessions",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global services ───────────────────────────────────────────────────────────
_session_mgr: Optional[SessionManager] = None

def get_session_mgr() -> SessionManager:
    global _session_mgr
    if _session_mgr is None:
        _session_mgr = SessionManager()
    return _session_mgr


@app.on_event("startup")
async def startup():
    print("🚀 Starting ML News Classifier v2 …")
    try:
        get_classifier()          # Warm up models
    except Exception as e:
        print(f"⚠️ Classifier warmup failed: {e}")
    get_session_mgr()
    print("✅ API ready at http://localhost:8000")


# ── Request schemas ───────────────────────────────────────────────────────────
class TextClassifyRequest(BaseModel):
    text: str
    session_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: str
    article_id: Optional[str] = None


class CreateSessionRequest(BaseModel):
    name: Optional[str] = ""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "categories": len(CATEGORIES)}


# ── Text Classification ───────────────────────────────────────────────────────
@app.post("/classify/text")
def classify_text(req: TextClassifyRequest):
    if not req.text.strip():
        raise HTTPException(422, "text cannot be empty")

    clf = get_classifier()
    result = clf.predict(req.text)

    article_id = None
    if req.session_id:
        sm = get_session_mgr()
        article_id = sm.add_article(req.session_id, {
            "original_text": req.text,
            "translated_text": req.text,
            "source_lang": "en",
            "was_translated": False,
            "category": result["category"],
            "label": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "word_count": result["word_count"],
            "source_type": "text",
        })

    return {**result, "article_id": article_id}


# ── Image Classification (OCR + Translation) ─────────────────────────────────
@app.post("/classify/image")
async def classify_image(
    file: UploadFile = File(...),
    language: str = Form(...),
    session_id: str = Form(default=""),
):
    """Upload image → OCR → Auto Translate to English → Classify"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(422, f"Expected image, got {file.content_type}")

    try:
        # Save temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # NEW: Use updated OCR + Translation service
        english_text = extract_text_from_image(temp_path, language)

        if not english_text or english_text.strip() == "":
            Path(temp_path).unlink(missing_ok=True)
            raise HTTPException(422, "Could not extract readable text from image. Try a clearer photo.")

        # Classify
        clf = get_classifier()
        result = clf.predict(english_text)

        # Save to session if requested
        article_id = None
        if session_id:
            sm = get_session_mgr()
            article_id = sm.add_article(session_id, {
                "original_text": english_text,   # we already translated
                "translated_text": english_text,
                "source_lang": "hi",             # assuming Hindi for now
                "was_translated": True,
                "category": result["category"],
                "label": result["label"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "word_count": result["word_count"],
                "source_type": "image",
                "image_filename": file.filename,
            })

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

        return {
            "success": True,
            "extracted_text": english_text,
            "classification": result,
            "article_id": article_id
        }

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}


# ── Chat & Session routes (unchanged) ────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    sm = get_session_mgr()
    context = sm.build_context(req.session_id, req.article_id or "")
    if not context:
        return {"answer": "No articles found in this session yet.", "mode": "rule"}

    history = sm.get_chat_history(req.session_id, req.article_id or "", limit=10)
    bot = get_chatbot()
    result = bot.answer(req.message, context, chat_history=history)

    sm.add_chat_message(req.session_id, "user", req.message, req.article_id or "")
    sm.add_chat_message(req.session_id, "assistant", result["answer"], req.article_id or "")
    return result


@app.post("/session")
def create_session(req: CreateSessionRequest):
    sm = get_session_mgr()
    sid = sm.create_session(req.name or "")
    return {"session_id": sid}


@app.get("/sessions")
def list_sessions():
    return {"sessions": get_session_mgr().list_sessions()}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    session = get_session_mgr().get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return session


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    get_session_mgr().delete_session(session_id)
    return {"deleted": True, "session_id": session_id}


@app.get("/categories")
def list_categories():
    return {
        "categories": [
            {
                "id": c.id,
                "name": c.name,
                "short": c.short,
                "icon": c.icon,
                "color": c.color,
                "description": c.description,
            }
            for c in CATEGORIES
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "categories": len(CATEGORIES)}
