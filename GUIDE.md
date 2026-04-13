# ML News Classifier v2 — Complete Setup Guide
## For PyCharm on Windows

---

## WHAT'S NEW IN v2

| Feature | Details |
|---------|---------|
| 12 Categories | World, Politics, Business, Technology, Science, Sports, Health, Environment, Entertainment, Crime, Education, Lifestyle |
| Image Upload + OCR | Upload a photo of a newspaper/article — text is extracted automatically |
| Indian Language Support | Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Urdu |
| Auto Translation | Non-English text is translated to English before classification |
| AI Chatbot | Ask questions about any article you uploaded — Who? What? When? Summary? |
| Session Memory | Every article and chat message is saved per session with full history |
| Short Text Handling | Works on 2-word headlines to 400+ word articles |

---

## FOLDER STRUCTURE

```
ml-news-v2/
├── ml_pipeline/
│   ├── data/
│   │   ├── categories.py           ← 12-category taxonomy
│   │   ├── preprocessor.py         ← text cleaning
│   │   ├── download_dataset.py     ← download AG News
│   │   └── create_custom_dataset.py ← add your own articles
│   ├── models/
│   │   └── classifier.py           ← smart routing (short/long text)
│   ├── services/
│   │   ├── ocr_service.py          ← image → text
│   │   ├── translation_service.py  ← Indian lang → English
│   │   ├── chatbot_service.py      ← local Q&A chatbot
│   │   └── session_manager.py      ← SQLite session store
│   └── training/
│       ├── train_baseline.py       ← TF-IDF + LR (87%)
│       ├── train_lstm.py           ← BiLSTM (92%)
│       └── train_distilbert.py     ← DistilBERT (94%)
├── backend/
│   └── api/
│       └── main.py                 ← FastAPI (all endpoints)
├── frontend/
│   └── src/
│       └── App.jsx                 ← React UI (dark theme)
├── requirements.txt
└── GUIDE.md  ← this file
```

---

## PHASE 1 — Python Environment Setup

### Step 1: Open project in PyCharm

1. Extract the ZIP to `C:\Users\YourName\Projects\ml-news-v2`
2. Open **PyCharm → File → Open** → select the folder
3. Click **Trust Project**
4. Press **Alt + F12** to open the Terminal

### Step 2: Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

You'll see **(venv)** in your terminal — good!

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

⏳ Takes 10–20 minutes (PyTorch, HuggingFace, EasyOCR, etc.)

---

## PHASE 2 — Install Tesseract OCR (for image reading)

Tesseract is a free OCR engine you need to read text from images.

### Windows install:

1. Download: https://github.com/UB-Mannheim/tesseract/wiki
2. Download the installer: **tesseract-ocr-w64-setup-5.x.exe**
3. During install, under "Additional language data", check:
   - ✅ Hindi
   - ✅ Bengali
   - ✅ Tamil
   - ✅ Telugu
   - ✅ Gujarati
4. Install to: `C:\Program Files\Tesseract-OCR\`
5. Add to Windows PATH:
   - Search "Environment Variables" in Start menu
   - Edit "Path" → Add: `C:\Program Files\Tesseract-OCR`
6. Restart PyCharm terminal and test:
   ```bash
   tesseract --version
   ```
   Should show: `tesseract 5.x.x`

---

## PHASE 3 — Train the Models

### Step 4: Download the AG News dataset

```bash
python ml_pipeline/data/download_dataset.py
```

Downloads ~120,000 news articles. Creates:
- `ml_pipeline/data/raw/train.json`
- `ml_pipeline/data/raw/val.json`
- `ml_pipeline/data/raw/test.json`

### Step 5: Create custom dataset (optional but recommended)

```bash
python ml_pipeline/data/create_custom_dataset.py
```

Interactive menu appears:
```
1. Add article manually
2. Import from CSV
3. Show statistics
4. Export for training
```

**To import your own CSV**, make a file with these columns:
```csv
text,category
"India won the cricket match","Sports"
"New AI chip launched by Google","Technology"
"Heavy floods in Mumbai","World & International"
```

Then choose option 2 and provide the CSV path.

**Valid category names:**
- World & International
- Politics & Governance
- Business & Finance
- Technology
- Science & Research
- Sports
- Health & Medicine
- Environment & Climate
- Entertainment & Culture
- Crime & Justice
- Education
- Lifestyle & Society

### Step 6: Train baseline model

```bash
python ml_pipeline/training/train_baseline.py
```

⏳ ~3 minutes. Output:
```
Test Accuracy : 87.20%
✅ Baseline training complete!
```

### Step 7: Train LSTM (optional, improves accuracy)

```bash
python ml_pipeline/training/train_lstm.py
```

⏳ ~20 minutes. Output:
```
Test Accuracy: 91.80%
✅ LSTM training complete!
```

### Step 8: Fine-tune DistilBERT (optional, best accuracy)

```bash
python ml_pipeline/training/train_distilbert.py
```

⏳ 30–90 minutes. If out-of-memory error, open `train_distilbert.py` and change `BATCH_SIZE = 16` to `BATCH_SIZE = 8`.

---

## PHASE 4 — Start the Backend API

### Step 9: Run the FastAPI server

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Open: **http://localhost:8000/docs**

You'll see all API endpoints with interactive testing.

### Test endpoints:

**Classify text:**
```bash
curl -X POST "http://localhost:8000/classify/text" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"PM Modi inaugurates new metro line in Delhi\"}"
```

**Create a session:**
```bash
curl -X POST "http://localhost:8000/session" \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"My first session\"}"
```

**List all sessions:**
```bash
curl http://localhost:8000/sessions
```

---

## PHASE 5 — Start the React Frontend

### Step 10: Install Node.js (if not installed)

Download from **nodejs.org** → install LTS version.

### Step 11: Create and run the React app

```bash
cd frontend
npm create vite@latest . -- --template react
# When asked "Current directory is not empty" → select "Ignore files and continue"
npm install
npm run dev
```

Open: **http://localhost:5173**

### Step 12: Replace the default App.jsx

Copy the `App.jsx` from `frontend/src/App.jsx` (already in your project) and it will auto-reload.

---

## HOW TO USE THE APP

### Upload an image (newspaper cutout):
1. Click the **Upload tab**
2. Select the image language (or leave "Auto")
3. Drag & drop a photo of the news article
4. The app automatically:
   - Extracts text via OCR
   - Detects the language
   - Translates to English (if needed)
   - Classifies into one of 12 categories
   - Shows confidence scores for all categories

### Chat with the article:
1. After uploading, click **"Chat about this article"**
2. Ask questions like:
   - "Summarize this article"
   - "Who is mentioned in this news?"
   - "What happened and when?"
   - "What is the main issue discussed?"

### View history:
1. Click **History tab**
2. See all articles in the current session
3. Click any article to chat about it specifically

### Multiple sessions:
- Click **"+ New Session"** in the left sidebar
- Each session keeps its articles and chat history
- Switch between sessions to compare news

---

## API ENDPOINTS REFERENCE

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/classify/text` | Classify text input |
| POST | `/classify/image` | OCR + translate + classify image |
| POST | `/chat` | Chat about uploaded articles |
| POST | `/session` | Create new session |
| GET | `/sessions` | List all sessions |
| GET | `/session/{id}` | Get session with articles + chat |
| DELETE | `/session/{id}` | Delete session |
| GET | `/categories` | List all 12 categories |

---

## COMMON ERRORS & FIXES

| Error | Fix |
|-------|-----|
| `tesseract is not installed` | Install Tesseract and add to PATH (Phase 2) |
| `deep-translator error` | Check internet connection (Google Translate needs internet) |
| `No module named 'easyocr'` | `pip install easyocr` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in training scripts |
| `Cannot extract text from image` | Use higher resolution image; try different language setting |
| `Port 8000 already in use` | Change port: `uvicorn ... --port 8001` |
| `langdetect failed` | `pip install langdetect` |

---

## ARCHITECTURE OVERVIEW

```
Image Upload
     │
     ▼
OCR (Tesseract / EasyOCR)
     │
     ▼
Language Detection (Unicode + langdetect)
     │
     ├── Indian Language? → Google Translate → English
     │
     ▼
Text Preprocessing (NLTK cleaning)
     │
     ├── < 10 words  → Keyword Voting
     ├── 10-50 words → TF-IDF + Logistic Regression
     └── 50+ words   → Ensemble (TF-IDF + LSTM + DistilBERT)
                              │
                              ▼
                    12-Category Classification
                              │
                              ▼
                    Saved to SQLite Session
                              │
                              ▼
                    Chatbot Q&A (RoBERTa + BlenderBot)
```
