"""
chatbot_service.py
───────────────────
Local Q&A chatbot powered by HuggingFace Transformers.

Two modes:
  1. EXTRACTIVE Q&A  — answers questions by finding spans in the article text
                        Model: deepset/roberta-base-squad2 (~500 MB, offline)
  2. GENERATIVE CHAT — conversational responses about the news
                        Model: facebook/blenderbot-400M-distill (~800 MB, offline)

No internet needed after first download.

Usage:
    from ml_pipeline.services.chatbot_service import ChatbotService
    bot = ChatbotService()
    answer = bot.answer(question="Who won?", context="India won the World Cup...")
"""

import re
import sys
from pathlib import Path

# ── Model config ───────────────────────────────────────────────────────────────

QA_MODEL   = "deepset/roberta-base-squad2"
CHAT_MODEL = "facebook/blenderbot-400M-distill"

# ── Intent patterns ────────────────────────────────────────────────────────────

_EXTRACTIVE_PATTERNS = re.compile(
    r"\b(who|what|when|where|which|how many|how much|"
    r"name|tell me|find|identify|mention|state|specify)\b",
    re.I,
)

_SUMMARY_PATTERNS = re.compile(
    r"\b(summarize|summary|brief|overview|describe|"
    r"explain|what is it about|main point|key point|gist|news all about)\b",
    re.I,
)


class ChatbotService:
    """
    Lazy-loads models on first use.
    Falls back gracefully if models are unavailable.
    """

    def __init__(self):
        self._qa_pipe   = None
        self._chat_pipe = None
        self._loaded_qa   = False
        self._loaded_chat = False

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        context: str,
        chat_history: list[dict] = None,
    ) -> dict:
        """
        Answer a question about the provided news context.

        Args:
            question    : User's question
            context     : The news article text (already translated to English)
            chat_history: Previous Q&A pairs for conversation memory

        Returns:
            {
                "answer"   : str,
                "mode"     : "extractive" | "generative" | "rule",
                "confidence": float,
            }
        """
        if not context.strip():
            return {"answer": "Please upload a news article first so I can answer questions about it.",
                    "mode": "rule", "confidence": 1.0}

        q = question.strip()

        # Summary request
        if _SUMMARY_PATTERNS.search(q):
            # return {"answer": self._summarize(context), "mode": "generative", "confidence": 0.8}
            return {"answer": self._summarize(context), "mode": "summary", "confidence": 0.9}

        # Rule-based responses for common meta-questions
        rule = self._handle_rules(q, context)
        if rule:
            return {"answer": rule, "mode": "rule", "confidence": 1.0}


        # Extractive Q&A for fact questions (who/what/when/where)
        if _EXTRACTIVE_PATTERNS.search(q):
            result = self._extractive_qa(q, context)
            if result and result["score"] > 0.15:
                return {
                    "answer":     result["answer"],
                    "mode":       "extractive",
                    "confidence": round(result["score"], 3),
                }

        # Generative fallback
        gen_answer = self._generative_chat(q, context, chat_history or [])
        return {"answer": gen_answer, "mode": "generative", "confidence": 0.6}

    def preload(self):
        """Pre-download models (run once, takes a few minutes)."""
        print("📥 Downloading Q&A model …")
        self._get_qa_pipe()
        print("📥 Downloading chat model …")
        self._get_chat_pipe()
        print("✅ All models ready!")

    # ── Rule-based responses ──────────────────────────────────────────────────

    @staticmethod
    def _handle_rules(question: str, context: str) -> str | None:
        q = question.lower()

        words = set(re.findall(r"\b\w+\b", q))

        word_count = len(context.split())
        sentences  = [s.strip() for s in re.split(r'[.!?]', context) if len(s.strip()) > 20]

        if any(w in q for w in ["how long", "length", "word count", "how many words"]):
            return f"This article contains approximately {word_count} words."

        if any(w in q for w in ["topic", "category", "type of news", "what kind"]):
            return ("This article has been classified by the ML model. "
                    "You can see the category and confidence score in the panel above.")

        if any(w in q for w in ["original language", "language", "translated from"]):
            return ("The original language and translation details are shown "
                    "in the article panel. Check the language badge there.")

        if any(w in q for w in ["hello", "hi", "hey", "how are you"]):
            return ("Hello! I'm your news assistant. Ask me anything about the article you uploaded — "
                    "who is involved, what happened, when, where, or ask for a summary!")

        if any(w in q for w in ["thank", "thanks", "great", "good"]):
            return "Glad I could help! Feel free to ask more questions about the article."

        if "word" in words and "count" in words:
            return f"This article contains approximately {word_count} words."

        if words & {"topic", "category", "type"}:
            return "Check the classification panel above for category."

        if words & {"hello", "hi", "hey"} or "how are you" in q:
            return "Hello! I'm your news assistant."

        return None

    # ── Extractive Q&A ────────────────────────────────────────────────────────

    def _get_qa_pipe(self):
        if self._loaded_qa:
            return self._qa_pipe
        try:
            from transformers import pipeline
            print(f"Loading Q&A model ({QA_MODEL}) …")
            self._qa_pipe = pipeline(
                "question-answering",
                model=QA_MODEL,
                tokenizer=QA_MODEL,
            )
            print("Q&A model ready ✅")
        except Exception as e:
            print(f"⚠️  Q&A model failed to load: {e}")
            self._qa_pipe = None
        self._loaded_qa = True
        return self._qa_pipe

    def _extractive_qa(self, question: str, context: str) -> dict | None:
        pipe = self._get_qa_pipe()
        if pipe is None:
            return None
        try:
            # Truncate context to 512 tokens (model limit)
            ctx = " ".join(context.split()[:400])
            return pipe(question=question, context=ctx)
        except Exception as e:
            print(f"QA error: {e}")
            return None

    # ── Generative chat ───────────────────────────────────────────────────────

    def _get_chat_pipe(self):
        if self._loaded_chat:
            return self._chat_pipe
        try:
            from transformers import pipeline
            print(f"Loading chat model ({CHAT_MODEL}) …")
            self._chat_pipe = pipeline(
                "text2text-generation",
                model=CHAT_MODEL,
                tokenizer=CHAT_MODEL,
            )
            print("Chat model ready ✅")
        except Exception as e:
            print(f"⚠️  Chat model failed to load: {e}")
            self._chat_pipe = None
        self._loaded_chat = True
        return self._chat_pipe

    def _generative_chat(
        self, question: str, context: str, history: list[dict]
    ) -> str:
        pipe = self._get_chat_pipe()
        if pipe is None:
            return self._fallback_answer(question, context)

        try:
            # Build a prompt grounded in the article
            history_text = ""
            if history:
                for msg in history[-4:]:
                    prefix = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{prefix}: {msg['content']}\n"

            ctx_snippet = " ".join(context.split()[:150])
            prompt = (
                f"News article: {ctx_snippet}\n\n"
                f"{history_text}"
                f"User: {question}\nAssistant:"
            )
            result = pipe(prompt, max_new_tokens=200, do_sample=False)
            answer = result[0]["generated_text"].strip()
            # Remove the prompt echo if present
            if "Assistant:" in answer:
                answer = answer.split("Assistant:")[-1].strip()
            return answer or self._fallback_answer(question, context)

        except Exception as e:
            return self._fallback_answer(question, context)

    # ── Summary ───────────────────────────────────────────────────────────────

    def _summarize(self, context: str) -> str:
        """Rule-based extractive summary using leading sentences."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 30]
        if not sentences:
            return context[:300]
        # First 3 sentences as summary
        summary = " ".join(sentences[:3])
        if len(sentences) > 3:
            summary += f"\n\n...and {len(sentences)-3} more sentences in the article."
        return summary

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_answer(question: str, context: str) -> str:
        """Simple keyword search fallback when models aren't loaded."""
        q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
        sentences = re.split(r'(?<=[.!?])\s+', context)
        best, best_score = "", 0
        for sent in sentences:
            score = sum(1 for w in q_words if w in sent.lower())
            if score > best_score:
                best_score, best = score, sent
        if best and best_score > 0:
            return best.strip()
        return ("I found the article text but couldn't determine a specific answer. "
                "Try rephrasing your question or ask for a summary.")


# ── Global singleton ──────────────────────────────────────────────────────────

_service: ChatbotService | None = None


def get_chatbot() -> ChatbotService:
    global _service
    if _service is None:
        _service = ChatbotService()
    return _service


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = ChatbotService()
    context = """
    India defeated Australia by 6 wickets in the final of the ICC Cricket World Cup 2023
    held at Narendra Modi Stadium in Ahmedabad on November 19, 2023.
    Virat Kohli scored 54 runs and was named Player of the Match.
    India won their third ODI World Cup title.
    """
    questions = [
        "Who won the World Cup?",
        "Where was the match held?",
        "Who was the Player of the Match?",
        "Summarize this article",
    ]
    for q in questions:
        result = bot.answer(q, context)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}  [{result['mode']}]")
