# providers.py
from typing import List
import os
import numpy as np
import torch
from functools import lru_cache
from sentence_transformers import SentenceTransformer, CrossEncoder

from dotenv import load_dotenv

# .env dosyasını oku
load_dotenv()

# API anahtarını al
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ Uyarı: GOOGLE_API_KEY .env dosyasında bulunamadı!")

# =========================
# CONFIG (env ile override)
# =========================
EMB_MODEL_NAME = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-small")
# Hız için default MiniLM; Jina kullanmak istersen RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual
RERANKER_NAME  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# =========================
# Embedding (E5)
# =========================

_emb_model: SentenceTransformer | None = None

def _get_emb_model() -> SentenceTransformer:
    global _emb_model
    if _emb_model is None:
        # CPU'da stabil ve hızlı çalışması için
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    return _emb_model

def embed(texts: List[str]) -> np.ndarray:
    """E5 embedding üretir (normalize etmez)."""
    model = _get_emb_model()
    vecs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vecs

# =========================
# Reranker (Cross-Encoder)
# =========================

_reranker: CrossEncoder | None = None

def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:

        trust = "jina" in RERANKER_NAME.lower()
        _reranker = CrossEncoder(
            RERANKER_NAME,
            max_length=384,
            trust_remote_code=trust,
        )
    return _reranker

def rerank(query: str, candidates: List[str]) -> List[float]:
    """Sorgu + aday pasajlar için alaka skorları döndürür (yüksek skor = daha alakalı)."""
    model = _get_reranker()
    pairs = [[query, c] for c in candidates]
    scores = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    return scores.tolist()

# =========================
# (Opsiyonel) Ekstraktif QA – TR SQuAD
# =========================

_QA_MODEL = os.getenv("QA_MODEL", "savasy/bert-base-turkish-squad")
_qa_pipe = None  # lazy load

def qa_extract(question: str, context: str) -> dict:
    """
    Pasajdan doğrudan cevap span'ı çıkarır.
    Dönen örnek: {'answer': '1907', 'score': 0.93, 'start': 123, 'end': 127}
    Kullanmazsan çağırma; yüklenmez ve hız etkisi olmaz.
    """
    global _qa_pipe
    if _qa_pipe is None:
        from transformers import pipeline  # import burada ki ihtiyaca göre yüklensin
        _qa_pipe = pipeline("question-answering", model=_QA_MODEL, tokenizer=_QA_MODEL)
    res = _qa_pipe(question=question, context=context)
    return dict(res)

# =========================
# LLM: Google Gemini
# =========================

def generate(prompt: str) -> str:
    """
    Gemini ile üretken cevap. GOOGLE_API_KEY yoksa 'LLM yapılandırılmadı.' döner.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "LLM yapılandırılmadı."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, max_output_tokens=300, top_p=0.8
            ),
        )
        return response.text.strip() if hasattr(response, "text") else "Cevap oluşturulamadı."
    except Exception as e:
        return f"LLM hata: {e}"