# data_preparation.py
import os, re, time, json
from typing import List, Dict, Tuple

import numpy as np
import faiss
from datasets import load_dataset, DownloadConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding sağlayıcısı (E5) -> providers.embed
from providers import embed

# =========================
# AYARLAR 
# =========================

DATASET_NAME = "wikimedia/wikipedia"   # HF dataset adı
WIKI_CONFIG  = "20231101.tr"           # Türkçe dump sürümü
MAX_PAGES    = 500                     # hızlı demo için 500
CHUNK_SIZE   = 1000                    # büyük parça = daha az chunk
CHUNK_OVERLAP= 100
MAX_CHUNKS   = 8000                    # güvenli üst limit (None yapıldığında sınırsız)

VSTORE_DIR   = "vectorstore"
META_JSONL   = "meta.jsonl"
FAISS_FILE   = "index.faiss"
SIGN_FILE    = "signature.json"

# =========================
# Yardımcılar
# =========================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def slugify_title(title: str) -> str:
    t = (title or "").strip().replace(" ", "_")
    t = re.sub(r"[^\w\-ÇçĞğİıÖöŞşÜü]+", "", t, flags=re.UNICODE)
    return t or "Sayfa"

# Gereksiz template token temizliği 
TOKEN_PAT = re.compile(r"<\|/?(system|user|assistant|start|end)\|>|<\/?s>|<\/?unk>", re.IGNORECASE)
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = TOKEN_PAT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

# =========================
# Chunking
# =========================

def chunk_documents(rows: List[Dict]) -> Tuple[List[str], List[Dict]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    texts_raw: List[str] = []
    metas: List[Dict] = []

    for r in rows:
        title = clean_text(r.get("title", ""))
        text  = clean_text(r.get("text", ""))
        if not text:
            continue

        url = f"https://tr.wikipedia.org/wiki/{slugify_title(title)}"
        chunks = splitter.split_text(text)

        for i, ch in enumerate(chunks):
            ch = ch.strip()
            if not ch:
                continue
            texts_raw.append(ch)  # meta'ya RAW metin yazılıyor
            metas.append({"title": title or "(başlık yok)", "chunk_id": i, "source": url})

            if MAX_CHUNKS and len(texts_raw) >= MAX_CHUNKS:
                return texts_raw, metas

    return texts_raw, metas

# =========================
# FAISS (HNSW) — hızlı ANN arama
# =========================

def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    """
    Cosine benzerliği için L2 normalize edip Inner-Product ile HNSW kullanılır.
    HNSW, IndexFlat'e yakın doğrulukta olup sorgu süresini ciddi düşürür.
    """
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    M = 32  # graph degree
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80  # inşa kalitesi (yüksek = daha iyi/az fark)
    index.add(vecs)
    return index

# =========================
# Ana akış
# =========================

def main():
    t0 = time.time()

    print("👉 Wikipedia(TR) yükleniyor...")
    split_expr = f"train[:{MAX_PAGES}]" if MAX_PAGES else "train"
    ds = load_dataset(
        DATASET_NAME, WIKI_CONFIG,
        split=split_expr,
        download_config=DownloadConfig(max_retries=5),
    )
    print(f"Toplam sayfa (seçim sonrası): {len(ds)}")

    print("👉 Chunk'lanıyor...")
    texts_raw, metas = chunk_documents([dict(x) for x in ds])
    print(f"Toplam chunk: {len(texts_raw)}")
    if not texts_raw:
        raise SystemExit("⚠️ Metin bulunamadı.")

    print("👉 Embedding hesaplanıyor (E5)...")
    # E5 kuralı: embed edilirken PASSAGE prefix kullan, meta'da görünmesin
    texts_for_emb = [f"passage: {t}" for t in texts_raw]
    vecs = np.asarray(embed(texts_for_emb), dtype="float32")
    if vecs.ndim != 2:
        raise ValueError(f"Beklenen (N,D) vektör, gelen {vecs.shape}")

    print("👉 FAISS (HNSW) indeks oluşturuluyor...")
    index = build_faiss_index(vecs)

    print("👉 Kaydediliyor...")
    ensure_dir(VSTORE_DIR)
    faiss.write_index(index, os.path.join(VSTORE_DIR, FAISS_FILE))

    meta_path = os.path.join(VSTORE_DIR, META_JSONL)
    with open(meta_path, "w", encoding="utf-8") as f:
        for t, m in zip(texts_raw, metas):
            f.write(json.dumps({"text": t, "metadata": m}, ensure_ascii=False) + "\n")

    sign_path = os.path.join(VSTORE_DIR, SIGN_FILE)
    with open(sign_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": f"{DATASET_NAME}:{WIKI_CONFIG}",
            "max_pages": MAX_PAGES,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "max_chunks": MAX_CHUNKS,
            "faiss": {"type": "HNSWFlat", "metric": "IP", "M": 32, "efConstruction": 80},
            "emb_model": os.getenv("EMB_MODEL", "intfloat/multilingual-e5-small"),
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ Tamamlandı. Süre: {(time.time()-t0):.1f} sn | Çıktı klasörü: {VSTORE_DIR}")

if __name__ == "__main__":
    main()