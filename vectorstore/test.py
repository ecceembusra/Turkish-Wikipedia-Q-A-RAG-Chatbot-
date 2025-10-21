import faiss
import numpy as np
import json, os

# Orijinal dosyaların olduğu klasör
VSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VSTORE_DIR, "index.faiss")
META_PATH  = os.path.join(VSTORE_DIR, "meta.jsonl")

os.makedirs(VSTORE_DIR, exist_ok=True)

# FAISS index'i oku
index = faiss.read_index(INDEX_PATH)

# Rastgele örnekleme ile küçültme (%60 tut)
keep_ratio = 0.6
ids = np.arange(index.ntotal)
np.random.shuffle(ids)
keep_ids = ids[:int(len(ids) * keep_ratio)]

# Vektörleri yeniden yapılandır
vectors = index.reconstruct_n(0, index.ntotal)
index_small = faiss.IndexFlatL2(vectors.shape[1])
index_small.add(vectors[keep_ids])

# Yeni klasör oluştur ve kaydet
os.makedirs("vectorstore_small", exist_ok=True)
faiss.write_index(index_small, "vectorstore_small/index.faiss")

# Meta dosyasını da azaltılmış versiyonla eşleştir
with open(META_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()
with open("vectorstore_small/meta.jsonl", "w", encoding="utf-8") as f:
    for i in keep_ids:
        f.write(lines[i])

print("✅ Küçültülmüş index kaydedildi: vectorstore_small/")
