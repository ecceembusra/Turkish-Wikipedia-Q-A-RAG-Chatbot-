# app.py
import os
import textwrap
import streamlit as st
from rag_pipeline import load_vectorstore, generate_answer, search_chunks  # ← search_chunks eklendi

st.set_page_config(page_title="Turkish Wikipedia Q&A (Gemini/OpenAI RAG)", page_icon="🧠", layout="wide")

st.title("🧠 Turkish Wikipedia Q&A")
status = st.empty()
status.info("İndeks yükleniyor... (ilk açılış biraz sürebilir)")

@st.cache_resource(show_spinner=False)
def _load_vdb():
    return load_vectorstore()  # (index, records)

# vektör deposunu yükle
try:
    index, records = _load_vdb()
    status.success("Hazır!")
except Exception as e:
    status.error(f"Başlatma hatası: {e}")
    st.stop()

with st.expander("⚙️ Ayarlar", expanded=False):
    top_k = st.slider("Top K (kaç pasaj getirilsin?)", 2, 8, 5)
    show_passages = st.checkbox("Getirilen pasaj özetini göster", value=True)
    provider_ok = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    st.info("LLM anahtarı: " + ("✅ bulundu" if provider_ok else "❌ yok"))

query = st.text_input("📝 Sorunuzu yazın (ör. “Türkiye'nin ilk kadın pilotu kimdir?”)")

if st.button("Cevabı Getir", type="primary", use_container_width=True) and query.strip():
    try:
        with st.spinner("Aranıyor ve cevap oluşturuluyor..."):
            q = query.strip()
            answer = generate_answer(q, index, records, top_k=top_k)
            hits = search_chunks(q, index, records, top_k=top_k)

        st.subheader("✅ Yanıt")
        st.write(answer)

        st.subheader("🔎 Kaynaklar")
        if hits:
            for i, h in enumerate(hits, 1):
                title = h.get("title") or "(başlık yok)"
                url   = h.get("source") or ""
                score = float(h.get("score_rerank", h.get("score_boosted", 0.0)))
                lead  = textwrap.shorten((h.get("text") or "").replace("\n", " "), width=220, placeholder="…")
                line = f"**{i}.** {title}  \nRerank skoru: `{score:.3f}`"
                st.markdown(f"[{line}]({url})" if url else line)
                if show_passages and lead:
                    st.caption(lead)
                st.markdown("---")
        else:
            st.write("_Kaynak bulunamadı._")

    except FileNotFoundError as e:
        st.error("Vektör deposu bulunamadı. Önce `python data_preparation.py` çalıştırın.")
        st.code(str(e))
    except Exception as e:
        st.error("Bir hata oluştu.")
        st.exception(e)
