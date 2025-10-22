# app.py
import os, textwrap
import streamlit as st
from rag_pipeline import load_vectorstore, search_chunks, generate_answer

st.set_page_config(page_title="Turkish Wikipedia Q&A (Gemini RAG)", page_icon="🧠")

@st.cache_resource(show_spinner=False)
def _load_vdb():
    return load_vectorstore()

st.title("🧠 Turkish Wikipedia Q&A")

with st.expander("⚙️ Ayarlar", expanded=False):
    top_k = st.slider("Top K (kaç pasaj getirilsin?)", 2, 8, 5)
    show_passages = st.checkbox("Getirilen pasaj özetini göster", value=True)
    st.info("LLM: " + ("Gemini ✅" if os.getenv("GOOGLE_API_KEY") else "Yapılandırılmadı ❌"))

query = st.text_input("📝 Sorunuzu yazın (ör. “Türkiye'nin ilk kadın pilotu kimdir?”)")

if st.button("Cevabı Getir", type="primary", use_container_width=True) and query.strip():
    try:
        with st.spinner("Aranıyor ve cevap oluşturuluyor..."):
            index, records = _load_vdb()
            answer = generate_answer(query.strip(), index, records, top_k=top_k)
            hits = search_chunks(query.strip(), index, records, top_k=top_k)

        st.subheader("✅ Yanıt")
        st.write(answer)

        st.subheader("🔎 Kaynaklar")
        if hits:
            for i, h in enumerate(hits, 1):
                title = h.get("title") or "(başlık yok)"
                url   = h.get("source") or ""
                score = h.get("score_rerank", 0.0)
                lead  = textwrap.shorten((h.get("text") or "").replace("\n", " "), width=220, placeholder="…")
                if url:
                    st.markdown(f"**{i}.** [{title}]({url})  \nRerank skoru: `{score:.3f}`")
                else:
                    st.markdown(f"**{i}.** {title}  \nRerank skoru: `{score:.3f}`")
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
