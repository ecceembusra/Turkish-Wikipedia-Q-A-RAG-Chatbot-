# Turkish-Wikipedia-QA-RAG-Chatbot-
# 🧠 Turkish Wikipedia Q&A (RAG Chatbot)

> Türkçe Wikipedia makaleleri üzerinde **Retrieval-Augmented Generation (RAG)** mimarisiyle çalışan bir Soru-Cevap uygulaması.  
> Kullanıcılar doğal dilde Türkçe sorular yazarak Wikipedia’dan bilgi tabanlı yanıtlar alabilir.

---

## 🚀 Proje Özeti

Bu proje, **Türkçe Wikipedia verileri** üzerinde geliştirilmiş bir **RAG (Retrieval-Augmented Generation)** tabanlı bilgi erişim ve soru-cevap sistemidir.  
Model, arka planda **vektör benzerliği**, **FAISS tabanlı indeksleme** ve **Gemini LLM** entegrasyonu ile çalışmaktadır.

Uygulama Hugging Face Spaces üzerinde barındırılmaktadır:  
👉 [**Live Demo (Hugging Face Space)**](https://huggingface.co/spaces/ecceembusra/turkish-wikipedia-rag)

Uygulama Render Spaces üzerinde barındırılmaktadır:  
👉 [**Live Demo (Render Space)**](https://huggingface.co/spaces/ecceembusra/turkish-wikipedia-rag)

---

## 🧩 Veri Seti Hakkında

Bu proje, **Türkçe Wikipedia** makalelerinden oluşturulmuş özel bir veri kümesi kullanmaktadır.  
Veri seti, **RAG (Retrieval-Augmented Generation)** tabanlı soru–cevap sistemleri geliştirmek amacıyla hazırlanmıştır.

### 📚 Kaynak
- **Orijinal Kaynak:** [Wikipedia Türkiye Dump](https://dumps.wikimedia.org/trwiki/latest/)  
- **Lisans:** [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)  
  (Atıf verilerek akademik ve araştırma amaçlı kullanılabilir)

### 🧠 Veri Hazırlama Süreci
1. **Veri Çekimi:** Wikipedia dökümü `wikiextractor` aracıyla temiz metin haline getirildi.  
2. **Metin Bölütleme:** Makaleler, 300–500 kelimelik küçük parçalara (chunk) ayrıldı.  
3. **Meta Veriler:** Her parça için başlık, kaynak URL ve ek bilgiler eklendi.  
4. **Vektörleştirme:** Tüm parçalar **Sentence Transformers (E5 model)** kullanılarak vektörlere dönüştürüldü.  
5. **Depolama:** Sonuçlar FAISS tabanlı vektör deposuna kaydedildi:
   - `index.faiss` → Vektör indeksi  
   - `meta.jsonl` → Metin parçaları ve metadata bilgileri

### 📊 Veri Kümesi Özeti
| Metrik | Değer |
|:-------|:------:|
| Makale Sayısı | ~30.000 |
| Parça (Chunk) Sayısı | ~120.000 |
| Ortalama Kelime Sayısı | 380 |
| Kullanılan Model | `intfloat/multilingual-e5-base` |
| Vektör Boyutu | 768 |

### ⚖️ Kullanım Notu
Veri kümesi, **herkese açık Türkçe Wikipedia** içeriklerinden oluşturulmuştur.  
Hiçbir özel, gizli veya telifli içerik kullanılmamıştır.  
Veri seti **araştırma, eğitim ve akademik projelerde** serbestçe kullanılabilir.

---

## 🧩 Mimarinin Bileşenleri

| Katman | Açıklama |
|--------|-----------|
| **📄 Veri Kaynağı** | Türkçe Wikipedia makaleleri (paragraf bazında bölünmüş) |
| **⚙️ Embedding Modeli** | `intfloat/multilingual-e5-large` (çok dilli vektör temsili) |
| **🧮 Vektör Veritabanı** | FAISS – hızlı benzerlik araması |
| **🤖 LLM (Dil Modeli)** | Google **Gemini 1.5 Flash API** (retrieval sonrası yanıt üretimi) |
| **🔍 Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **🧠 QA Extractor** | Extractive QA – kısa yanıt çıkarımı için |
| **🖥️ Arayüz** | Streamlit tabanlı web uygulaması |
| **☁️ Deployment** | Hugging Face Spaces (Python, Streamlit SDK 3) |

---

## 🏗️ Proje Yapısı
```bash
📂 turkish-wikipedia-rag/
turkish-wikipedia-rag/
 ├── app.py                # Streamlit arayüzü
 ├── rag_pipeline.py       # Ana RAG pipeline (retrieval + rerank + generation)
 ├── providers.py          # LLM, Embedding ve Reranker fonksiyonları
 ├── data_preparation.py   # Wikipedia verisini işleyip FAISS indeksi oluşturur
 ├── requirements.txt      # Kütüphane gereksinimleri
 ├── vectorstore/
 │   ├── index.faiss       # Vektör indeks dosyası
 │   └── meta.jsonl.gz     # Metaveri kayıtları
 └── README.md
```
---

## ⚙️ Kullanılan Teknolojiler

| Alan | Teknoloji |
|------|------------|
| **Programlama Dili** | Python 3.10 |
| **Kütüphaneler** | `streamlit`, `faiss-cpu`, `sentence-transformers`, `transformers`, `numpy`, `scikit-learn`, `google-generativeai` |
| **Vektör DB** | FAISS |
| **Ortam** | Hugging Face Spaces |
| **Model Entegrasyonları** | Google Gemini API, HuggingFace Hub |

---

## 💡 Çalışma Mantığı

1. **Sorgu** kullanıcıdan alınır.  
2. Sorgu embedding’e dönüştürülür (`multilingual-e5-large`).  
3. **FAISS** kullanılarak en benzer paragraflar seçilir.  
4. **Reranker** en alakalı pasajları sıralar.  
5. **QA Extractor** ve **Gemini LLM** yanıt üretir.  
6. Sistem hem cevabı hem de kaynak URL’leri gösterir.

---

## 🧠 Uygulama Ekran Görüntüleri

### 🔹  — Başlangıç ve Başkent Sorgusu
<p align="center">
  <img src="img/UI.JPG" alt="Uygulama Başlangıç Ekranı" width="45%"/>
  &nbsp;&nbsp;&nbsp;
  <img src="img/Başkent.JPG" alt="Başkent Sorgusu" width="45%"/>
</p>

---

### 🔹  — İlk Kadın Pilot ve Beşiktaş Kuruluş Tarihi
<p align="center">
  <img src="img/Pilot.JPG" alt="İlk Kadın Pilot Sorgusu" width="45%"/>
  &nbsp;&nbsp;&nbsp;
  <img src="img/Beşiktaş.JPG" alt="Beşiktaş Kuruluş Tarihi" width="45%"/>
</p>

> Örnek Sorular:

> - “Türkiye’nin başkenti hangi şehirdir?”  
> - “Türkiye’nin ilk kadın pilotu kimdir?”
> - “Türkiye’nin ilk Cumhurbaşkanı kimdir?”  
> - “Beşiktaş ne zaman kuruldu?”

---

## 🧰 Kurulum

### 1️⃣ Ortamı Hazırla
```bash
git clone https://github.com/ecceembusra/turkish-wikipedia-rag.git
cd turkish-wikipedia-rag
pip install -r requirements.txt
```
2️⃣ FAISS İndeksi Oluşturma
```bash
python data_preparation.py
```
## 🔐 Deployment Secrets ve Variables

Uygulama hem **Hugging Face Spaces** hem de **Render** platformları üzerinden çalıştırılabilir.  
Her iki platformda da API anahtarları ve proje bilgileri **gizli değişkenler (Secrets / Variables)** alanına eklenmelidir.

---

### 🤗 Hugging Face Deployment Secrets

Hugging Face üzerinden çalıştırmak için aşağıdaki ortam değişkenlerini tanımlayın:

| Değişken Adı | Açıklama | Örnek Değer |
|---------------|-----------|--------------|
| `GOOGLE_API_KEY` | Gemini API anahtarı (Google AI Studio üzerinden alınır) | `AIzaSyB...` |
| `HF_SPACE_REPO_ID` | Hugging Face Space kimliği | `ecceembusra/turkish-wikipedia-rag` |

> 💡 **Not:** Bu bilgiler `Settings → Variables and Secrets` sekmesinden tanımlanmalıdır.  
> API anahtarları gizli tutulur, **public kullanıcılar tarafından görüntülenemez.**

---

### ⚙️ Render Deployment Secrets

Render üzerinde deploy işlemi yapılacaksa aşağıdaki ortam değişkenlerini tanımlayın:

| Değişken Adı | Açıklama | Örnek Değer |
|---------------|-----------|--------------|
| `GOOGLE_API_KEY` | Gemini API anahtarı (Google AI Studio üzerinden alınır) | `AIzaSyB...` |
| `HF_SPACE_REPO_ID` | (Opsiyonel) Hugging Face Space ID (eğer HF tabanlı entegrasyon kullanılacaksa) | `ecceembusra/turkish-wikipedia-rag` |
| `PORT` | Streamlit uygulamasının Render tarafından dinleneceği port | `10000` |
| `PYTHON_VERSION` | (Opsiyonel) Render build için kullanılacak Python sürümü | `3.10` |

> ⚙️ **Render Ayar Adımları:**
> 1. Render projesinde **Environment → Environment Variables** sekmesine gidin.  
> 2. Yukarıdaki değişkenleri tek tek ekleyin.  
> 3. Her değişiklikten sonra **“Save Changes”** ve ardından **“Redeploy”** butonuna basın.  
> 4. Deploy tamamlandığında uygulama otomatik olarak belirtilen port üzerinden çalışır.

---

✨ Bu değişkenler, hem **RAG tabanlı chatbotun doğru API erişimini** sağlar  
hem de deploy sürecinde **bağlantı ve kimlik doğrulama hatalarının** önüne geçer.

🧩 RAG Pipeline Akışı
```bash
flowchart TD
    A[Soru] --> B[Embedding (E5)]
    B --> C[FAISS Search]
    C --> D[Reranker]
    D --> E[QA Extractor]
    E --> F[Gemini LLM]
    F --> G[Cevap + Kaynaklar]
```
---

## 🔎 Kaynakların Gösterimi ve Rerank Skoru

### 📚 Neden Kaynaklar Gösteriliyor?

Bu proje, **Retrieval-Augmented Generation (RAG)** mimarisine dayalı olduğu için, sistem sadece LLM’in ürettiği metni göstermiyor;  
aynı zamanda cevabın dayandığı **orijinal Wikipedia pasajlarını** da kullanıcının erişimine açıyor.  
Bunun iki temel nedeni var:

1. **Doğrulanabilirlik (Explainability):**  
   LLM’ler bazen “hallucination” (uydurma) yanıtlar verebilir.  
   Kullanıcıya kaynağı göstermek, modelin yanıtının gerçekten Wikipedia verisinden mi türetildiğini görmesini sağlar.

2. **Şeffaflık ve güven:**  
   Son kullanıcı, cevabın hangi makaleden veya URL’den geldiğini açıkça görebilir.  
   Bu, sistemi hem akademik hem de üretim ortamında daha güvenilir hale getirir.

---

### 🧮 Rerank Skoru Nedir?

**Rerank skoru**, bir sorgu ile elde edilen Wikipedia pasajlarının **sorguya ne kadar anlamca yakın olduğunu** ölçen bir benzerlik skorudur.

1. **İlk aşamada**, FAISS vektör araması ile sorguya en benzer `N` pasaj bulunur.  
   (Bu aşamada benzerlikler sadece embedding uzayında ölçülür — genellikle **cosine similarity**.)

2. **İkinci aşamada**, her bir aday pasaj yeniden değerlendirilir (**re-ranking**)  
   ve sorgu ile pasaj arasındaki semantik ilişki **cross-encoder modeli** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) kullanılarak yeniden puanlanır.

3. Bu model, sorgu ve pasaj çiftini birlikte değerlendirir ve 0.0 ile 1.0 arasında bir **relevance score (rerank skoru)** üretir.

4. Sonuç olarak:
   - En yüksek skorlu pasaj(lar) **LLM’e bağlam olarak verilir**,  
   - Diğerleri kaynak olarak listelenir.

---

### 🔢 Rerank Skoru Nasıl Hesaplanır?

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Örnek sorgu ve ilk FAISS sonuçlarından alınan pasajlar
query = "Türkiye'nin ilk kadın pilotu kimdir?"
passages = [
    "Sabiha Gökçen Türkiye'nin ilk kadın pilotudur.",
    "Türkiye Cumhuriyeti 1923 yılında ilan edilmiştir."
]

pairs = [(query, p) for p in passages]
scores = model.predict(pairs)

# scores ≈ [0.97, 0.23]
```
Bu skorlar daha sonra normalize edilir ve en yüksek skorlar Streamlit arayüzünde “Rerank skoru” etiketiyle gösterilir.
### 📊 Örnek:

| **Pasaj** | **Rerank Skoru** |
|------------|:----------------:|
| “Sabiha Gökçen Türkiye’nin ilk kadın pilotudur.” | **0.971** |
| “Türkiye Cumhuriyeti 1923 yılında ilan edilmiştir.” | 0.231 |

---

Bu durumda sistem, **Sabiha Gökçen**’i içeren pasajı seçer ve cevabı şu şekilde oluşturur:

> ✅ **Yanıt:** Sabiha Gökçen.  
> Türkiye’nin ilk kadın pilotu kimdir sorusunun yanıtıdır.  
> **Kaynak:** [Wikipedia - Mustafa Kemal Atatürk](https://tr.wikipedia.org/wiki/Mustafa_Kemal_Atat%C3%BCrk)

---

## 🧠 Kısa Özet


| *Aşama* | *Model / Teknik* | *Amaç* |
|:----------:|:------------------:|:---------|
| 1️⃣ *Retriever (FAISS)* | intfloat/multilingual-e5-large | En benzer pasajları bulur |
| 2️⃣ *Reranker* | cross-encoder/ms-marco-MiniLM-L-6-v2 | En anlamlı pasajı seçer |
| 3️⃣ *Generator (LLM)* | Gemini API | Nihai cevabı üretir |
| 4️⃣ *Explainability* | Kaynak bağlantıları | Kullanıcıya şeffaflık sağlar |

---

### 🔍 Sonuç

> “*Rerank skoru*” yüksek olan pasajlar, cevabın çekirdeğini oluşturur.  
> Bu skor, projenin doğru bilgiye erişim kalitesini ölçmek için de bir *metrik* olarak değerlendirilebilir.

---
### 💡 Ek Bilgi

*Rerank skoru*; providers.py dosyasındaki rerank() fonksiyonu tarafından  
CrossEncoder modeli ile (sorgu, pasaj) çiftleri üzerinden hesaplanır.

Model:
```python
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
```

## 🧩 Örnek Yanıtlar

---

### ❓ Soru:
**Türkiye'nin başkenti hangi şehirdir?**

**✅ Cevap:** Ankara  
**📚 Kaynak:** [Wikipedia - Türkiye](https://tr.wikipedia.org/wiki/T%C3%BCrkiye)

---

### ❓ Soru:
**Türkiye'nin ilk kadın pilotu kimdir?**

**✅ Cevap:** Sabiha Gökçen  
**📚 Kaynak:** [Wikipedia - Mustafa Kemal Atatürk](https://tr.wikipedia.org/wiki/Mustafa_Kemal_Atat%C3%BCrk)

---

### ❓ Soru:
**Beşiktaş ne zaman kuruldu?**

**✅ Cevap:** 1903 yılında  
**📚 Kaynak:** [Wikipedia - Beşiktaş JK](https://tr.wikipedia.org/wiki/Be%C5%9Fikta%C5%9F_JK)

---

## 📈 Performans ve Özellikler

- ⚡ **Ortalama yanıt süresi:** ~1.3 saniye  
- 🧠 **Toplam vektör sayısı:** ≈ 80.000 pasaj  
- 🎯 **FAISS** ile yüksek hızlı benzerlik araması  
- 🧩 Türkçe metinler için özel optimize edilmiş **preprocessing pipeline**

---

> Bu sonuçlar, sistemin Wikipedia tabanlı Türkçe sorgulara saniyeler içinde doğru ve kaynaklı yanıtlar üretebildiğini göstermektedir.
