import os
import json
import logging
from pathlib import Path
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
import wikipedia
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()
nltk.download("punkt")
nltk.download("punkt_tab")

# --- Ayarlar ---
DATA_DIR = os.getenv("DATA_DIR", "data")
CHUNK_DIR = os.getenv("CHUNK_DIR", "chunks")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wiki_collection")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embed_model_name=EMBED_MODEL):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        vector = self.embed_model.encode(query).tolist()

        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k
        )
        return [
            {
                "score": hit.score,
                "title": hit.payload.get("title"),
                "chunk_index": hit.payload.get("chunk_index")
            }
            for hit in results
        ]
        
def interactive_query():
    retriever = Retriever()
    while True:
        query = input("\nSoru (çıkmak için q): ")
        if query.lower() == "q":
            break
        results = retriever.search(query)
        for i, res in enumerate(results, 1):
            print(f"\n[{i}] Başlık: {res['title']}, Chunk: {res['chunk_index']}, Skor: {res['score']:.4f}")


# --- Wikipedia Fetcher ---
class WikiFetcher:
    def __init__(self, save_dir: str = DATA_DIR):
        self.save_path = Path(save_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
        wikipedia.set_lang("tr")

    def fetch_and_save(self, title: str) -> Path:
        try:
            content = wikipedia.page(title).content
            clean_title = title.replace(" ", "_")
            file_path = self.save_path / f"{clean_title}.json"
            data = {"title": title, "content": content}
            file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"[✓] Kaydedildi: {file_path}")
            return file_path
        except wikipedia.exceptions.PageError:
            logger.warning(f"[X] Sayfa bulunamadı: {title}")
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"[X] Çok anlamlı başlık: {title} - Öneriler: {e.options}")
        return None


# --- Document Ingestor ---
class DocumentIngestor:
    def __init__(self):
        self.data_dir = Path(DATA_DIR)
        self.chunk_dir = Path(CHUNK_DIR)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.chunk_token_limit = 500
        self.chunk_overlap = 50

        self._init_collection()

    def _init_collection(self):
        # Mevcut koleksiyonları al
        collections = [c.name for c in self.qdrant.get_collections().collections]

        if COLLECTION_NAME not in collections:
            self.qdrant.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.embed_model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                )
            )
            logger.info(f"[✓] Koleksiyon oluşturuldu: {COLLECTION_NAME}")
        else:
            logger.info(f"[i] Koleksiyon zaten mevcut: {COLLECTION_NAME}")


    def chunk_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text, language="turkish")
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.chunk_token_limit:
                chunks.append(" ".join(current_chunk))
                overlap = current_chunk[-self.chunk_overlap:] if self.chunk_overlap < len(current_chunk) else current_chunk
                current_chunk = overlap.copy()
                current_len = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sent)
            current_len += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_and_store(self):
        for file_path in self.data_dir.glob("*.json"):
            logger.info(f"[→] İşleniyor: {file_path.name}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", file_path.stem)
            content = data.get("content", "")

            chunks = self.chunk_text(content)
            points = []

            for idx, chunk in enumerate(chunks):
                try:
                    vector = self.embed_model.encode(chunk).tolist()
                except Exception as e:
                    logger.error(f"[X] Embed hatası: {e} ({title}, chunk {idx})")
                    continue

                meta = {"title": title, "chunk_index": idx}
                point = PointStruct(id=f"{title}_{idx}", vector=vector, payload=meta)
                points.append(point)

            chunk_file = self.chunk_dir / f"{title}_chunks.json"
            with open(chunk_file, "w", encoding="utf-8") as cf:
                json.dump([{"content": c, "metadata": {"title": title, "chunk_index": i}} for i, c in enumerate(chunks)],
                          cf, ensure_ascii=False, indent=2)

            if points:
                self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                logger.info(f"[✓] {len(points)} vektör yüklendi: {title}")
            else:
                logger.warning(f"[!] Vektör bulunamadı: {title}")


# --- main ---
if __name__ == "__main__":
    '''
        titles = [
            "Python (programlama dili)", "Yapay zeka", "Makine öğrenmesi",
            "Derin öğrenme", "Doğal dil işleme", "Veri bilimi", "Blockchain",
            "Siber güvenlik", "Kriptografi", "Robotik", "Bilgisayarla görme",
            "Sinir ağları", "Programlama dilleri", "Java (programlama dili)",
            "C++", "Yazılım mühendisliği", "Algoritma", "Veri yapıları",
            "İnternet tarihi", "Matematik", "Fizik", "Kimya", "Biyoloji",
            "Ekonomi", "Psikoloji", "Felsefe", "Tarih", "Coğrafya", "Sağlık",
            "Tıp", "Genetik", "Astronomi", "Uzay araştırmaları", "Çevre bilimi",
            "Eğitim teknolojileri", "Yapay zeka etiği", "Oyun teorisi",
            "Yapay sinir ağları", "Derin öğrenme", "Veri madenciliği",
            "Doğal dil üretimi", "TensorFlow", "PyTorch", "Keras", "Google",
            "NASA", "CERN", "YouTube", "Django", "OpenAI", "Gemini (yapay zeka)"
        ]

        fetcher = WikiFetcher()
        for title in titles:
            fetcher.fetch_and_save(title)
    '''
    
    ingestor = DocumentIngestor()
    ingestor.process_and_store()
    
    
    #    interactive_query()
