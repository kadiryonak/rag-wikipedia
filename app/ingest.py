import logging
import json
from pathlib import Path
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from config import settings

nltk.download("punkt")  

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class DocumentIngestor:
    def __init__(
        self,
        data_dir: str = None,
        chunk_dir: str = "chunks",
        qdrant_host: str = "localhost",
        qdrant_port: int = ****,
        collection_name: str = "wiki_collection",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_token_limit: int = 500,
        chunk_overlap: int = 50,
    ):
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.chunk_dir = Path(chunk_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

        self.embed_model = SentenceTransformer(embed_model_name)
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.chunk_token_limit = chunk_token_limit
        self.chunk_overlap = chunk_overlap

        self._init_collection()

    def _init_collection(self):
        from qdrant_client.http.models import Distance, VectorParams

        if not self.qdrant.get_collection(self.collection_name, ignore_missing=True):
            logger.info(f"Koleksiyon bulunamadı, oluşturuluyor: {self.collection_name}")
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embed_model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Koleksiyon zaten mevcut: {self.collection_name}")

    def chunk_text(self, text: str) -> List[str]:
        """Cümle bazlı ve token limitli chunklama, overlap ile."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.chunk_token_limit:
                # Chunk tamamlandı
                chunks.append(" ".join(current_chunk))
                # Overlap için son cümleleri al
                overlap_sentences = current_chunk[-self.chunk_overlap:] if self.chunk_overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_sentences.copy()
                current_len = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sent)
            current_len += sent_len

        # Son kalan chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_and_store(self):
        """data klasöründeki json dosyalarını işle, chunkla, embed et ve Qdrant'a yükle."""
        try:
            for file_path in self.data_dir.glob("*.json"):
                logger.info(f"İşleniyor: {file_path.name}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                title = data.get("title", file_path.stem)
                content = data.get("content", "")

                chunks = self.chunk_text(content)

                chunk_records = []
                points = []
                for idx, chunk in enumerate(chunks):
                    meta = {
                        "title": title,
                        "chunk_index": idx,
                    }
                    chunk_record = {
                        "content": chunk,
                        "metadata": meta
                    }
                    chunk_records.append(chunk_record)

                    try:
                        vector = self.embed_model.encode(chunk).tolist()
                    except Exception as e:
                        logger.error(f"Embed hatası: {e} - Chunk index: {idx} başlık: {title}")
                        continue

                    point = PointStruct(
                        id=f"{title}_{idx}",
                        vector=vector,
                        payload=meta
                    )
                    points.append(point)

                chunk_file = self.chunk_dir / f"{title}_chunks.json"
                with open(chunk_file, "w", encoding="utf-8") as cf:
                    json.dump(chunk_records, cf, ensure_ascii=False, indent=2)

                if points:
                    self.qdrant.upsert(collection_name=self.collection_name, points=points)
                    logger.info(f"[✓] İşlendi ve kaydedildi: {title} ({len(chunks)} chunk)")
                else:
                    logger.warning(f"Embedlenmiş chunk bulunamadı: {title}")

        except Exception as e:
            logger.error(f"Genel işlem hatası: {e}")





