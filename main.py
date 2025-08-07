from scripts.fetch_wiki import WikiFetcher
from app.ingest import DocumentIngestor
from app.config import Settings

if __name__ == "__main__":
    
    
    fetcher = WikiFetcher()
    # Buraya istediğin Wikipedia sayfa başlığını yazabilirsin
    fetcher = WikiFetcher()
    titles = [
        "Python (programlama dili)",
        "Yapay zeka",
        "Makine öğrenmesi",
        "Derin öğrenme",
        "Doğal dil işleme",
        "Veri bilimi",
        "Blockchain",
        "Siber güvenlik",
        "Kriptografi",
        "Robotik",
        "Bilgisayarla görme",
        "Sinir ağları",
        "Programlama dilleri",
        "Java (programlama dili)",
        "C++",
        "Yazılım mühendisliği",
        "Algoritma",
        "Veri yapıları",
        "İnternet tarihi",
        "Matematik",
        "Fizik",
        "Kimya",
        "Biyoloji",
        "Ekonomi",
        "Psikoloji",
        "Felsefe",
        "Tarih",
        "Coğrafya",
        "Sağlık",
        "Tıp",
        "Genetik",
        "Astronomi",
        "Uzay araştırmaları",
        "Çevre bilimi",
        "Eğitim teknolojileri",
        "Yapay zeka etiği",
        "Oyun teorisi",
        "Yapay sinir ağları",
        "Derin öğrenme",
        "Veri madenciliği",
        "Doğal dil üretimi",
        "TensorFlow",
        "PyTorch",
        "Keras",
        "Google",
        "NASA",
        "CERN",
        "YouTube",
        "Django",
        "OpenAI",
        "Gemini (yapay zeka)"
    ]


    for title in titles:
        fetcher.fetch_and_save(title)
    
    
    
    ingestor = DocumentIngestor()
    ingestor.process_and_store()
