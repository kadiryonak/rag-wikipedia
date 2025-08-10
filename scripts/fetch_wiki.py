<<<<<<< HEAD
import wikipedia
import json
from pathlib import Path

class WikiFetcher:
    def __init__(self, save_dir: str = "data"):
        self.save_path = Path(save_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
        wikipedia.set_lang("tr")  

    def fetch_page(self, title: str) -> str:
        """Wikipedia sayfasını getirir."""
        try:
            content = wikipedia.page(title).content
            return content
        except wikipedia.exceptions.PageError:
            print(f"[HATA] Sayfa bulunamadı: {title}")
            return ""
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[HATA] Çok anlamlı başlık: {title}. Öneriler: {e.options}")
            return ""

    def save_page(self, title: str, content: str) -> Path:

        clean_title = title.replace(" ", "_")
        file_path = self.save_path / f"{clean_title}.json"

        data = {
            "title": title,
            "content": content
        }

        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[✓] Kaydedildi: {file_path}")
        return file_path

    def fetch_and_save(self, title: str) -> Path:

        content = self.fetch_page(title)
        if content:
            return self.save_page(title, content)
        return None


=======
import wikipedia
import json
from pathlib import Path

class WikiFetcher:
    def __init__(self, save_dir: str = "data"):
        self.save_path = Path(save_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
        wikipedia.set_lang("tr")  # Türkçe içerik için

    def fetch_page(self, title: str) -> str:
        """Wikipedia sayfasını getirir."""
        try:
            content = wikipedia.page(title).content
            return content
        except wikipedia.exceptions.PageError:
            print(f"[HATA] Sayfa bulunamadı: {title}")
            return ""
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[HATA] Çok anlamlı başlık: {title}. Öneriler: {e.options}")
            return ""

    def save_page(self, title: str, content: str) -> Path:
        """Sayfayı .json dosyası olarak kaydeder."""
        clean_title = title.replace(" ", "_")
        file_path = self.save_path / f"{clean_title}.json"

        data = {
            "title": title,
            "content": content
        }

        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[✓] Kaydedildi: {file_path}")
        return file_path

    def fetch_and_save(self, title: str) -> Path:
        """Tek adımda getir ve kaydet."""
        content = self.fetch_page(title)
        if content:
            return self.save_page(title, content)
        return None



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
    
    
>>>>>>> 0e3c223 (Save local changes before pull)
