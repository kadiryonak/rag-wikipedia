import json
import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import requests
import re

class TurkishRAGSystem:
    def __init__(self, data_folder_path: str, model_type: str = "deepseek", api_key: str = None):
        """
        Türkçe RAG sistemi başlatıcısı
        
        Args:
            data_folder_path: JSON dosyalarının bulunduğu klasör yolu
            model_type: "deepseek", "openai", "local", "ollama"
            api_key: API anahtarı (deepseek/openai için)
        """
        self.data_folder_path = data_folder_path
        self.model_type = model_type
        self.api_key = api_key
        
        # ChromaDB istemcisi
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Türkçe için optimize edilmiş embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Text splitter - Türkçe metinler için optimize edilmiş
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "]
        )
        
        # Vector store
        self.vector_store = None
        
        # Türkçe prompt template
        self.system_prompt = """Sen Türkçe konuşan bir yapay zeka asistanısın. Görentin verilen BAĞLAM bilgilerini kullanarak kullanıcının sorusunu yanıtla.

KURALLAR:
1. Sadece verilen BAĞLAM bilgilerini kullan
2. Bağlamda bilgi yoksa "Bu konuda verilen bilgilerde yeterli detay bulunmuyor" de
3. Kısa ve öz yanıt ver
4. Bağlamdan doğrudan alıntı yapabilirsin
5. Türkçe yanıt ver

BAĞLAM:
{context}

SORU: {question}

YANITÍN:"""

    def load_json_data(self) -> List[Document]:
        """
        Data klasöründeki tüm JSON dosyalarını yükler ve Document objelerine dönüştürür
        """
        documents = []
        
        if not os.path.exists(self.data_folder_path):
            print(f"❌ {self.data_folder_path} klasörü bulunamadı!")
            return documents
        
        # Data klasöründeki tüm JSON dosyalarını bul
        json_files = [f for f in os.listdir(self.data_folder_path) if f.endswith('.json')]
        print(f"📂 {len(json_files)} JSON dosyası bulundu: {json_files}")
        
        for filename in json_files:
            file_path = os.path.join(self.data_folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # Eğer data bir liste ise, her elemanı işle
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        doc_content = self._extract_text_from_json(item)
                        if doc_content and len(doc_content.strip()) > 10:
                            doc = Document(
                                page_content=doc_content,
                                metadata={
                                    "source": filename,
                                    "item_index": i,
                                    "file_path": file_path,
                                    "item_count": len(data)
                                }
                            )
                            documents.append(doc)
                
                # Eğer data tek bir obje ise
                else:
                    doc_content = self._extract_text_from_json(data)
                    if doc_content and len(doc_content.strip()) > 10:
                        doc = Document(
                            page_content=doc_content,
                            metadata={
                                "source": filename,
                                "file_path": file_path
                            }
                        )
                        documents.append(doc)
                        
            except Exception as e:
                print(f"❌ {filename} dosyası yüklenirken hata: {str(e)}")
        
        print(f"✅ Toplam {len(documents)} döküman yüklendi.")
        return documents

    def _extract_text_from_json(self, json_obj: Dict[str, Any], parent_key: str = "") -> str:
        """
        JSON objesinden metin içeriğini daha akıllı şekilde çıkarır
        """
        text_parts = []
        
        def clean_text(text: str) -> str:
            # Gereksiz karakterleri temizle
            text = re.sub(r'[=]{2,}', '', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()
            return text
        
        def extract_recursive(obj, current_key=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and len(value.strip()) > 2:
                        clean_value = clean_text(value)
                        if clean_value:
                            if current_key:
                                text_parts.append(f"{key}: {clean_value}")
                            else:
                                text_parts.append(f"{key}: {clean_value}")
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, str) and len(item.strip()) > 2:
                        clean_item = clean_text(item)
                        if clean_item:
                            text_parts.append(clean_item)
                    elif isinstance(item, (dict, list)):
                        extract_recursive(item, current_key)
        
        extract_recursive(json_obj, parent_key)
        
        # Tekrarları kaldır ve birleştir
        unique_parts = []
        for part in text_parts:
            if part not in unique_parts and len(part.strip()) > 5:
                unique_parts.append(part)
        
        return "\n".join(unique_parts)

    def create_vector_store(self):
        """
        Vector store oluşturur ve dökümanları yükler
        """
        print("📚 JSON veriler yükleniyor...")
        documents = self.load_json_data()
        
        if not documents:
            raise ValueError("❌ Hiç döküman yüklenemedi! 'data' klasörünü kontrol edin.")
        
        print("✂️ Metinler parçalara bölünüyor...")
        split_docs = self.text_splitter.split_documents(documents)
        
        print(f"📄 Toplam {len(split_docs)} metin parçası oluşturuldu.")
        
        # Mevcut collection'ı temizle
        try:
            self.chroma_client.delete_collection("turkish_rag_collection")
        except:
            pass
        
        print("🗄️ Vector store oluşturuluyor...")
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            client=self.chroma_client,
            collection_name="turkish_rag_collection"
        )
        
        print("✅ Vector store başarıyla oluşturuldu!")

    def search_relevant_docs(self, question: str, k: int = 5) -> List[Document]:
        """
        Soruya en uygun dökümanları bulur
        """
        if not self.vector_store:
            return []
        
        # Similarity search
        docs = self.vector_store.similarity_search(question, k=k)
        return docs

    def call_llm(self, prompt: str) -> str:
        """
        LLM'i çağırır ve yanıt alır
        """
        if self.model_type == "deepseek" and self.api_key:
            return self._call_deepseek(prompt)
        elif self.model_type == "openai" and self.api_key:
            return self._call_openai(prompt)
        elif self.model_type == "ollama":
            return self._call_ollama(prompt)
        else:
            return self._call_local_llm(prompt)

    def _call_deepseek(self, prompt: str) -> str:
        """
        DeepSeek API çağrısı
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ DeepSeek API hatası: {response.status_code}")
                return f"API hatası: {response.status_code}"
                
        except Exception as e:
            print(f"❌ DeepSeek bağlantı hatası: {str(e)}")
            return "DeepSeek'e bağlanılamadı. Local model kullanılıyor."

    def _call_openai(self, prompt: str) -> str:
        """
        OpenAI API çağrısı
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"OpenAI API hatası: {response.status_code}"
                
        except Exception as e:
            return f"OpenAI bağlantı hatası: {str(e)}"

    def _call_ollama(self, prompt: str) -> str:
        """
        Ollama API çağrısı
        """
        try:
            payload = {
                "model": "llama2:7b",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                return f"Ollama hatası: {response.status_code}"
                
        except Exception as e:
            return f"Ollama bağlantısı başarısız: {str(e)}"

    def _call_local_llm(self, prompt: str) -> str:
        """
        Basit local LLM (context-aware)
        """
        # Context ve soruyu ayır
        if "BAĞLAM:" in prompt and "SORU:" in prompt:
            context_start = prompt.find("BAĞLAM:") + 8
            context_end = prompt.find("SORU:")
            context = prompt[context_start:context_end].strip()
            
            question_start = prompt.find("SORU:") + 5
            question_end = prompt.find("YANITÍN:")
            question = prompt[question_start:question_end].strip()
            
            return self._generate_contextual_answer(context, question)
        
        return "Prompt formatı hatalı."

    def _generate_contextual_answer(self, context: str, question: str) -> str:
        """
        Context'e dayalı akıllı yanıt üretir
        """
        if not context or len(context.strip()) < 10:
            return "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."
        
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Algoritma sorusu
        if "algoritma" in question_lower:
            if "algoritma" in context_lower:
                # Context'ten algoritma ile ilgili kısmı bul
                lines = context.split('\n')
                relevant_lines = []
                for line in lines:
                    if 'algoritma' in line.lower() or 'algorithm' in line.lower():
                        relevant_lines.append(line)
                        
                if relevant_lines:
                    return f"Algoritma hakkında şu bilgiler bulundu: {' '.join(relevant_lines[:3])}"
                else:
                    # Context'te genel bilgi varsa
                    context_sample = context[:300] + "..." if len(context) > 300 else context
                    return f"Algoritma konusunda doğrudan bilgi bulunamadı, ancak şu genel bilgiler mevcut: {context_sample}"
            else:
                return "Algoritma konusunda verilen bilgilerde detay bulunmuyor."
        
        # Genel soru tipleri
        if any(word in question_lower for word in ["nedir", "ne"]):
            # Tanım arıyor
            definition_lines = []
            lines = context.split('\n')
            for line in lines:
                if ':' in line and len(line) < 200:
                    definition_lines.append(line)
            
            if definition_lines:
                return f"Tanım: {definition_lines[0]}. {definition_lines[1] if len(definition_lines) > 1 else ''}"
            else:
                context_sample = context[:200] + "..." if len(context) > 200 else context
                return f"Bu konuda şu bilgiler mevcut: {context_sample}"
        
        elif any(word in question_lower for word in ["hangi", "ne gibi", "nasıl"]):
            # Liste/örnekler arıyor
            items = []
            lines = context.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or 
                           line.startswith('*') or ':' in line):
                    items.append(line)
            
            if items:
                return f"Şunlar bulundu: {', '.join(items[:5])}"
            else:
                return f"Bu konuda şu bilgiler mevcut: {context[:250]}..."
        
        elif any(word in question_lower for word in ["önemli", "kritik", "temel"]):
            # Önemli bilgiler arıyor
            important_lines = []
            lines = context.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['önemli', 'temel', 'ana', 'başlıca', 'kritik']):
                    important_lines.append(line)
            
            if important_lines:
                return f"Önemli bilgiler: {' '.join(important_lines[:2])}"
            else:
                # İlk birkaç cümleyi önemli kabul et
                sentences = context.split('.')[:2]
                return f"Temel bilgiler: {'. '.join(sentences)}."
        
        # Varsayılan: bağlamın başından örnekle
        context_lines = context.split('\n')
        relevant_lines = [line for line in context_lines if line.strip() and len(line) > 20][:3]
        
        if relevant_lines:
            return f"Bu konuda şu bilgiler mevcut: {' '.join(relevant_lines)}"
        else:
            context_sample = context[:300] + "..." if len(context) > 300 else context
            return f"Bağlamda şu bilgiler var: {context_sample}"

    def ask_question(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        RAG sistemi ile soru sorar
        
        Args:
            question: Türkçe soru
            k: Kaç döküman getirileceği
            
        Returns:
            Yanıt ve kaynak dökümanlar
        """
        if not self.vector_store:
            return {
                "question": question,
                "answer": "❌ Vector store henüz hazır değil!",
                "source_documents": [],
                "context_used": ""
            }
        
        try:
            # 1. İlgili dökümanları bul
            print(f"🔍 '{question}' sorusu için dökümanlar aranıyor...")
            relevant_docs = self.search_relevant_docs(question, k=k)
            
            if not relevant_docs:
                return {
                    "question": question,
                    "answer": "Bu soruyla ilgili hiçbir bilgi bulunamadı.",
                    "source_documents": [],
                    "context_used": ""
                }
            
            # 2. Context oluştur
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                context_parts.append(f"[Kaynak {i} - {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # 3. Prompt oluştur
            full_prompt = self.system_prompt.format(
                context=context,
                question=question
            )
            
            print(f"🤖 LLM yanıtı üretiliyor ({self.model_type})...")
            
            # 4. LLM'den yanıt al
            answer = self.call_llm(full_prompt)
            
            # 5. Sonucu döndür
            return {
                "question": question,
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get("source", "Bilinmiyor"),
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "context_used": context[:500] + "..." if len(context) > 500 else context
            }
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"❌ Hata oluştu: {str(e)}",
                "source_documents": [],
                "context_used": ""
            }

    def initialize_system(self):
        """
        Tüm sistemi başlatır
        """
        print("🚀 RAG sistemi başlatılıyor...")
        self.create_vector_store()
        print("✅ RAG sistemi hazır!")

    def test_system(self):
        """
        Sistemi test eder
        """
        test_questions = [
            "Bu veriler hakkında ne biliyorsun?",
            "Algoritma nedir?",
            "Hangi konular ele alınıyor?",
            "Veri bilimi hakkında ne var?",
            "En önemli bilgiler nelerdir?"
        ]
        
        print("\n" + "="*60)
        print("🧪 SİSTEM TESTİ BAŞLIYOR")
        print("="*60)
        
        for question in test_questions:
            print(f"\n❓ SORU: {question}")
            print("-" * 40)
            
            result = self.ask_question(question)
            print(f"💬 YANIT: {result['answer']}")
            
            if result['source_documents']:
                print(f"\n📚 KAYNAK SAYISI: {len(result['source_documents'])}")
                for i, doc in enumerate(result['source_documents'][:2], 1):
                    print(f"   {i}. {doc['source']}: {doc['content'][:100]}...")

# Kullanım örneği
def main():
    # Konfigürasyon
    DATA_FOLDER = "./data"
    from apiKey import deepseekapikey    
    # Model seçimi
    print("🤖 Model seçenekleri:")
    print("1. Local (Hızlı, API gerekmez)")
    print("2. DeepSeek (Kaliteli, API key gerekir)")  
    print("3. OpenAI (Kaliteli, API key gerekir)")
    print("4. Ollama (Yerel, Ollama kurulumu gerekir)")
    
    choice = input("Seçiminiz (1-4, varsayılan: 1): ").strip() or "1"
    
    model_type = "local"
    api_key = None
    
    if choice == "2":
        model_type = "deepseek"
        api_key = deepseekapikey
    elif choice == "3":
        model_type = "openai"  
        api_key = input("OpenAI API key: ").strip()
    elif choice == "4":
        model_type = "ollama"
    
    try:
        # RAG sistemini başlat
        rag_system = TurkishRAGSystem(
            data_folder_path=DATA_FOLDER,
            model_type=model_type,
            api_key=api_key
        )
        
        rag_system.initialize_system()
        
        # Test çalıştır
        rag_system.test_system()
        
        # İnteraktif mod
        print("\n" + "="*60)
        print("💬 İNTERAKTİF MOD (çıkmak için 'exit')")
        print("="*60)
        
        while True:
            question = input("\n❓ Sorunuz: ").strip()
            if question.lower() in ['exit', 'çıkış', 'quit']:
                break
                
            if question:
                result = rag_system.ask_question(question)
                print(f"\n💬 {result['answer']}")
        
    except Exception as e:
        print(f"❌ Sistem hatası: {str(e)}")

if __name__ == "__main__":
    main()