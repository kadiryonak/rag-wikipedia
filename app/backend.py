from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import threading
import time
from main import TurkishRAGSystem
from apiKey import deepseekapikey
app = Flask(__name__)
CORS(app)

# Global değişkenler
rag_system = None
system_ready = False
system_status = "Başlatılıyor..."
initialization_error = None

def initialize_rag_system():
    """RAG sistemini arka planda başlatır"""
    global rag_system, system_ready, system_status, initialization_error
    
    try:
        system_status = "Veri dosyaları kontrol ediliyor..."
        print("🚀 RAG sistemi başlatılıyor...")
        
        # Data klasörünü kontrol et
        if not os.path.exists("./data"):
            os.makedirs("./data")
            create_sample_data()
        
        # JSON dosyaları var mı kontrol et
        json_files = [f for f in os.listdir("./data") if f.endswith('.json')]
        if not json_files:
            create_sample_data()
        
        system_status = "Vector store oluşturuluyor..."
        
        # Model tipini belirle (API anahtarı varsa DeepSeek, yoksa local)
        api_key = os.getenv(deepseekapikey) or os.getenv('OPENAI_API_KEY')
        model_type = "local"  # Varsayılan olarak local
        
        if api_key and len(api_key) > 10:
            if os.getenv(deepseekapikey):
                model_type = "deepseek"
            elif os.getenv('OPENAI_API_KEY'):
                model_type = "openai"
        
        # RAG sistemini oluştur
        rag_system = TurkishRAGSystem(
            data_folder_path="./data",
            model_type=model_type,
            api_key=api_key
        )
        
        system_status = "Dökümanlar işleniyor..."
        rag_system.initialize_system()
        
        system_status = "Hazır"
        system_ready = True
        print("✅ RAG sistemi hazır!")
        
    except Exception as e:
        initialization_error = str(e)
        system_status = f"Hata: {str(e)}"
        system_ready = False
        print(f"❌ RAG sistemi başlatma hatası: {str(e)}")

def create_sample_data():
    """Örnek JSON verisi oluşturur"""
    sample_data = [
        {
            "id": 1,
            "konu": "Algoritma",
            "tanim": "Algoritma, belirli bir problemi çözmek için tasarlanmış adımların sıralı bir listesidir.",
            "detay": "Algoritma, bilgisayar biliminin temel taşlarından biridir. Bir problemi çözmek için izlenmesi gereken adımları tanımlar. Algoritmalar matematik, mühendislik ve günlük yaşamda yaygın olarak kullanılır.",
            "ornekler": ["Sıralama algoritmaları", "Arama algoritmaları", "Grafik algoritmaları"],
            "kategori": "Bilgisayar Bilimi"
        },
        {
            "id": 2,
            "konu": "Veri Yapıları",
            "tanim": "Veri yapıları, bilgisayar belleğinde verileri organize etme ve depolama yöntemleridir.",
            "detay": "Veri yapıları, verilerin etkili bir şekilde depolanması ve erişilmesi için kullanılan düzenlemelerdir. Array, linked list, stack, queue, tree ve graph gibi çeşitli türleri vardır.",
            "ornekler": ["Dizi (Array)", "Bağlı Liste (Linked List)", "Yığın (Stack)", "Kuyruk (Queue)"],
            "kategori": "Bilgisayar Bilimi"
        },
        {
            "id": 3,
            "konu": "Yapay Zeka",
            "tanim": "Yapay zeka, makinelerin insan benzeri zeka gerektiren görevleri yerine getirebilme yeteneğidir.",
            "detay": "Yapay zeka, machine learning, deep learning, natural language processing ve computer vision gibi alt dalları içerir. Günümüzde sağlık, finans, ulaşım ve eğitim gibi birçok alanda kullanılmaktadır.",
            "ornekler": ["Machine Learning", "Deep Learning", "Doğal Dil İşleme", "Bilgisayar Görüşü"],
            "kategori": "Teknoloji"
        },
        {
            "id": 4,
            "konu": "Python Programlama",
            "tanim": "Python, kolay öğrenilen ve güçlü bir yüksek seviye programlama dilidir.",
            "detay": "Python, temiz sözdizimi ve geniş kütüphane desteği ile veri bilimi, web geliştirme, otomasyon ve yapay zeka projelerinde yaygın olarak kullanılır. Guido van Rossum tarafından geliştirilmiştir.",
            "ornekler": ["Web geliştirme (Django, Flask)", "Veri analizi (Pandas, NumPy)", "Makine öğrenmesi (Scikit-learn)"],
            "kategori": "Programlama"
        }
    ]
    
    with open("./data/sample_knowledge.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print("✅ Örnek veri dosyası oluşturuldu: sample_knowledge.json")

# Sistem başlatma thread'i
initialization_thread = threading.Thread(target=initialize_rag_system, daemon=True)
initialization_thread.start()

@app.route('/')
def index():
    """Ana sayfa"""
    return send_from_directory('.', 'scripts.html')

@app.route('/status')
def get_status():
    """Sistem durumunu döndürür"""
    return jsonify({
        "ready": system_ready,
        "status": system_status,
        "error": initialization_error,
        "model_type": rag_system.model_type if rag_system else "unknown"
    })
@app.route('/ask', methods=['POST'])
def ask_question():
    """Soru sorma endpoint'i"""
    if not system_ready:
        return jsonify({
            "error": "Sistem henüz hazır değil",
            "status": system_status
        }), 503
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Soru boş olamaz"}), 400
        
        print(f"📝 Soru alındı: {question}")
        
        # RAG sistemi ile cevap al
        result = rag_system.ask_question(question, k=5)
        
        # Hata kontrolü ekle
        if "error" in result or result.get("answer", "").startswith("❌"):
            return jsonify({
                "error": result.get("answer", "Bilinmeyen hata"),
                "question": question
            }), 500
        
        return jsonify({
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "sources": [
                {
                    "title": doc["source"],
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                }
                for doc in result["source_documents"]
            ],
            "context_used": result.get("context_used", "")
        })
        
    except Exception as e:
        print(f"❌ Soru yanıtlama hatası: {str(e)}")
        return jsonify({
            "error": f"Soru yanıtlanırken hata oluştu: {str(e)}"
        }), 500

@app.route('/test')
def test_system():
    """Sistem test endpoint'i"""
    if not system_ready:
        return jsonify({"error": "Sistem hazır değil"}), 503
    
    test_questions = [
        "Algoritma nedir?",
        "Python hakkında ne biliyorsun?",
        "Yapay zeka ile ilgili bilgiler neler?",
        "Veri yapıları nedir?"
    ]
    
    results = []
    for question in test_questions:
        try:
            result = rag_system.ask_question(question)
            results.append({
                "question": question,
                "answer": result["answer"],
                "sources_count": len(result["source_documents"])
            })
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e)
            })
    
    return jsonify({"test_results": results})

@app.route('/data-info')
def get_data_info():

    if not system_ready:
        return jsonify({"error": "Sistem hazır değil"}), 503
    
    try:
        data_folder = r"C:\Users\w\Desktop\Kodlama\VsCode\HelloWorld\RAG\data"
        json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
        
        file_info = []
        total_documents = 0
        
        for filename in json_files:
            file_path = os.path.join(data_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    doc_count = len(data)
                else:
                    doc_count = 1
                
                file_info.append({
                    "filename": filename,
                    "document_count": doc_count,
                    "size_kb": round(os.path.getsize(file_path) / 1024, 2)
                })
                
                total_documents += doc_count
                
            except Exception as e:
                file_info.append({
                    "filename": filename,
                    "error": str(e)
                })
        
        return jsonify({
            "files": file_info,
            "total_files": len(json_files),
            "total_documents": total_documents
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🌐 Flask server başlatılıyor...")
    print("📂 Web arayüzü: http://localhost:5000")
    print("🔧 API endpoint'leri:")
    print("   - GET  /status     : Sistem durumu")
    print("   - POST /ask        : Soru sor")
    print("   - GET  /test       : Test soruları")
    print("   - GET  /data-info  : Veri bilgisi")
    
    app.run(debug=True, host='0.0.0.0', port=5000)