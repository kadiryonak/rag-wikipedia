Turkish RAG System - Wikipedia Integration
A Retrieval-Augmented Generation (RAG) based question-answering and information retrieval system, integrated with and optimized for Turkish Wikipedia content.


<img width="1155" height="775" alt="Ekran görüntüsü 2025-09-16 034623" src="https://github.com/user-attachments/assets/b1f3c88c-6c57-43d8-9067-f15310cefa45" />


📋 Overview
This system is a RAG (Retrieval-Augmented Generation) pipeline that allows users to ask questions in natural language and receive AI-powered, intelligent answers based on Turkish Wikipedia content. It combines semantic search with large language models to provide accurate, context-aware responses.

✨ Features
🌍 Turkish Language Support: Fully optimized for the Turkish language.

📚 Wikipedia Integration: Works seamlessly with Turkish Wikipedia data dumps.

🤖 Multi-Model Support: Choose from DeepSeek, OpenAI, Ollama, or a local fallback model.

⚡ Real-Time Semantic Search: Vector-based similarity search for retrieving relevant information.

🎯 Context-Aware Answers: Provides precise answers that cite their sources.

🌐 Web Interface: User-friendly Flask-based web interface.

🔧 REST API: Full-featured API for programmatic access.

🚀 Quick Start
Prerequisites
Python 3.8 or higher

pip (Python package manager)

Internet connection (for API-based models like DeepSeek/OpenAI)

Installation & Setup
Clone the repository:

git clone https://github.com/kadiryonak/rag-wikipedia.git
cd rag-wikipedia
Install required packages:

pip install -r requirements.txt
Configure your API Key (Optional for DeepSeek/OpenAI):
Edit the apiKey.py file or set an environment variable:


# Edit apiKey.py or use:
export DEEPSEEK_API_KEY="sk-your-actual-key-here"
Run the system:

python run_system.py
The script will guide you through the options: starting the web interface, interactive test mode, or system check.

🎮 Usage
Web Interface
By default, the web interface will start at http://localhost:5000.

Open your browser and go to http://localhost:5000.

Type your question into the text box.

Click the "Soru Sor" (Ask Question) button.

View the generated answer and the source documents it was derived from.

Command Line Interface
For a direct terminal-based interaction:


python main.py
API Usage
You can interact with the system directly via its REST API:


# Check system status
curl http://localhost:5000/status

# Ask a question
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Algoritma nedir?"}'
🏗️ Architecture
Core Components
Data Layer: Processes structured data from JSON files (e.g., Wikipedia dumps).

Vector Database: ChromaDB for storing and querying text embeddings.

Embedding Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 for creating vector representations of Turkish text.

LLM Integration: Orchestrator for multiple LLM providers (DeepSeek, OpenAI, Ollama, Local).

Web Service: Flask backend serving a REST API and a modern frontend.

Workflow
Data Ingestion: JSON files are loaded and parsed into documents.

Embedding Creation: Text is split into chunks and converted into vector embeddings.

Query Processing: The user's question is converted into an embedding.

Semantic Search: The system finds the most relevant text chunks based on vector similarity.

Answer Generation: A prompt containing the question and context is sent to the chosen LLM to generate a coherent answer.

Response Delivery: The final answer and source documents are presented to the user.

⚙️ Configuration
Model Options
The system can be configured to use one of four LLM providers:

DeepSeek (Recommended): High-quality Turkish responses using the deepseek-chat model. Requires an API key.

OpenAI: Uses GPT models (e.g., gpt-3.5-turbo). Requires an OpenAI API key.

Ollama: For use with locally running Ollama models (e.g., llama2:7b).

Local: A basic, rule-based fallback model that requires no API calls.

Data Preparation
Place your Turkish Wikipedia JSON data dumps in the ./data/ directory. The system will automatically process all .json files it finds. You can use the sample data created by the system on first run as a template.

📁 Project Structure
text
rag-wikipedia/
├── backend.py          # Flask server & API endpoints
├── run_system.py       # Main launcher & management script
├── main.py             # Core RAG system logic (TurkishRAGSystem class)
├── apiKey.py           # API key configuration (gitignored)
├── scripts.html        # Web interface frontend
├── data/               # Directory for JSON data files (e.g., Wikipedia)
├── chroma_db/          # Persistent storage for vector database
├── requirements.txt    # Python dependencies
└── README.md           # This file
🔮 Future Enhancements
Direct Wikipedia API integration for real-time fetching.

Enhanced support for more Turkish embedding models.

Advanced chunking strategies for better context retrieval.

User authentication and session management.

Deployment guides for Docker and cloud platforms.

👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙋‍♂️ Support
If you have any questions or run into issues, please open an issue on this GitHub repository.





