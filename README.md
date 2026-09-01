# 🎬 AI Video Assistant

> **Transform any video or audio into structured intelligence — transcribe, summarize, extract insights, and chat with your content using AI.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?logo=langchain)](https://www.langchain.com/)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-mistral--small-orange)](https://mistral.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ **Audio Transcription** | Local Whisper model for English; Sarvam AI for Hinglish |
| 📝 **Smart Summarization** | Map-reduce summarization with Mistral LLM |
| ✅ **Action Item Extraction** | Automatically surfaces tasks and follow-ups |
| 🔑 **Key Decision Extraction** | Highlights decisions made during the meeting |
| ❓ **Open Question Detection** | Flags unresolved questions from the transcript |
| 💬 **RAG-powered Q&A** | Chat with your video/meeting using a full RAG pipeline |
| 🌐 **YouTube Support** | Paste a YouTube URL — audio is downloaded automatically |
| 📄 **Export** | Save results as PDF or TXT |
| 🖥️ **Streamlit UI** | Beautiful, dark-themed web interface |

---

## 🏗️ Architecture

```
AI-Video-Assistant/
├── app.py                  # Streamlit web app (UI entry point)
├── main.py                 # CLI entry point
├── Requirements.txt        # Python dependencies
│
├── core/
│   ├── transcriber.py      # Whisper (English) + Sarvam AI (Hinglish) STT
│   ├── summarizer.py       # Map-reduce summarization via Mistral LLM
│   ├── extractor.py        # Action items, key decisions, open questions
│   ├── rag_engine.py       # LangChain LCEL RAG pipeline (Mistral + ChromaDB)
│   └── vector_store.py     # ChromaDB vector store builder & retriever
│
└── utils/
    └── audio_processor.py  # YouTube download (yt-dlp) + audio chunking (pydub)
```

### Pipeline Flow

```
Input (YouTube URL / Local File)
        │
        ▼
  Audio Extraction (yt-dlp / ffmpeg)
        │
        ▼
  Audio Chunking (pydub)
        │
        ▼
  Transcription
  ├── English  → Whisper (local model)
  └── Hinglish → Sarvam AI (translates to English)
        │
        ▼
  Mistral LLM Processing
  ├── Summary (Map-Reduce)
  ├── Action Items
  ├── Key Decisions
  └── Open Questions
        │
        ▼
  RAG Pipeline (ChromaDB + HuggingFace Embeddings)
        │
        ▼
  Interactive Q&A Chat
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **FFmpeg** installed and available in your system `PATH`
  - Windows: `winget install ffmpeg` or [ffmpeg.org](https://ffmpeg.org/download.html)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### 1. Clone the Repository

```bash
git clone https://github.com/risitasutar/AI-Video-Assistant.git
cd AI-Video-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r Requirements.txt
```

> ⚠️ Installing `torch` and `openai-whisper` may take several minutes depending on your internet speed.

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
# Required for LLM summarization and RAG Q&A
MISTRAL_API_KEY=your_mistral_api_key_here

# Required ONLY for Hinglish transcription
SARVAM_API_KEY=your_sarvam_api_key_here

# Optional: change Whisper model size (tiny / base / small / medium / large)
WHISPER_MODEL=small

# Optional: change Sarvam STT model
SARVAM_STT_MODEL=saaras:v2.5
```

Get your API keys:
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/)
- **Sarvam AI** *(Hinglish only)*: [sarvam.ai](https://www.sarvam.ai/)

---

## 🖥️ Usage

### Web App (Streamlit)

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Workflow in the UI:**
1. Paste a **YouTube URL** or upload a **local audio/video file**
2. Select the language: **English** or **Hinglish**
3. Click **Analyze** and wait for processing
4. Browse the generated **Summary**, **Action Items**, **Key Decisions**, and **Open Questions**
5. Use the **Chat** tab to ask any question about the content

---

### CLI (Command Line)

```bash
python main.py
```

Follow the prompts:
```
Enter YouTube URL or local file path: https://www.youtube.com/watch?v=...
Language (english/hinglish): english
```

After processing, results are printed to the terminal and you enter an interactive Q&A loop.

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | *(required)* | Mistral API key for LLM features |
| `SARVAM_API_KEY` | *(required for Hinglish)* | Sarvam API key for Hinglish transcription |
| `WHISPER_MODEL` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `SARVAM_STT_MODEL` | `saaras:v2.5` | Sarvam STT model version |

> **Whisper Model Tradeoffs**: Larger models are more accurate but slower and require more RAM. `small` is a good balance for most use cases.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `openai-whisper` | Local speech-to-text (English) |
| `yt-dlp` | YouTube audio downloading |
| `pydub` / `ffmpeg-python` | Audio processing & chunking |
| `langchain` + `langchain-mistralai` | LLM orchestration (LCEL) |
| `mistralai` | Mistral AI API client |
| `chromadb` | Local vector store for RAG |
| `sentence-transformers` | HuggingFace embeddings |
| `streamlit` | Web UI framework |
| `reportlab` / `fpdf2` | PDF export |
| `python-dotenv` | `.env` file loading |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Risita Sutar**
- GitHub: [@risitasutar](https://github.com/risitasutar)

---

<p align="center">Made with ❤️ using Python, LangChain, and Mistral AI</p>
