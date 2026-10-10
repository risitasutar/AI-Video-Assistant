# Deployment & API Key Guide

How to run **AI Video Assistant** locally and deploy it on **Streamlit Community Cloud**.

---

## 1. Configuration at a glance

| Variable | Required? | Used in | What stops working without it |
|---|---|---|---|
| `GEMINI_API_KEY` | **Mandatory** | `core/llm.py` (used by `core/summarizer.py`, `core/extractor.py`, `core/rag_engine.py`) | Title, summary, action items, key decisions, open questions, RAG chat (the app refuses to start an analysis) |
| `SUPADATA_API_KEY` | **Recommended for YouTube URLs** | `utils/supadata.py` | YouTube URLs are fetched as captions via Supadata; without it VAANI downloads the audio instead, which YouTube often blocks (HTTP 403) on Streamlit Cloud. Uploads are unaffected |
| `SARVAM_API_KEY` | **Only for Hinglish** | `core/transcriber.py` | Hinglish transcription (English/Whisper still works) |
| `GEMINI_MODEL` | Optional (default `gemini-3.5-flash-lite`) | `core/llm.py` | — any Gemini model your key can use, e.g. `gemini-2.5-flash-lite` |
| `WHISPER_MODEL` | Optional (default `small`) | `core/transcriber.py` | — (`tiny` / `base` / `small` / `medium` / `large`) |
| `SARVAM_STT_MODEL` | Optional (default `saaras:v3`) | `core/transcriber.py` | — |

Where the app reads them:

- **Locally:** a `.env` file in the project root (loaded by `python-dotenv`).
- **On Streamlit Cloud:** the app's **Secrets**. `app.py` copies them into the environment before the `core/` modules load.

Keys are never printed, logged or shown in the UI.

---

## 2. Local setup (Windows PowerShell)

Prerequisites:

- **Python 3.12**. Do not use 3.13: pydub needs the `audioop` module, which Python 3.13 removed.
- **FFmpeg** on `PATH`: `winget install Gyan.FFmpeg`, then open a new terminal and check with `ffmpeg -version`.

```powershell
cd C:\Users\hp\AI-Video-Assistant
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env      # then open .env and paste your keys
python -m streamlit run app.py   # opens http://localhost:8501
```

**If `Activate.ps1` fails with "running scripts is disabled on this system":** Windows blocks PowerShell scripts by default. Allow them for the current terminal window only, without changing any system setting:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, skip activation and call the venv's Python directly, e.g. `.\.venv\Scripts\python.exe -m streamlit run app.py`.

In VS Code, select the interpreter with **Ctrl+Shift+P → Python: Select Interpreter → `.venv`**.

---

## 3. Get the API keys

### Google Gemini (mandatory)

1. **Why the project needs it:** every LLM step runs on the Gemini model set by `GEMINI_MODEL` (default `gemini-3.5-flash-lite`) via LangChain (`langchain-google-genai`): title, map-reduce summary, the three extractors and RAG Q&A.
2. **Create a key:** sign in at <https://aistudio.google.com/apikey> with a Google account and click **Create API key**.
3. **Cost:** the Gemini API free tier lists Flash and Flash-Lite models as "Free of charge", with rate limits (requests per minute/day) shown on your [AI Studio rate-limit page](https://aistudio.google.com/rate-limit). Daily limits reset at midnight Pacific time. Do not enable billing if you want to stay free. Google's free-tier terms say content may be used to improve their products.
4. **Local:** add `GEMINI_API_KEY=...` to `.env`.
5. **Cloud:** add it to the app's Secrets (see §4).

Docs: <https://ai.google.dev/gemini-api/docs/api-key>

### Sarvam AI (only for Hinglish)

1. **Why the project needs it:** Hinglish audio is sent in 25-second pieces to Sarvam's speech-to-text-translate API, which returns English text.
2. **Create an account and a key:** <https://dashboard.sarvam.ai> → create a new API key. It is sent as the `api-subscription-key` header.
3. **Cost:** Sarvam bills in credits. Check current credits, pricing and rate limits on the dashboard; they aren't stated in the quickstart docs.
4. **Local:** add `SARVAM_API_KEY=...` to `.env`.
5. **Cloud:** add it to Secrets.
6. **Without it:** selecting *hinglish* shows "Sarvam API key is required for Hinglish transcription."

> Uses **Saaras v3** on Sarvam's `/speech-to-text` endpoint with `mode=translate` (Hinglish speech → English text), in ≤25-second pieces because the REST API accepts at most 30 s per request. Sarvam bills speech-to-text/translate at ₹30 per hour of audio and gives every new account ₹100 of free credits (≈3 hours of audio). Starter rate limit: 60 requests/min (the app retries on HTTP 429).

Docs: <https://docs.sarvam.ai/api/getting-started/models/saaras>

---

## 4. Deploy on Streamlit Community Cloud

**Before you start:**

- Commit and push these changes to GitHub.
- Use a GitHub account that has access to the repository (`risitasutar/AI-Video-Assistant`). If you don't have push access, fork it and deploy the fork.

Steps:

1. Go to <https://share.streamlit.io> and **sign in with GitHub**. Authorise Streamlit to read your repositories.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `risitasutar/AI-Video-Assistant` (or your fork)
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick any free subdomain
4. Open **Advanced settings**:
   - **Python version:** **3.12**. This is required: the Linux CPU-only torch wheel in `requirements.txt` is built for Python 3.12. The version can only be set at deploy time; to change it later you must delete and redeploy the app.
   - **Secrets:** paste the block below, with your real keys:

     ```toml
     GEMINI_API_KEY = "paste-your-gemini-key-here"
     SARVAM_API_KEY = "paste-your-sarvam-key-here"
     SUPADATA_API_KEY = "paste-your-supadata-key-here"
     WHISPER_MODEL = "small"
     SARVAM_STT_MODEL = "saaras:v3"
     ```

     - `GEMINI_API_KEY` is mandatory.
     - `SARVAM_API_KEY` is only needed for Hinglish; delete that line if you don't use it.
     - `SUPADATA_API_KEY` (from https://supadata.ai, dashboard → API key) makes YouTube URLs work on Streamlit Cloud: VAANI fetches the video's English captions instead of downloading audio. Free plan: 100 credits/month, 1 credit per video with existing captions, 2 credits per minute when Supadata has to generate a transcript. Without it, YouTube URLs fall back to audio download.
     - The last two lines are optional (those are the defaults).
5. Click **Deploy**.
   - Cloud installs `ffmpeg` from `packages.txt` and the Python packages from `requirements.txt`.
   - The first build takes several minutes, mostly for torch and Whisper.
   - The first analysis also downloads the Whisper model (~460 MB) and the embedding model (~90 MB).
6. To change secrets later: **App → ⋮ → Settings → Secrets**. Saving restarts the app.

**Never commit `.env` or `.streamlit/secrets.toml`.** Both are in `.gitignore`.

---

## 5. Post-deployment checklist

- [ ] Build log shows no dependency errors, and the app launches
- [ ] Clicking **Analyse** without input shows "Please enter a YouTube URL…" (UI works)
- [ ] Gemini configured: no "Gemini API key is not configured" message
- [ ] Sarvam configured, if Hinglish is used
- [ ] **Local upload:** a small MP3/MP4 (1–2 min) goes through the whole pipeline
- [ ] **English transcription (Whisper):** the transcript expander shows correct text
- [ ] **Summary** works
- [ ] **Action items** work
- [ ] **Key decisions** work
- [ ] **Open questions** work
- [ ] **RAG chat:** a question about the content gets an answer from the transcript, and an unrelated question gets "I could not find this information…"
- [ ] **TXT export** downloads
- [ ] **PDF export** downloads
- [ ] **YouTube URL:** a short public video works, or shows a clear "Unable to download audio…" message
- [ ] **Hinglish transcription (Sarvam)** works, if a key is configured
- [ ] No FFmpeg errors
- [ ] No API keys visible in the UI or the app logs

---

## 6. Known limitations on Community Cloud

**Resources:** about 2.7 GB RAM and up to 2 shared CPU cores (Streamlit's documented limits).
- Whisper `small` fits, but transcription is CPU-only and slow: plan on minutes per minute of audio.
- Use short videos.
- Concurrent users share one container, so expect one heavy analysis at a time.
- If the app runs out of memory, set `WHISPER_MODEL = "base"` (or `"tiny"`) in Secrets. This is faster and uses less memory, with lower accuracy.

**Storage:** the filesystem is temporary.
- Audio files and chunks are deleted after each run.
- The RAG vector store lives in memory for the current session.
- Results disappear when the session ends; use Export to keep them.

**YouTube:**
- Some videos fail because of YouTube-side restrictions: age or region locks, private videos, or bot checks on cloud IPs. The app shows "Unable to download audio from this YouTube URL" with the reason.
- Uploading the file always works.

**Other:**
- The app sleeps after 12 hours without traffic; waking it reloads the models.
- The upload size limit is 200 MB (Streamlit default).
