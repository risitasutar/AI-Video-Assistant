import whisper
import os
import time
import requests
from pydub import AudioSegment
from core.model_loading import MODEL_LOAD_LOCK
from utils.errors import UserFacingError

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
# saaras:v3 / v4 → /speech-to-text with mode="translate" (Hinglish speech → English text).
# Legacy saaras:v2.5 (deprecated by Sarvam) only works on /speech-to-text-translate.
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

_model = None


def load_model():

    global _model  

    if _model is None: 
        with MODEL_LOAD_LOCK:  # never load concurrently with another model / another session
            if _model is None:
                print(f"Loading Whisper model: {WHISPER_MODEL} ...")
                _model = whisper.load_model(WHISPER_MODEL) 
                print("Whisper model loaded.")
    return _model 


def transcribe_chunk_whisper(chunk_path: str) -> str:

    model = load_model()  

    # Whisper is only used when the user selected English, so don't let it guess the language
    # from the first 30 s (music/noise intros can be misdetected). fp16 is unavailable on CPU anyway.
    result = model.transcribe(chunk_path, task="transcribe", language="en", fp16=False)
    # Whisper "hears" words such as "you" in silence/music. Drop segments it itself rates as
    # probably-not-speech; measured real speech scores no_speech <= 0.06 and logprob >= -0.27.
    return "".join(seg["text"] for seg in result["segments"]
                   if not (seg["no_speech_prob"] > 0.6 and seg["avg_logprob"] < -0.5))


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    if SARVAM_MODEL.startswith("saaras:v2"):
        url, data = SARVAM_STT_TRANSLATE_URL, {"model": SARVAM_MODEL, "with_diarization": "false"}
    else:
        url, data = SARVAM_STT_URL, {"model": SARVAM_MODEL, "mode": "translate"}

    for attempt in range(3):
        with open(piece_path, "rb") as f:
            files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
            response = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        if response.status_code != 429:  # 429 = rate limited → back off and retry
            break
        time.sleep(2 ** (attempt + 1))

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        if response.status_code in (401, 403):
            raise UserFacingError("Sarvam API key was rejected. Check SARVAM_API_KEY.")
        raise UserFacingError(f"Sarvam transcription failed (HTTP {response.status_code}). See server logs for details.")

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise UserFacingError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms].set_channels(1).set_frame_rate(16000)  # Sarvam recommends 16 kHz mono
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

   



def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:

    full_transcript = "" 

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)  

        full_transcript += text + " "  

    print("Transcription complete.")

    if not full_transcript.strip():
        raise UserFacingError("No speech was detected in this audio, so there is nothing to analyse. "
                           "Check that the file or video contains spoken audio.")

    return full_transcript.strip()  
