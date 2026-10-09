"""YouTube transcripts via the Supadata Transcript API, with VAANI's audio pipeline as fallback.

Supadata returns a video's existing YouTube captions (or an AI transcript when none exist), so
YouTube URLs can be analysed without downloading audio - which YouTube often blocks (HTTP 403)
from cloud servers such as Streamlit Community Cloud.

API: GET https://api.supadata.ai/v1/transcript?url=<video url>   header: x-api-key
Key: SUPADATA_API_KEY (environment / .env locally, Streamlit secrets on Cloud). Never logged.
"""
import html
import os
import re
import time

import requests

from utils.audio_processor import process_input, cleanup_files
from core.transcriber import transcribe_all
from utils.errors import UserFacingError

SUPADATA_URL = "https://api.supadata.ai/v1/transcript"
REQUEST_TIMEOUT = (10, 60)     # seconds: (connect, read)
POLL_INTERVAL = 2              # seconds between job polls (free plan allows 1 request/second)
POLL_MAX_WAIT = 120            # seconds before giving up on an asynchronous transcript job

# Readable messages for Supadata's documented status codes
_STATUS_MESSAGES = {
    400: "Supadata could not process this URL.",
    401: "the Supadata API key was rejected (check SUPADATA_API_KEY).",
    402: "the Supadata plan limit was reached (upgrade required).",
    403: "the Supadata API key is not allowed to use this endpoint (check SUPADATA_API_KEY).",
    404: "Supadata could not find this video.",
    206: "no transcript is available for this video.",
    429: "the Supadata rate limit or monthly credits were exceeded.",
}


class SupadataError(Exception):
    """A Supadata failure with a message that is safe to show to users (never contains the key)."""


def is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _normalize(content) -> str:
    """Join Supadata's transcript segments into one clean transcript string."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list) and all(isinstance(seg, dict) and "text" in seg for seg in content):
        text = " ".join(str(seg["text"]) for seg in content)
    else:
        raise SupadataError("Supadata returned an unexpected response format.")
    text = html.unescape(text)                 # captions contain entities such as &#39;
    return re.sub(r"\s+", " ", text).strip()


def _json(response) -> dict:
    try:
        data = response.json()
    except ValueError:
        raise SupadataError("Supadata returned an unexpected (non-JSON) response.") from None
    if not isinstance(data, dict):
        raise SupadataError("Supadata returned an unexpected response format.")
    return data


def _check_status(response):
    if response.status_code in (200, 202):
        return
    message = _STATUS_MESSAGES.get(response.status_code)
    if message is None:
        message = ("the Supadata service had an internal error." if response.status_code >= 500
                   else f"Supadata returned HTTP {response.status_code}.")
    raise SupadataError(message)


def _get(url: str, params: dict, key: str):
    try:
        return requests.get(url, params=params, headers={"x-api-key": key}, timeout=REQUEST_TIMEOUT)
    except requests.Timeout:
        raise SupadataError("the request to Supadata timed out.") from None
    except requests.RequestException:
        raise SupadataError("could not connect to Supadata.") from None


def fetch_youtube_transcript(url: str) -> str:
    """Return the video's English transcript as plain text, or raise SupadataError."""
    key = (os.getenv("SUPADATA_API_KEY") or "").strip()
    if not key:
        raise SupadataError("SUPADATA_API_KEY is not configured.")

    response = _get(SUPADATA_URL, {"url": url, "lang": "en"}, key)
    _check_status(response)
    data = _json(response)

    if response.status_code == 202:  # long video: transcript is generated asynchronously
        job_id = data.get("jobId")
        if not job_id:
            raise SupadataError("Supadata returned an unexpected response format.")
        deadline = time.monotonic() + POLL_MAX_WAIT
        while True:
            time.sleep(POLL_INTERVAL)
            job = _get(f"{SUPADATA_URL}/{job_id}", {}, key)
            _check_status(job)
            data = _json(job)
            status = data.get("status")
            if status == "completed":
                break
            if status == "failed":
                raise SupadataError("Supadata could not generate a transcript for this video.")
            if time.monotonic() > deadline:
                raise SupadataError("Supadata took too long to prepare the transcript.")

    if "content" not in data:
        raise SupadataError("Supadata returned an unexpected response format.")
    transcript = _normalize(data["content"])
    if not transcript:
        raise SupadataError("Supadata returned an empty transcript.")
    lang = str(data.get("lang") or "")
    if lang and not lang.lower().startswith("en"):
        # VAANI's analysis works on English text (Whisper English / Sarvam translate); the audio
        # pipeline handles other languages (e.g. Hinglish via Sarvam) instead.
        raise SupadataError(f"only a '{lang}' transcript is available (VAANI needs English).")
    return transcript


def get_transcript_for_source(source: str, language: str, on_step=None) -> tuple:
    """Transcript for a URL or local file path typed into VAANI.

    YouTube/other URL: Supadata first; if it fails, the existing audio pipeline (yt-dlp + Whisper/Sarvam).
    Local file path: the existing audio pipeline only.
    Returns (transcript, method) where method is "supadata" or "audio".
    on_step(key, state) reports progress with the app's step keys ("audio", "transcript").
    """
    on_step = on_step or (lambda key, state: None)
    supadata_error = None
    if is_url(source):
        try:
            transcript = fetch_youtube_transcript(source)
            print(f"Transcript from Supadata ({len(transcript)} chars); skipping audio download and transcription.")
            on_step("audio", "done")
            on_step("transcript", "done")
            return transcript, "supadata"
        except SupadataError as e:
            supadata_error = e
            print(f"Supadata transcript unavailable ({e}); falling back to audio download.")

    chunks = []
    try:
        chunks = process_input(source)
        on_step("audio", "done")
        on_step("transcript", "active")
        transcript = transcribe_all(chunks, language)
        on_step("transcript", "done")
        return transcript, "audio"
    except Exception as e:
        if supadata_error is None:
            raise  # unchanged behaviour when Supadata was not involved
        audio_reason = str(e) if isinstance(e, UserFacingError) else "the audio could not be downloaded or processed."
        raise UserFacingError(
            f"Could not get a transcript for this YouTube video. Transcript service: {supadata_error} "
            f"Audio download: {audio_reason} Please download the video or audio file yourself and use "
            "'Upload a video or audio file' in the sidebar instead."
        ) from e
    finally:
        cleanup_files(chunks)
