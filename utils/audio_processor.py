import yt_dlp
from pydub import AudioSegment
import os
import shutil
import time
import uuid
from utils.errors import UserFacingError

# Anchored to the project root so it works regardless of the current working directory
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'downloades')
os.makedirs(DOWNLOAD_DIR,exist_ok = True)


def sweep_stale_files(max_age_hours: float = 6):
    """Delete leftovers of runs that were killed before their cleanup ran (crash / hard stop).
    DOWNLOAD_DIR only holds this app's temp files; the age limit protects in-progress runs."""
    cutoff = time.time() - max_age_hours * 3600
    for name in os.listdir(DOWNLOAD_DIR):
        path = os.path.join(DOWNLOAD_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


sweep_stale_files()


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise UserFacingError(
            "FFmpeg is not available. Install it and make sure 'ffmpeg' is on your PATH "
            "(Streamlit Cloud: add 'ffmpeg' to packages.txt)."
        )


def download_youtube_audio(url :str) ->str:
    # Unique, filesystem-safe name (video id, not title) so special characters in
    # titles can't break paths and concurrent sessions don't overwrite each other
    output_path = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4().hex[:8]}_%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    # yt-dlp needs a JavaScript runtime (Deno) to solve YouTube's challenges, otherwise
    # downloads fail with HTTP 403. The pip 'deno' package may not be on PATH, so pass its path.
    try:
        import deno
        ydl_opts["js_runtimes"] = {"deno": {"path": deno.find_deno_bin()}}
    except Exception:
        pass  # fall back to a system-wide deno on PATH, if any
    # YouTube intermittently answers 403 for a stream URL; each attempt re-extracts fresh URLs,
    # and a retry usually succeeds. Other errors (unavailable/private video...) fail immediately.
    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
            break
        except yt_dlp.utils.DownloadError as e:
            if "403" in str(e) and attempt < 2:
                print(f"YouTube returned 403, retrying ({attempt + 2}/3)...")
                time.sleep(2)
                continue
            reason = str(e).replace("ERROR: ", "").strip()
            message = f"Unable to download audio from this YouTube URL. {reason}"
            if "403" in str(e):  # still blocked after all retries
                message += (" YouTube may block downloads from cloud-hosted servers. Please download the "
                            "video or audio file yourself and use 'Upload a video or audio file' in the "
                            "sidebar instead.")
            raise UserFacingError(message) from e
    if not os.path.exists(filename):
        raise UserFacingError("Unable to download audio from this YouTube URL (no audio file was produced).")
    return filename



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)

    return chunks


def cleanup_files(paths: list):
    """Delete generated temp files (chunks etc.); ignore files already gone."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"Could not delete temp file {path}: {e}")


def process_input(source: str) -> list:
    check_ffmpeg()
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    try:
        chunks = chunk_audio(wav_path)
    finally:
        # The full-length WAV is generated by us (never the user's original file);
        # only the chunks are needed from here on
        cleanup_files([wav_path])
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks

