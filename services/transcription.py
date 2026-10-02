from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from services.settings import (
    MAX_UPLOAD_SIZE,
    SUPPORTED_AUDIO_TYPES,
    WHISPER_MODEL,
)


@lru_cache(maxsize=1)
def _load_whisper_model(model_size: str) -> WhisperModel:
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(audio_path: str | Path) -> str:
    """Transcribe audio locally with faster-whisper."""
    source_path = Path(audio_path)
    extension = source_path.suffix.lower().lstrip(".")
    if extension not in SUPPORTED_AUDIO_TYPES:
        raise ValueError(
            "Unsupported file type. Upload an MP3, WAV, M4A, or MP4 file."
        )
    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise ValueError("The uploaded recording is empty.")
    if source_path.stat().st_size > MAX_UPLOAD_SIZE:
        raise ValueError("The recording exceeds the 1 GB upload limit.")

    try:
        model = _load_whisper_model(WHISPER_MODEL)
        segments, _ = model.transcribe(
            str(source_path),
            beam_size=5,
            vad_filter=True,
        )
        transcript = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
    except Exception as error:
        raise RuntimeError(
            "Local transcription failed. Confirm faster-whisper dependencies "
            "are installed, the first-run model download completed, and the "
            "audio file can be decoded."
        ) from error

    if not transcript:
        raise ValueError(
            "The recording produced an empty transcript. Check that it "
            "contains clear speech."
        )
    return transcript
