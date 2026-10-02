# AI Meeting Intelligence Assistant

A standalone Streamlit demo that runs transcription, meeting analysis,
embeddings, and transcript question answering locally. It makes no OpenAI API
calls and does not require an API key or an external vector database.

## Project structure

```text
app.py
services/
  transcription.py
  llm.py
  rag.py
  settings.py
.streamlit/
  config.toml
```

The RAG index is stored in Streamlit session state. Transcript chunks and user
questions are embedded locally, ranked by cosine similarity, and only relevant
chunks are sent to the local language model.

## Requirements

- Python 3.10 or newer
- FFmpeg is optional; local transcription reads MP3, WAV, M4A, and MP4
- Ollama installed and running
- Enough disk space and memory for the local models

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install Ollama from [ollama.com/download](https://ollama.com/download), then
open a new PowerShell window and download the language and embedding models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

The Whisper `small` model is downloaded by faster-whisper the first time the
app transcribes audio. Initial setup requires internet access to download the
models; after downloads are cached, inference runs locally without paid API
calls. The default Whisper and LLM models run on CPU, though processing speed
depends on your computer.

## Run

```powershell
streamlit run app.py
```

Upload an MP3, WAV, M4A, or MP4 recording up to 1 GB. Transcription uses
faster-whisper `small`; meeting analysis and Q&A use Ollama `llama3.2:3b`;
RAG embeddings use `nomic-embed-text`.

The Ollama model names can be changed in `services/settings.py`. The Streamlit
upload limit is set to 1 GB in `.streamlit/config.toml`.

## Privacy and API keys

The app does not read or need `OPENAI_API_KEY`. Audio and transcript content
are processed by local models. No API key should be added to this project.
Keep `.streamlit/secrets.toml` ignored by Git if you use secrets for another
local purpose.

## Sharing and deployment

GitHub stores the source code; it does not run Ollama or the local Whisper
model. Streamlit Community Cloud cannot use the Ollama installation on your
personal computer. A hosted deployment must have Ollama, both Ollama models,
the faster-whisper model, Python dependencies, and enough memory/storage
installed on the server.

For a student demo without a paid inference API, the simplest option is to run
the app and models on your computer and share it temporarily through a secure
tunnel. Your computer must stay on and connected; the tunnel link is temporary.
Do not expose real or confidential meeting recordings.

Before pushing to GitHub:

1. Create a repository on GitHub.
2. Keep `.streamlit/secrets.toml` and recordings out of Git; `.gitignore`
   excludes them.
3. Review `git status` and confirm no keys, recordings, or model files are
   staged.
4. Push the source. Follow the README setup steps on any computer or server
   that will run the demo.
