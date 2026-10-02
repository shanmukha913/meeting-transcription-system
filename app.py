import tempfile
from pathlib import Path

import streamlit as st

from services.llm import analyze_meeting, answer_from_context
from services.rag import (
    build_transcript_index,
    retrieve_relevant_chunks,
)
from services.settings import (
    MAX_UPLOAD_SIZE,
    SUPPORTED_AUDIO_TYPES,
)
from services.transcription import transcribe_audio


st.set_page_config(
    page_title="AI Meeting Intelligence Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Meeting Intelligence Assistant")
st.write(
    "Transcribe a recording, review meeting highlights, and ask questions "
    "about what was discussed."
)
st.caption("Your recording is processed by AI models running on this computer.")

for key, default in (
    ("transcript", ""),
    ("meeting_intelligence", None),
    ("rag_index", []),
    ("answer", ""),
    ("answer_sources", []),
    ("upload_id", None),
):
    if key not in st.session_state:
        st.session_state[key] = default

st.header("Upload a recording")
uploaded_file = st.file_uploader(
    "Choose an audio or video file",
    type=list(SUPPORTED_AUDIO_TYPES),
)
if uploaded_file is None:
    st.caption("Supported formats: MP3, WAV, M4A, MP4 · Maximum size: 1 GB")

upload_id = (
    (uploaded_file.name, uploaded_file.size) if uploaded_file else None
)
if upload_id != st.session_state.upload_id:
    st.session_state.transcript = ""
    st.session_state.meeting_intelligence = None
    st.session_state.rag_index = []
    st.session_state.answer = ""
    st.session_state.answer_sources = []
    st.session_state.upload_id = upload_id

if uploaded_file:
    extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_AUDIO_TYPES:
        st.error("Unsupported file type. Upload an MP3, WAV, M4A, or MP4 file.")
    else:
        st.audio(uploaded_file)
        if st.button("Transcribe meeting", type="primary"):
            if uploaded_file.size == 0:
                st.error("The uploaded recording is empty.")
            elif uploaded_file.size > MAX_UPLOAD_SIZE:
                st.error(
                    "The recording exceeds the 1 GB upload limit."
                )
            else:
                st.session_state.transcript = ""
                st.session_state.meeting_intelligence = None
                st.session_state.rag_index = []
                st.session_state.answer = ""
                st.session_state.answer_sources = []

                stage = "transcription"
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="meeting-upload-"
                    ) as temp_dir:
                        safe_filename = Path(uploaded_file.name).name
                        audio_path = Path(temp_dir) / safe_filename
                        with audio_path.open("wb") as audio_stream:
                            audio_stream.write(uploaded_file.getbuffer())

                        with st.status(
                            "Working on your meeting...", expanded=False
                        ) as status:
                            transcript = transcribe_audio(audio_path)
                            st.session_state.transcript = transcript

                            status.update(
                                label="Preparing question search...",
                                state="running",
                            )
                            try:
                                st.session_state.rag_index = (
                                    build_transcript_index(transcript)
                                )
                            except Exception as error:
                                st.warning(
                                    "The transcript is ready, but question "
                                    "search is unavailable. "
                                    f"({type(error).__name__}: {error})"
                                )

                            stage = "meeting analysis"
                            status.update(
                                label="Analyzing the meeting...",
                                state="running",
                            )
                            st.session_state.meeting_intelligence = (
                                analyze_meeting(transcript)
                            )
                            status.update(
                                label="Meeting analysis is ready.",
                                state="complete",
                            )
                except Exception as error:
                    st.error(
                        f"{stage.capitalize()} failed "
                        f"({type(error).__name__}): {error}"
                    )

if st.session_state.transcript:
    st.divider()
    st.header("Transcript")
    st.text_area(
        "Generated transcript",
        value=st.session_state.transcript,
        height=260,
        disabled=True,
    )

if st.session_state.meeting_intelligence:
    intelligence = st.session_state.meeting_intelligence
    st.divider()
    st.header("Meeting highlights")
    st.subheader("Summary")
    st.write(intelligence["summary"])

    left, right = st.columns(2)
    with left:
        st.subheader("Key discussion points")
        if intelligence["key_points"]:
            for point in intelligence["key_points"]:
                st.markdown(f"- {point}")
        else:
            st.write("No key discussion points identified.")
    with right:
        st.subheader("Important decisions")
        if intelligence["decisions"]:
            for decision in intelligence["decisions"]:
                st.markdown(f"- {decision}")
        else:
            st.write("No decisions identified.")

    st.subheader("Action items")
    if intelligence["action_items"]:
        st.dataframe(
            intelligence["action_items"],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No action items identified.")

if st.session_state.transcript:
    st.divider()
    st.header("Ask about this meeting")
    if not st.session_state.rag_index:
        st.info(
            "Question search is unavailable right now. You can still read "
            "the transcript above."
        )

    with st.form("meeting_question_form"):
        question = st.text_input(
            "What would you like to know?",
            placeholder="What decisions were made?",
        )
        submitted = st.form_submit_button("Get answer", type="primary")

    if submitted:
        st.session_state.answer = ""
        st.session_state.answer_sources = []
        if not question.strip():
            st.warning("Enter a question to continue.")
        elif not st.session_state.rag_index:
            st.error(
                "I can't search this transcript right now. Try processing "
                "the recording again."
            )
        else:
            stage = "question embedding and retrieval"
            try:
                with st.spinner("Searching the transcript..."):
                    sources = retrieve_relevant_chunks(
                        question.strip(),
                        st.session_state.rag_index,
                    )
                    st.session_state.answer_sources = sources

                if not sources:
                    st.session_state.answer = (
                        "I couldn't find that information in the meeting."
                    )
                else:
                    stage = "grounded answer generation"
                    with st.spinner("Preparing an answer..."):
                        st.session_state.answer = answer_from_context(
                            question.strip(), sources
                        )
            except Exception as error:
                st.error(
                    f"{stage.capitalize()} failed "
                    f"({type(error).__name__}): {error}"
                )

    if st.session_state.answer:
        st.subheader("Answer")
        st.write(st.session_state.answer)
        if st.session_state.answer_sources:
            with st.expander("Show transcript passages used"):
                for source in st.session_state.answer_sources:
                    st.markdown(
                        f"**Passage {source['chunk_id']}** · "
                        f"Relevance {source['similarity']:.0%}"
                    )
                    st.write(source["text"])
        else:
            st.caption("No matching transcript passages were found.")
