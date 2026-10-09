import threading

# Process-wide lock held while ANY model (Whisper or the HuggingFace embedding model) is being loaded.
# HuggingFace transformers temporarily patches torch globally while loading ("meta device" init),
# so constructing another model in a different thread at the same moment produces empty "meta"
# weights ("Cannot copy out of meta tensor"). Streamlit runs each session in its own thread.
MODEL_LOAD_LOCK = threading.RLock()
