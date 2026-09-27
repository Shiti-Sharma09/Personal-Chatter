"""Local LLM loading via llama-cpp-python (GGUF), replacing the deprecated
CTransformers + GGML stack. GGML has been superseded by GGUF for ~2 years;
llama-cpp-python is the actively maintained runtime for it."""
import os

import config


def load_llm(streaming_callback=None):
    """Load a local GGUF model. Raises FileNotFoundError with setup
    instructions if MODEL_PATH doesn't point at a real file yet."""
    if not os.path.isfile(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No GGUF model found at '{config.MODEL_PATH}'. Download a GGUF "
            "model (e.g. a Llama-3 or Mistral instruct quant) and set "
            "MODEL_PATH in your .env to its path."
        )

    from langchain_community.llms import LlamaCpp

    callbacks = [streaming_callback] if streaming_callback else []
    return LlamaCpp(
        model_path=config.MODEL_PATH,
        n_ctx=config.LLM_CONTEXT_WINDOW,
        max_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        n_gpu_layers=config.LLM_GPU_LAYERS,
        streaming=streaming_callback is not None,
        callbacks=callbacks,
        verbose=False,
    )
