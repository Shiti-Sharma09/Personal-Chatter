"""Local LLM loading via llama-cpp-python (GGUF).

Uses the model's own embedded chat template via create_chat_completion,
not raw-text completion. Raw completion gives an instruct model (e.g.
Qwen's ChatML-tuned models) no learned signal for when a turn is
"done" -- verified live: prompted as flat text, a 3B instruct model
answered correctly, then kept restating the same answer with new
phrasing until it hit the 512-token ceiling. Prompted through its own
chat template instead, the same model, same question, stopped itself
correctly after ~40 tokens (finish_reason: "stop"). GGML has been
superseded by GGUF for ~2 years; llama-cpp-python is the actively
maintained runtime for it.
"""
import os

from app import config


class ChatLLM:
    """Adapts llama_cpp.Llama's chat-completion API to the .invoke()/
    .stream() shape the rest of this codebase uses, so callers pass a
    list of {'role', 'content'} messages instead of a flat string."""

    def __init__(self, llama):
        self._llama = llama

    def invoke(self, messages, stop=None):
        result = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            repeat_penalty=config.LLM_REPEAT_PENALTY,
            stop=stop,
        )
        return result["choices"][0]["message"]["content"] or ""

    def stream(self, messages, stop=None):
        chunks = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            repeat_penalty=config.LLM_REPEAT_PENALTY,
            stop=stop,
            stream=True,
        )
        for chunk in chunks:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta


def load_llm():
    """Load a local GGUF model. Raises FileNotFoundError with setup
    instructions if MODEL_PATH doesn't point at a real file yet."""
    if not os.path.isfile(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No GGUF model found at '{config.MODEL_PATH}'. Download a GGUF "
            "model (e.g. a Llama-3 or Mistral instruct quant) and set "
            "MODEL_PATH in your .env to its path."
        )

    from llama_cpp import Llama

    llama = Llama(
        model_path=config.MODEL_PATH,
        n_ctx=config.LLM_CONTEXT_WINDOW,
        n_gpu_layers=config.LLM_GPU_LAYERS,
        # Most modern GGUF conversions embed the model's own chat
        # template (auto-detected when left None). If a particular GGUF
        # doesn't, set LLM_CHAT_FORMAT in .env to an explicit format
        # name llama-cpp-python recognizes (e.g. "chatml", "llama-3",
        # "mistral-instruct").
        chat_format=config.LLM_CHAT_FORMAT or None,
        verbose=False,
    )
    return ChatLLM(llama)
