"""Conversational memory: the original app treated every message as a
fresh, context-free query, so follow-ups like "what about page 3?" or
"and the second one?" couldn't be resolved against anything. This module
condenses (history + follow-up) into a standalone query before retrieval.
"""

CONDENSE_SYSTEM_PROMPT = (
    "Given a conversation history and a follow-up question, rewrite the follow-up as a "
    "standalone question containing all context needed to understand it on its own. If it's "
    "already standalone, return it unchanged. Reply with only the rewritten question -- a "
    "single line, no explanation, no other text."
)


def build_condense_messages(chat_history, question):
    return [
        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Chat history:\n{format_history(chat_history)}\n\nFollow-up question: {question}"},
    ]


def format_history(chat_history, max_turns=6):
    recent = chat_history[-max_turns:]
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)


# Substrings that show up when a model echoes the prompt's own scaffolding
# instead of actually answering it (seen in practice with both a 0.5B and
# a 3B instruct model) -- a real rewritten question should never contain
# its own instructions.
_ECHOED_SCAFFOLDING = (
    "standalone question",
    "follow-up question",
    "rewritten question",
    "chat history",
    "answer only",
    "answer:",
)


def condense_question(chat_history, question, llm):
    """chat_history: list of {'role': 'user'|'assistant', 'content': str}.
    Returns a standalone question string.

    A standalone question is expected to be a single line, so generation
    is stopped at the first newline and hard-capped at a short length.
    That alone isn't enough, though: a model that doesn't follow "answer
    only with the rewritten question" can cram the prompt's own
    scaffolding onto that single line (e.g. "Rewritten standalone
    question: ... Answer only with the above format.") rather than
    hitting a newline at all. Since that text becomes both the retrieval
    query and part of the next prompt, garbage here silently poisons the
    real answer -- so anything that looks like echoed scaffolding is
    discarded in favor of just using the original, un-condensed question.
    """
    if not chat_history:
        return question

    messages = build_condense_messages(chat_history, question)
    raw = llm.invoke(messages, stop=["\n"])
    standalone = (raw or "").strip().splitlines()[0].strip() if (raw or "").strip() else ""
    standalone = standalone[:300]

    if not standalone or any(marker in standalone.lower() for marker in _ECHOED_SCAFFOLDING):
        return question
    return standalone
