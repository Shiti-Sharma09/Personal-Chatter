"""Conversational memory: the original app treated every message as a
fresh, context-free query, so follow-ups like "what about page 3?" or
"and the second one?" couldn't be resolved against anything. This module
condenses (history + follow-up) into a standalone query before retrieval.
"""

CONDENSE_PROMPT = """Given the conversation history and a follow-up question, \
rewrite the follow-up as a standalone question that contains all the \
context needed to understand it on its own. If the follow-up is already \
standalone, return it unchanged. Answer only with the rewritten question.

Chat history:
{history}

Follow-up question: {question}
Standalone question:"""


def format_history(chat_history, max_turns=6):
    recent = chat_history[-max_turns:]
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)


def condense_question(chat_history, question, llm):
    """chat_history: list of {'role': 'user'|'assistant', 'content': str}.
    Returns a standalone question string."""
    if not chat_history:
        return question

    prompt = CONDENSE_PROMPT.format(history=format_history(chat_history), question=question)
    standalone = llm.invoke(prompt)
    return standalone.strip() or question
