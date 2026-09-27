import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class FakeLLM:
    """Stand-in for a real GGUF model so the pipeline can be exercised
    without downloading multi-GB weights. Records prompts it was called
    with so tests can assert on them."""

    def __init__(self, response="This is a fake answer."):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response

    def stream(self, prompt):
        self.prompts.append(prompt)
        for word in self.response.split(" "):
            yield word + " "


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def sample_data_dir(tmp_path):
    data_dir = tmp_path / "Data"
    data_dir.mkdir()

    (data_dir / "guide.md").write_text(
        "# Personal Chatter Guide\n"
        "This document explains how the chatbot works.\n\n"
        "## Setup\n"
        "Install the requirements and download a GGUF model before starting.\n"
        "Configure MODEL_PATH in your .env file to point at it.\n\n"
        "## Usage\n"
        "Upload a document in the chat and ask questions about it. "
        "The bot retrieves relevant passages and answers using only that context.\n"
    )
    (data_dir / "notes.txt") .write_text(
        "Quarterly retrospective notes.\n"
        "The team shipped the hybrid retrieval upgrade in Q3.\n"
        "Latency stayed under 2 seconds for local CPU inference.\n"
    )
    return str(data_dir)
