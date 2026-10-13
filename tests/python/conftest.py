"""
Shared test fixtures.

The runtime tests deliberately avoid downloading real weights: a tiny randomly
initialised Llama is enough to prove that the paged cache, the scheduler and the
speculative sampler behave correctly, and it keeps the suite fast and offline.
Correctness against real weights is covered by benchmarks/run_live_engine.py.
"""

import os

# Must be set before anything imports server.config, which reads the environment
# at import time. Keeps the API tests on a tiny model and off the network.
os.environ.setdefault("MEMFORGE_EAGER_LOAD", "false")
os.environ.setdefault("MEMFORGE_LOAD_DRAFT", "false")
os.environ.setdefault("MEMFORGE_MAX_MODEL_LEN", "256")
os.environ.setdefault("MEMFORGE_KV_BUDGET_MB", "32")
os.environ.setdefault("MEMFORGE_MAX_NUM_SEQS", "4")

import pytest  # noqa: E402
import torch  # noqa: E402

VOCAB_SIZE = 256


def _tiny_llama(num_layers: int = 2, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=1,
        bos_token_id=0,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


class StubTokenizer:
    """
    Minimal tokenizer protocol: the engine only needs __call__, decode and a few
    attributes, so this avoids pulling a real tokenizer into unit tests.
    """

    eos_token_id = 1
    pad_token_id = 0
    chat_template = None

    def __call__(self, text, return_tensors=None):
        # No truncation: the engine is responsible for enforcing context limits.
        ids = [(ord(c) % (VOCAB_SIZE - 2)) + 2 for c in text] or [2]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((int(i) % 26) + 97) for i in ids)


@pytest.fixture(scope="session")
def tiny_model():
    return _tiny_llama(num_layers=2, seed=0)


@pytest.fixture(scope="session")
def tiny_draft_model():
    return _tiny_llama(num_layers=1, seed=1)


@pytest.fixture(scope="session")
def stub_tokenizer():
    return StubTokenizer()


@pytest.fixture
def prompt_ids():
    return [7, 11, 23, 42, 9, 13, 31, 5]
