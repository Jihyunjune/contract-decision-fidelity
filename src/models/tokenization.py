from functools import lru_cache
from transformers import AutoTokenizer
from src.common import ROOT


@lru_cache(maxsize=2)
def tokenizer(role="decision"):
    return AutoTokenizer.from_pretrained(
        str(ROOT / "data/tokenizers" / role), local_files_only=True
    )


def count(text, role="decision"):
    return len(tokenizer(role).encode(text, add_special_tokens=False))


def chat_count(messages, role):
    return len(
        tokenizer(role).apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    )
