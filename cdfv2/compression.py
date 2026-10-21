"""Official LongLLMLingua, question-aware coarse and iterative token compression."""

import time, importlib.metadata, re
from cdfv2.common import ROOT, config, text_hash, sha, read_json
from src.models.tokenization import count


def chunks_with_offsets(text, tokenizer, limit):
    chunks = []
    start = 0
    end = 0
    # Pack contiguous original lines. Long lines are partitioned, never omitted.
    for line in text.splitlines(keepends=True):
        newend = end + len(line)
        if (
            len(tokenizer.encode(text[start:newend], add_special_tokens=False)) > limit
            and end > start
        ):
            chunks.append((start, end))
            start = end
        while len(tokenizer.encode(text[start:newend], add_special_tokens=False)) > limit:
            offsets = tokenizer(
                text[start:newend], add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            cut = start + offsets[limit - 1][1]
            if cut <= start:
                raise ValueError("Nonprogressing token offset")
            chunks.append((start, cut))
            start = cut
        end = newend
    if start < len(text):
        chunks.append((start, len(text)))
    assert "".join(text[a:b] for a, b in chunks) == text
    return chunks


class Compressor:
    def __init__(self):
        import torch
        from transformers import BitsAndBytesConfig
        from llmlingua import PromptCompressor

        self.torch = torch
        cfg = config()["compressor"]
        start = time.perf_counter()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for compressor")
        torch.manual_seed(config()["seed"])
        self.engine = PromptCompressor(
            model_name=str(ROOT / "models/llama2-chat"),
            device_map="cuda",
            model_config={
                "trust_remote_code": False,
                "local_files_only": True,
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
            },
            use_llmlingua2=False,
        )
        self.engine.model.eval()
        self.load_seconds = time.perf_counter() - start
        self.forward_tokens = 0
        self.forward_calls = 0

        def hook(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                self.forward_tokens += int(ids.numel())
                self.forward_calls += 1

        self.hook = self.engine.model.register_forward_pre_hook(hook, with_kwargs=True)
        self.cache = {}

    def compress(self, document, hypothesis, rate):
        cfg = config()["compressor"]
        text = document.original_text
        key = text_hash(text)
        if key not in self.cache:
            self.cache[key] = chunks_with_offsets(text, self.engine.tokenizer, cfg["chunk_tokens"])
        offsets = self.cache[key]
        # SentencePiece can merge a leading blank with the prefix boundary. Trim
        # boundary whitespace only, retaining the original offsets and all words.
        active = [(a, b) for a, b in offsets if text[a:b].strip()]
        contexts = [text[a:b].strip() for a, b in active]
        # The hypothesis is a query, never an asserted fact in the returned context.
        question = (
            "Determine whether the contract entails, contradicts, or does not mention this statement: "
            + hypothesis.hypothesis_text
        )
        source_tokens = len(self.engine.tokenizer.encode(text, add_special_tokens=False))
        target = max(1, int(source_tokens * rate))
        before = self.forward_tokens
        calls = self.forward_calls
        self.torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = self.engine.compress_prompt(
            context=contexts,
            instruction="",
            question=question,
            target_token=target,
            iterative_size=cfg["iterative_size"],
            condition_compare=cfg["condition_compare"],
            condition_in_question=cfg["condition_in_question"],
            rank_method="longllmlingua",
            reorder_context=cfg["reorder_context"],
            dynamic_context_compression_ratio=cfg["dynamic_context_compression_ratio"],
            context_budget=cfg["context_budget"],
            use_context_level_filter=True,
            use_token_level_filter=True,
            use_sentence_level_filter=False,
            concate_question=False,
        )
        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        compressed = result["compressed_prompt"]
        if not compressed.strip():
            raise ValueError("Empty compressed context")
        # Library includes an illustrative GPT-4 saving field, NOT an actual cost.
        library = {k: v for k, v in result.items() if k != "saving"}
        return {
            "document_id": document.document_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "status": "SUCCESS",
            "compressed_context": compressed,
            "input_hash": key,
            "question_hash": text_hash(question),
            "compressed_hash": text_hash(compressed),
            "requested_rate": rate,
            "compression_ratio": count(compressed) / count(text),
            "original_tokens": count(text),
            "compressed_tokens": count(compressed),
            "compressor_original_tokens": source_tokens,
            "target_compressor_tokens": target,
            "library_output": library,
            "chunk_offsets": offsets,
            "library_chunk_offsets": active,
            "boundary_normalization": "strip whitespace only; no content removal",
            "ranked_chunk_ids": self.engine.context_idxs[-1] if self.engine.context_idxs else [],
            "latency": seconds,
            "compressor_forward_tokens": self.forward_tokens - before,
            "compressor_forward_calls": self.forward_calls - calls,
            "peak_gpu_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "model_load_seconds": self.load_seconds,
            "llmlingua_version": importlib.metadata.version("llmlingua"),
            "model_name": cfg["model_name"],
            "model_revision": cfg["revision"],
            "gold_input": False,
            "hypothesis_appended_to_context": False,
            "dollar_cost": 0,
            "electricity_cost": "not measured",
        }

    def close(self):
        import gc

        self.hook.remove()
        del self.engine
        gc.collect()
        self.torch.cuda.empty_cache()
