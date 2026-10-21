"""Local-only inference. This module has no dataset/reference imports."""

import json, time, requests
from urllib.parse import urlparse
from src.common import config, digest, now
from src.models.tokenization import chat_count, count


class ContextOverflow(ValueError):
    pass


class InfrastructureError(RuntimeError):
    pass


def context_guard(estimated, num_predict, num_ctx):
    if estimated + num_predict + 512 > num_ctx:
        raise ContextOverflow(f"{estimated} input + {num_predict} output + 512 safety > {num_ctx}")


class Ollama:
    def __init__(self, role):
        self.cfg = config()
        self.role = role
        self.model = self.cfg[f"{role}_model"]
        self.endpoint = self.cfg["endpoint"]
        if urlparse(self.endpoint).hostname not in ["127.0.0.1", "localhost", "::1"]:
            raise ValueError("Local runtime only")
        self.runtime = self.request("GET", "/api/version")
        models = self.request("GET", "/api/tags")["models"]
        self.identity = next((m for m in models if m["name"] == self.model), None)
        if not self.identity:
            raise InfrastructureError(f"Model not installed: {self.model}")
        self.show = self.request("POST", "/api/show", {"model": self.model})

    def request(self, verb, path, payload=None):
        try:
            r = requests.request(
                verb, self.endpoint + path, json=payload, timeout=self.cfg["timeout_seconds"]
            )
            r.raise_for_status()
            obj = r.json()
            if "error" in obj:
                raise InfrastructureError(obj["error"])
            return obj
        except requests.RequestException as e:
            raise InfrastructureError(str(e)) from e

    def unload(self):
        self.request("POST", "/api/generate", {"model": self.model, "keep_alive": 0})

    def generate(self, messages, schema=None):
        maximum = self.cfg[f"{self.role}_num_predict"]
        estimated = chat_count(messages, self.role)
        context_guard(estimated, maximum, self.cfg["num_ctx"])
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": self.cfg["temperature"],
                "seed": self.cfg["seed"],
                "num_ctx": self.cfg["num_ctx"],
                "num_predict": maximum,
            },
        }
        if self.role == "decision":
            payload["think"] = False
        if schema is not None:
            payload["format"] = schema
        start = time.perf_counter()
        response = self.request("POST", "/api/chat", payload)
        content = response.get("message", {}).get("content", "")
        thinking = response.get("message", {}).get("thinking", "")
        return {
            "raw_output": content,
            "thinking_output": thinking,
            "response": response,
            "latency": time.perf_counter() - start,
            "input_tokens": response.get("prompt_eval_count"),
            "output_tokens": response.get("eval_count"),
            "estimated_input_tokens": estimated,
            "request_hash": digest(payload),
            "created_utc": now(),
            "done_reason": response.get("done_reason"),
            "model_id": self.model,
            "model_digest": self.identity["digest"],
        }
