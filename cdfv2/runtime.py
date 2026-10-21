from cdfv2.common import config
from src.models.ollama import Ollama as V1Ollama, InfrastructureError, ContextOverflow


class Ollama(V1Ollama):
    def __init__(self, role):
        super().__init__(role)
        self.cfg = config()
        if self.identity["digest"] != self.cfg["model_digests"][role]:
            raise InfrastructureError("Frozen Qwen model digest mismatch")
        if self.runtime["version"] != "0.40.2":
            raise InfrastructureError("Expected Ollama 0.40.2")
