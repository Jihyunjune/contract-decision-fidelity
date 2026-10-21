# Third-party notices and distribution boundaries

Project contributors authorized **MIT** for their project code during release preparation. The original projects had no code LICENSE; this authorization does not relicense dependencies, datasets or models. No upstream LLMLingua, Transformers, Ollama, Streamlit or model implementation is vendored. The project calls installed libraries and a local Ollama service. Shared files copied from the team's v1 implementation are identified in the provenance manifest and remain project code.

| Component | Version / identity used | License | Distribution |
|---|---|---|---|
| ContractNLI | official archive SHA-256 `e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b` | CC BY 4.0 and accompanying terms | Download script only; no contract text or derived excerpts |
| Qwen2.5-7B-Instruct | Ollama digest in config; tokenizer revision `a09a35458c702b33eeacc393d103063234e8bc28` | Apache-2.0 | No weights/tokenizer assets bundled |
| Qwen3-8B | Ollama digest in config; tokenizer revision `b968826d9c46dd6066d109eabc6255188de91218` | Apache-2.0 | No weights/tokenizer assets bundled |
| LLMLingua / LongLLMLingua | `llmlingua==0.2.2` | MIT, Microsoft | Installed dependency; no vendored library code |
| Llama 2 7B Chat | NousResearch mirror revision `351844e75ed0bcbbe3f10671b3c808d2b83894ee` | Llama 2 Community License Agreement and Acceptable Use Policy | Separate download; **not** covered by project MIT |
| Streamlit | 1.65.0 | Apache-2.0 | Installed dependency |
| Transformers / Accelerate | 4.46.3 / 0.33.0 | Apache-2.0 | Installed dependencies |
| PyTorch | 2.8.0+cu128 | BSD-style with bundled third-party notices | Installed dependency; CUDA components have separate terms |
| NumPy / pandas / scikit-learn | 1.26.4 / 2.2.3 / 1.7.2 | BSD-3-Clause | Installed dependencies |
| Pydantic / requests / pytest | 2.14.1 / 2.32.5 / 8.4.2 | MIT / Apache-2.0 / MIT | Installed dependencies |
| bitsandbytes | 0.49.2 | MIT | Installed dependency |
| Matplotlib | 3.10.9 | Matplotlib/PSF-derived license | Installed dependency |

See `docs/dependency_licenses.json` for installed distribution metadata; transitive dependency notices remain in their installed distributions. This list does not grant rights beyond upstream licenses.

## Dataset attribution

Yuta Koreeda and Christopher D. Manning. 2021. *ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts*. Findings of EMNLP. [Paper](https://aclanthology.org/2021.findings-emnlp.164/), [official dataset and license](https://stanfordnlp.github.io/contract-nli/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The download retains official JSON and original span positions; separate projections select the historical v2 Train partition and tokenize inputs. Original annotations are not edited. No endorsement by the dataset authors is implied. Although CC BY 4.0 permits redistribution under its conditions, this repository deliberately distributes download instructions, IDs/hashes and aggregate measurements rather than underlying NDA text. Dataset terms also address third-party links and infringement notices; downloading users should review them.

## Model and implementation sources

- [Qwen2.5 Apache-2.0 license](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE)
- [Qwen3 Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE)
- [LLMLingua MIT license](https://github.com/microsoft/LLMLingua/blob/main/LICENSE)
- [Pinned Llama 2 license](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/351844e75ed0bcbbe3f10671b3c808d2b83894ee/LICENSE.txt) and [acceptable use policy](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf/blob/351844e75ed0bcbbe3f10671b3c808d2b83894ee/USE_POLICY.md)
- [Streamlit license](https://github.com/streamlit/streamlit/blob/develop/LICENSE)

The local pinned Llama license file was inspected directly during release preparation. A mirror being downloadable does not remove Meta's license conditions. Do not upload Llama weights to this repository or imply that they are MIT/Apache licensed. Any downstream redistribution or model-service deployment must satisfy the applicable upstream agreement and acceptable-use conditions.

## Methodological adaptation

The official LongLLMLingua algorithm is used through `PromptCompressor`, with question-aware ranking and token compression; LLMLingua-2 is disabled. Project-authored adaptations are ContractNLI hypothesis queries, contiguous 256-token chunk packing, boundary whitespace normalization, 4-bit scoring, local inference orchestration and audit/cost accounting. The query is not appended as an asserted fact to the returned context. No gold labels or evidence are passed to compression. This is a domain adaptation, not a claim to reproduce every experiment of the paper.
