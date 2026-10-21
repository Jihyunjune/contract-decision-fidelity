# Contract Decision Fidelity — 팀원 안내

공개 패키지는 완성된 v2 실험의 구현과 집계 결과를 독립적으로 실행하도록 정리한 것입니다. 원본 v1/v2를 import하지 않으며, 이번 공개 준비 과정에서 6,800개 판단을 다시 생성한 것으로 주장하지 않습니다.

## 핵심 결과와 해석

100개 계약서 × 17개 가설 × 4개 방법 = 6,800개 판단입니다. LongLLMLingua는 Naive 대비 정확도 **+9.12%p**, Decision Flip **−17.24%p**였지만, 원문 대비 정확도는 **−6.12%p**였습니다. 실제 토큰 보존율은 **31.46%**로 목표 상한 30%를 초과했습니다. 전체 처리시간도 증가했습니다.

| 방법 | Strict Accuracy | Macro-F1 | Decision Flip | 실제 보존율 |
|---|---:|---:|---:|---:|
| Original | 67.71% | 63.59% | 0.00% | 100.00% |
| Naive | 52.47% | 49.04% | 47.00% | 11.42% |
| Contextualized | 54.82% | 51.58% | 43.41% | 12.66% |
| LongLLMLingua | 61.59% | 58.19% | 29.76% | 31.46% |

질문을 보는 LongLLMLingua와 질문을 보지 않는 범용 요약은 정보 조건이 다릅니다. 더 많은 문맥을 보존하므로 순수 알고리즘 효과로 해석할 수 없습니다. 모든 방법의 유효 레이블 출력률은 100%지만 인용문 유효성과 의미적 정확성은 별개입니다. Decision Flip은 원문 판단과의 불일치이며 오류 자체가 아닙니다.

## 설치 및 오프라인 화면

프로젝트 루트에서 실행합니다. Python 3.10 및 Windows/RTX 5080 환경으로 검증했습니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m cdfv2.cli verify
python -m pytest -q
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

추론 없이 집계 결과만 보려면 `python -m pip install -r requirements-viewer.txt`로 설치할 수 있습니다. Overview는 데이터·Ollama 없이 작동합니다.

## 데이터와 모델 준비

별도 터미널에서 Ollama 0.40.2를 11436 포트로 실행합니다. 모델 digest가 config와 다르면 실행을 중단합니다. `.env.example`은 자동 로딩되지 않으므로 환경변수를 직접 설정합니다.

```powershell
$env:OLLAMA_HOST="127.0.0.1:11436"
ollama serve
```

프로젝트 터미널:

```powershell
$env:OLLAMA_HOST="127.0.0.1:11436"
$env:CDF_OLLAMA_URL="http://127.0.0.1:11436"
ollama pull qwen2.5:7b-instruct
ollama pull qwen3:8b
python -m cdfv2.cli runtime-check
python -m cdfv2.cli download
python -m cdfv2.cli assets
python -m cdfv2.cli audit
```

Llama 2 라이선스·사용 정책을 읽은 뒤 압축 모델을 내려받습니다. 가중치가 약 13.5GB이므로 여유 공간을 확보하세요.

```powershell
python -m cdfv2.cli assets --compressor
python scripts/release_smoke.py
python scripts/run_experiment.py --split development
python -m cdfv2.cli evaluate --split development
python -m cdfv2.cli lock
python scripts/run_experiment.py --split sealed
python -m cdfv2.cli evaluate --split sealed
```

전체 재현에는 시간이 필요합니다. 이미 완료된 체크포인트는 동일 입력·설정에서 재사용합니다. 원래 확정된 30% 예산과 100개 평가 ID를 유지하며, 결과를 보고 다시 선택하지 않습니다. 새 실행은 독립적인 신규 테스트가 아니라 기존 실험의 재현입니다. 원본 Test 123건은 이미 관찰되었으므로 v2 최종 평가로 재사용하지 않았습니다. Train 전체 통계의 과거 노출 등 독립성 한계를 유지합니다.

## 파일과 보존

- `results/published/`: 원본 최종 집계값. 평가 실행이 덮어쓰지 않습니다.
- `results/sealed/`: 사용자가 새로 생성한 결과. Git에서 제외됩니다.
- `results/manifests/`: 기존 출처 해시, 분할 ID, 공개 파일 목록.
- `data/`, `models/`: 공식 다운로드로 생성하며 공개 저장소에서 제외합니다.
- `reports/public_release_audit.md`: 실제 실행 검사와 공개 전 감사 결과.

ContractNLI는 CC BY 4.0이지만 계약서 원문·요약·인용은 저장소에 넣지 않습니다. 프로젝트 코드는 권리자 승인에 따라 MIT이며 Qwen/LLMLingua/Llama 2/라이브러리의 라이선스는 별도입니다. Llama 2 가중치를 MIT로 재배포할 수 있다는 뜻이 아닙니다. 법률 자문이나 실제 계약 효력 판정 도구가 아닙니다.

GitHub 저장소 생성과 Push는 이 공개 준비 작업에 포함되지 않으며 별도 사용자 승인 후 수행합니다.
