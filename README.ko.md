# Draftfit

[English](README.md) | [한국어](README.ko.md)

**내 모델, 내 작업, 내 데이터에 맞는 speculative decoding draft를 훈련하세요.**

**DSpark · DFlash · DFlash2 · EAGLE3 · PEagle · Domino**

[빠른 시작](#빠른-시작) · [Draft 방법](#draft-방법) · [백엔드](#백엔드) · [예시 모델](#예시-모델) · [문서](docs/README.md)

Draftfit은 개인 개발자와 에이전트 개발자가 실제 작업에 맞는 작은 draft 모델을
훈련하도록 돕습니다. 대화 데이터를 준비하고 타깃과 draft 방법을 선택하면,
하나의 CLI로 훈련·내보내기·벤치마크를 진행할 수 있습니다.
추론 서비스 팀과 연구자도 같은 과정으로 방법별 결과를 비교할 수 있습니다.

타깃 모델은 고정합니다. Draft가 토큰을 제안하고, 타깃이 검증합니다.
**내 작업에 맞춘, 나만의 draft.**

소스 릴리스 **0.2.0** · [변경 이력](CHANGELOG.md) · [향후 계획](plans/README.md)

https://github.com/user-attachments/assets/ba787371-e71f-4413-aa03-d697090ccbb1

[29초 워크플로 영상 보기](https://github.com/JayYun98/Draftfit/releases/download/v0.2.0/draftfit-system-demo-en.mp4) — 영어 설명 영상이며, 실제 속도 향상 측정 영상은 아닙니다.

## 왜 Draftfit인가요?

- **내 데이터를 사용합니다.** 검토한 대화 데이터를 준비하고 holdout을 분리합니다.
- **Draft를 직접 설정합니다.** 타깃, 방법, 깊이, feature layer, block size를 선택합니다. 호환되는 기존 draft를 추가 학습하거나 새로 훈련할 수 있습니다.
- **훈련 방식을 선택합니다.** 저장된 타깃 feature를 재사용하거나 별도 trainer로 실시간 전달합니다. 중단된 훈련을 재개하고 결과를 내보낼 수 있습니다.
- **실제 속도를 측정합니다.** Loss나 토큰 수용률뿐 아니라 내 작업의 end-to-end 처리량을 비교합니다. 속도 향상은 보장이 아닌 측정 결과입니다.

```text
내 대화 → 데이터 준비 → Feature 추출 → 훈련 → 내보내기 → 벤치마크
                        teacher       draft              타깃 단독과 비교
```

## 빠른 시작

먼저 [호환되는 uv 환경](docs/ENVIRONMENT.md)을 설치하세요.
준비된 실행 환경에서 검토한 대화 데이터와 DSpark 훈련 프로젝트를 만듭니다.

~~~sh
draftfit data prepare --input ./my-conversations.jsonl \
  --output ./data/train.jsonl --split-eval --eval-output ./data/holdout.jsonl

draftfit target prepare /path/to/target --local-only \
  --strategy dspark --train-data ./data/train.jsonl --output-dir ./my-draft

draftfit train -c ./my-draft/train.json --plan
draftfit train -c ./my-draft/train.json
~~~

이 온라인 예제는 기본적으로 GPU 0을 teacher, GPU 1을 trainer에 사용합니다.
저장된 feature를 이용한 훈련, 기존 draft 추가 학습, 복구와 내보내기는
[전체 워크플로](docs/PUBLIC_WORKFLOW.md)를 참고하세요.

## Draft 방법

| 방법 | Offline feature | Online capture | 주요 조건 |
| --- | --- | --- | --- |
| DSpark | 지원 | 지원 | 보조 layer와 타깃의 최종 hidden state |
| DFlash | 지원 | 지원 | 호환 hidden feature와 mask token |
| DFlash2 | 지원 | 지원 | 전체 vocabulary; convolution·selector 훈련 |
| EAGLE3 | 지원 | 지원 | Draft 1개 layer; 온라인 훈련용 vocabulary mapping |
| PEagle | 미지원 | 지원 | Flex attention과 방법별 feature layout |
| Domino | 지원 | 지원 | 호환 draft 설정을 명시적으로 제공 |

## 백엔드

모든 teacher는 같은 **PyTorch draft trainer**에 연결됩니다.
아래 등급은 구현과 검증 범위를 나타내며, 모든 조합의 production 보장은 아닙니다.

| 구성 요소 | 역할 | 지원 등급 | 검증 범위 |
| --- | --- | --- | --- |
| PyTorch | Draft trainer | Core | 훈련, 동일 world size 재개, export; FSDP1 기본, FSDP2 선택 |
| SGLang | Offline / online teacher | Maintained, 검증된 설정 | 실제 Ling capture·훈련, 버전을 고정한 제한적 serving/state 검사 |
| Hugging Face Transformers | Offline / online teacher | Native, 제한적 GPU 검증 | Qwen3-0.6B BF16 feature parity; DSpark/DFlash2 훈련·재개·export |
| vLLM | Offline / online teacher | Native, 실험적 | Qwen3-0.6B FP32 parity와 훈련·재개·export; **BF16 parity 실패** |
| TensorRT-LLM / TokenSpeed | Teacher | 미구현 | 이번 릴리스에 통합 없음 |

Native HF/vLLM capture의 현재 범위는 단일 장치의 Llama/Qwen2/Qwen3 텍스트
디코더입니다. Teacher 지원이 내보낸 draft의 serving 지원까지 뜻하지는 않습니다.
DFlash2의 실제 serving은 아직 검증되지 않았습니다.
[백엔드 요구사항과 방법별 제약 →](docs/PUBLIC_SUPPORT.md)

## 예시 모델

예제는 시작용 설정이며, 모든 모델 × 방법 조합의 동작을 보장하지 않습니다.
아래 실제 가중치 검증은 제한된 기능 검사이며 production·속도 향상 인증이 아닙니다.

| 타깃 유형 | 현재 예제 | 확보한 근거 |
| --- | --- | --- |
| Dense + KV | [Qwen3-8B](configs/qwen3-8b-dspark.json), [Llama3-8B](configs/llama3-8B-eagle3.json) | 설정 예제; 별도의 Qwen3-0.6B 실제 가중치 DSpark/DFlash2 훈련·재개·export 검증 |
| MoE + KV | [Qwen3-30B-A3B](configs/qwen3-30B-A3B-eagle3.json) | 설정 예제; 이전 TP2 MoE capture는 dummy 가중치 사용 |
| MoE / conv hybrid | [Ling-3.0-tiny](configs/ling-3.0-tiny-dspark.json) | 실제 가중치 DSpark capture·훈련·export와 고정 버전의 제한적 serving/replay |
| Linear + KV hybrid | [Qwen3.5-4B](configs/qwen3.5-4b-dflash.json) | 설정 예제; 타깃 전체 워크플로 미인증 |
| Conv + KV | [LFM2.5-1.2B-Instruct](configs/lfm2.5-1.2b-instruct-dflash.json) | 실제 가중치 load/chat smoke만 완료; draft 워크플로 미검증 |
| Recurrent | [RWKV7-1.5B](configs/target_specs/rwkv7-1.5b.json) | Metadata/state fixture만 존재; 호환 백엔드 필요 |

[전체 예제](examples/configs/README.md), [정확한 검증 범위](docs/MODEL_VALIDATION.md),
[최신 모델 실험 계획](plans/README.md)을 참고하세요.
[타깃 설정과 adapter](docs/TARGET_EXTENSION.md)로 모델을 추가할 수 있지만,
설정만으로 미지원 capture·serving 백엔드를 구현할 수는 없습니다.

## 문서

상세 문서는 현재 영어로 제공하며, [영문 README](README.md)가 기준입니다.

| 목적 | 가이드 |
| --- | --- |
| 설치·teacher 백엔드 선택 | [환경 설정](docs/ENVIRONMENT.md) |
| 데이터 준비·훈련·추가 학습·재개·export | [전체 워크플로](docs/PUBLIC_WORKFLOW.md) |
| 새 타깃·runtime 설정 | [타깃 확장](docs/TARGET_EXTENSION.md) · [Runtime profiles](docs/RUNTIME_PROFILES.md) |
| 타깃 단독과 draft 추론 비교 | [작업별 평가](docs/PERFORMANCE_GATE.md) |
| 지원·미검증 조합 확인 | [지원 범위](docs/PUBLIC_SUPPORT.md) |
| 개발·검증 재현 | [문서 목록](docs/README.md) · [소스 도구](tests/VALIDATION.md) |

자동 trace 수집과 continual deployment는 [향후 계획](plans/README.md)입니다.
수용된 draft가 길어도 더 빠르다는 보장은 없습니다. Holdout 요청에서 전체 처리량을
측정하고 정확성을 별도로 검증하세요.

## 출처와 감사

Draftfit은 [SpecForge](https://github.com/sgl-project/SpecForge)에서 파생된 훈련 runtime을
직접 유지보수하며, [TorchSpec](https://github.com/lightseekorg/TorchSpec),
[AngelSpec](https://github.com/Tencent/AngelSpec), DSpark 관련 프로젝트의 구현을
참고·적용하고 출처를 표기합니다.
[코드 소유 범위](docs/CODE_OWNERSHIP.md), [외부 코드 고지](THIRDPARTY_NOTICES.md),
[라이선스](LICENSE)를 참고하세요.
