---
title: "Draftfit 검증 매트릭스: 내 데이터로 학습한 draft를 어떻게 평가할까"
description: "Draftfit의 train·fine-tune·evaluate 흐름에서 학습 loss, feature parity, 복구, 실제 serving 성능을 구분하고 검증 근거와 실패를 정리합니다."
date: 2026-09-18
slug: "honest-release-matrix-packaging-upstream"
categories: ["추측 디코딩", "릴리스 엔지니어링"]
tags: ["Draftfit", "평가", "speculative decoding", "release matrix"]
draft: true
images: []
---

## 답부터 말하면

**Draftfit — Your workload. Your draft.** 내 프롬프트와 응답으로 draft를 학습하거나 기존 draft를 미세조정한 다음, 학습에 쓰지 않은 데이터와 실제 serving 조건에서 평가하는 것이 제품의 방향입니다. 학습이 실행된다는 사실과 내 워크로드에서 유용하다는 결론 사이에는 별도의 검증이 필요합니다.

패키지가 만들어지거나 CPU 검사가 통과해도 GPU 지원·품질·속도 향상은 증명되지 않습니다. 이 글은 [현재 모델 검증 기록](../docs/MODEL_VALIDATION.md)을 기준으로, 무엇을 구현했고 어떤 조건에서 통과하거나 실패했는지 구분하는 초안입니다. 과거 캠페인 결과는 Draftfit의 새 이름으로 모든 경로를 재검증했다는 뜻이 아닙니다.

## 1. 내 데이터를 기준으로 질문을 나누기

사용자가 제공하는 데이터는 개인 또는 팀의 실제 작업 분포를 반영해야 합니다. 예를 들어 코드 질문과 한국어 답변이 주된 요청이라면, 다른 분포의 범용 테스트만으로 그 워크로드의 draft를 선택하기 어렵습니다. 사용할 권한이 있는 데이터를 준비하고 민감정보를 검토한 뒤 train과 held-out 평가 데이터를 분리합니다. 이는 데이터 준비 원칙이며 자동 개인정보 정제 기능을 뜻하지 않습니다.

평가 질문도 분리합니다.

- **학습:** loss·gradient가 finite하고 draft 가중치가 실제로 바뀌는가?
- **입력 계약:** 동일 token과 mask에서 teacher feature가 기준 경로와 맞는가?
- **복구:** checkpoint가 optimizer·RNG·데이터 위치까지 맞게 복원하는가?
- **품질:** 학습에 쓰지 않은 요청에서 acceptance가 유용한가?
- **serving:** 내 요청 길이·동시성에서 target-only 대비 latency·처리량이 개선되는가?

마지막 두 질문은 짧은 반복 샘플 학습이나 export 성공으로 답할 수 없습니다. Draftfit의 학습·평가 경로를 사용할 수 있다는 설명과 자동 serving A/B·배포 시스템을 제공한다는 설명도 다릅니다.

## 2. 구현과 모델 인증을 분리하기

DSpark는 Draftfit 안의 draft 방법 중 하나입니다. DFlash, DFlash2, EAGLE3, P-EAGLE, Domino도 각자의 구현과 제약이 있으며, 방법 이름을 바꾸는 것만으로 가중치나 feature가 변환되지는 않습니다. DFlash2는 현재 구현되어 있고 실제 Qwen3-0.6B CUDA 학습·resume·export 근거가 있습니다. 과거 초안의 “DFlash2 미지원”은 더 이상 맞지 않습니다.

반면 DFly, DFlare, MTP, multimodal training은 공개 capability로 주장하지 않습니다. dense·MoE·hybrid·recurrent 이름이 config에 나타나도 모든 조합이 인증된 것은 아닙니다. target revision, backend, method, GPU, precision, TP와 cache 조건을 함께 봐야 합니다.

## 3. 실제 GPU 결과를 어떻게 읽을까

아래는 [MODEL_VALIDATION](../docs/MODEL_VALIDATION.md)의 제한된 기능 근거 요약입니다.

| 조건 | 확인한 결과 | 확대하면 안 되는 주장 |
| --- | --- | --- |
| H200 Ling online, teacher GPU0 / trainer GPU1 | 100 finite optimizer steps; live producer/store를 유지한 consumer 복구 | cold-host 복구나 held-out 품질 |
| H100 Ling offline | TP2 capture, 2-rank 학습, step 100→450 resume와 export | acceptance 기준 통과나 speedup |
| H100 Ling native replay | 67 committed-state rounds; 별도 144/144 token 비교 | 모든 GPU·cache·동시 요청의 상태 정확성 |
| RTX 5090 Ling native replay | state parity 실패, 해당 경로 비활성 | Blackwell native replay 지원 |
| 2×RTX 3090, Qwen3-0.6B, HF BF16 | 네 프롬프트 feature parity, DSpark·DFlash2 각각 20 updates, exact next-step resume, HF export/reload | 장기 품질·실제 serving 속도 |
| 같은 Qwen 타깃, vLLM BF16 | feature parity 실패, 최대 절대 오차 32 | BF16 cross-engine equivalence |
| 같은 Qwen 타깃, vLLM FP32 + deterministic trainer | feature parity·20 updates·exact resume·HF export/reload 통과 | BF16 실패 해소 또는 다른 모델 인증 |

vLLM FP32의 초기 비결정적 trainer 실행은 exact next-step replay에 실패했습니다. deterministic rerun의 성공으로 원래 실패를 지우지 않습니다. Managed HF·vLLM CLI의 각각 2-step 통과 역시 기능 smoke이지 BF16 parity나 restart 검증이 아닙니다.

실험 revision과 환경 pin은 원본 검증 문서에 남깁니다. 숫자를 제품 소개에 가져올 때도 그 한계를 함께 가져와야 합니다.

## 4. 패키징과 소유권도 별도 경계다

Draftfit은 SpecForge에서 파생한 학습 코드를 자체 플랫폼 안에서 유지합니다. TorchSpec·AngelSpec는 설계 참고와 해당 재사용 코드의 출처로 구분합니다. native HF/vLLM teacher 서비스와 외부 custom SGLang build, 설치된 Mooncake transport의 책임도 서로 다릅니다. 모두를 하나의 외부 wrapper 또는 하나의 자체 엔진이라고 부르면 정확하지 않습니다.

[코드 소유권](../docs/CODE_OWNERSHIP.md)과 [출처 기록](../docs/SOURCE_ATTRIBUTION.md)을 함께 읽어야 합니다. 원본 LICENSE, 파일별 header, third-party notices를 보존하는 것은 필요하지만 법률상 clearance를 보증하지는 않습니다.

wheel 설치, CLI, config, 오류 거부 검사는 배포물의 동작을 확인합니다. 그 통과를 GPU 모델 인증으로 쓰지 않고, 과거 CPU 검사 개수를 현재 릴리스의 결과처럼 재사용하지 않습니다. 공개 게시·패키지 업로드는 이 초안 수정의 범위가 아닙니다.

## 5. 지금의 워크플로와 미래 방향

현재는 사용자가 데이터를 준비하고 config, 학습 실행, checkpoint, 평가 조건을 명시적으로 선택하는 흐름입니다. OpenTelemetry로 운영 트래픽을 자동 수집하고, 변화에 맞춰 지속 재학습하며, 새 draft를 자동 배포·롤백하는 제품 루프는 **미래 방향이며 현재 미구현**입니다.

이 구분은 “내 워크로드에 맞춘다”를 “데이터가 알아서 수집되고 운영까지 자동화된다”로 오해하지 않게 합니다. 타깃 모델 자체의 개인화나 답변 품질 향상을 보장하는 설명도 아닙니다.

## FAQ

### CPU 검사가 통과하면 GPU 지원인가요?

아닙니다. 새 조합에는 real-weight capture, finite update, export reload와 필요한 serving/state 검사가 별도로 필요합니다.

### 학습 loss가 낮아지면 더 빠른가요?

아닙니다. held-out acceptance와 실제 target-only 대비 serving 비교를 따로 측정해야 합니다.

### 어떤 결과를 지원 문장으로 써도 되나요?

검증 문서에 남아 있는 target·revision·backend·hardware·precision 조건과 측정한 항목을 함께 적습니다. 실패와 미검증 항목을 빼지 않습니다.

## 관련 글과 근거

[Draftfit 연재 목차](README.md), [타깃과 chat mask](draftfit-target-onboarding-chat-masks.md), [GPU 역할과 feature 경로](gpu0-teacher-gpu1-draft-online-offline.md), [checkpoint 복구](checkpoint-warmstart-resume-spot-backup.md)를 이어서 보세요. 모든 수치의 기준은 [모델 검증 기록](../docs/MODEL_VALIDATION.md)입니다.
