---
title: "Draftfit 평가 노트 — Native Replay와 학습의 차이: H100 67라운드와 144토큰의 의미"
description: "H100에서 검증한 Ling native ReplaySSM 상태·토큰 parity를 설명하고, 학습·품질·속도 주장으로 확대하면 안 되는 이유를 정리합니다."
date: 2026-09-18
slug: "native-replay-h100-67-rounds"
categories: ["추측 디코딩", "추론 런타임", "검증"]
tags: ["Ling", "ReplaySSM", "H100", "state parity", "토큰 검증", "SGLang"]
draft: true
images: []
---

## 답부터 말하면

Draftfit — **Your workload. Your draft.** 내 데이터로 학습·미세조정한 draft를 평가할 때는 학습 성과와 추론 상태의 정확성을 분리해야 합니다. 이 글은 제품 전체의 성능 소개가 아니라, 별도 Ling serving 경로에서 확보한 과거의 제한된 상태 검증을 설명합니다.

제가 대여한 H100 캠페인은 Ling의 native replay가 특정 정적 단일 요청 조건에서 legacy state 경로와 같은 상태를 유지한다는 기능 근거를 만들었습니다. H100 PCIe의 세 프롬프트에서 실제 커밋 라운드가 모두 67회였고, legacy와 native의 temporal/convolution state와 출력 토큰이 일치했습니다. 별도의 대여 H100 SXM 호스트에서는 state hook 없이 native serving과 target-only를 비교해 48-token 요청 세 건의 `144/144` 토큰 ID가 일치했습니다.

그러나 이 숫자는 draft 학습 성능, production 품질, 처리량, 보편적 하드웨어 지원을 뜻하지 않습니다. 67라운드 상태 gate와 144-token ID gate는 서로 다른 호스트 캠페인이고, 둘 다 TP1·단일 active request·정적 검증 조건에 묶여 있습니다. Native replay는 teacher/draft 학습을 분산하는 기능이 아니라, 검증 과정에서 accepted prefix만 recurrent state에 커밋하고 rejected draft 뒤의 convolution state를 되돌리는 추론 상태 메커니즘입니다.

## Native replay가 해결하는 문제

Ling처럼 attention과 recurrent/convolution 성격의 상태가 함께 있는 타깃에서는 speculative decoding의 draft 제안이 모두 채택되지 않습니다. target이 일부 draft token을 확인한 뒤 첫 번째 불일치에서 멈추면, 다음 요청 라운드는 “채택된 prefix를 반영한 상태”에서 계속해야 합니다. rejected token까지 state에 남겨두면 다음 검증의 입력 상태가 target-only 실행과 달라집니다.

native ReplaySSM의 핵심은 이 commit boundary입니다. accepted prefix만 temporal state에 fold하고, rejected suffix에 해당하는 convolution state는 rollback합니다. 이 과정은 draft를 학습하는 producer/consumer disaggregation과 다른 계층에 있습니다. 학습 workflow가 GPU0 target과 GPU1 draft를 어떻게 배치하는지와, inference server가 한 요청의 state를 어떻게 되돌리는지는 각각 별도의 contract와 gate를 가져야 합니다.

## H100 PCIe의 67 committed rounds

첫 번째 H100 캠페인은 explanation, Python code, Korean 세 프롬프트를 사용했습니다. 세 케이스에서 실제 commit round가 모두 67회 발생했습니다. acceptance를 시뮬레이션하지 않았고, rejected prefix를 없애지도 않았습니다. 실제 speculative verify 결과의 accepted/rejected prefix를 포함한 상태를 legacy 경로와 native 경로에서 비교했습니다.

매 커밋 뒤 temporal state와 convolution state를 비교했고, 보존된 검증 요약에서 67회의 상태와 출력 token ID 일치를 확인했습니다. 비교 observer는 원래 commit 이후에만 동작하도록 두었습니다. pre-commit 동기화를 넣으면 race를 숨길 수 있기 때문에, 진단용 initial capture를 최종 완료 근거로 사용하지 않았습니다.

이 결과가 말해 주는 것은 “이 pinned target/draft/backend와 이 실행 조건에서 native fold의 상태가 legacy 상태와 기능적으로 일치했다”입니다. 그것보다 크게 말하면 안 됩니다. state parity가 통과해도 draft가 장기적으로 좋은 후보를 생성하는지, 실제 서비스에서 평균 acceptance length가 좋아지는지, memory나 throughput이 개선되는지는 측정되지 않았습니다.

## H100 SXM의 144/144 token 비교

두 번째 H100 SXM 캠페인은 state tensor comparator와 분리된 token-only check였습니다. 세 개의 48-token 요청을 native와 target-only로 실행했고, 생성된 token ID가 `144/144`로 일치했습니다. 실제 verify count는 `44/44/46`이었고, 각 요청에서 accepted draft token 수는 `3/3/1`이었습니다. native ring fold가 활성화된 것도 로그에서 확인했습니다.

이 검사는 state hook을 설치하지 않은 target-only 비교라는 점이 중요합니다. 따라서 144/144는 상태 tensor bitwise parity의 추가 증거가 아니라, native serving 경로가 target-only greedy token sequence를 바꾸지 않았다는 별도 결과입니다. 첫 번째 PCIe state gate와 두 번째 SXM token gate를 한 번의 단일 실험처럼 합쳐서 보고하지 않는 이유입니다.

## 검증 조건과 제외 범위

두 H100 결과는 대략 다음 조건의 좁은 교집합을 다룹니다.

- TP1, single active request, static ragged verify
- speculative verify 8, top-k 1, greedy temperature 0
- Triton attention/linear-attention backend
- float32 persistent SSM state
- radix cache off, CUDA graph off
- pinned target, draft, backend revision

따라서 radix-on, CUDA graphs, multi-request prefix-cache reuse, 다른 TP, 다른 target, 다른 GPU를 자동으로 지원한다고 말할 수 없습니다. H200의 별도 18-round replay 결과도 같은 식의 bounded functional evidence이며, H100의 67/144 수치를 모든 환경으로 확장하는 근거가 아닙니다.

특히 RTX5090 캠페인에서는 native ReplaySSM state parity가 실패했습니다. 비교 원소의 68.9%가 달랐고 최대 절대 차이는 `0.0684608221`이었습니다. 이 원인은 아직 분리되지 않았으며 SM120 profile은 native replay를 비활성화합니다. H100 통과가 이 실패 원인을 해결했다거나 Blackwell 지원을 증명한다는 뜻은 아닙니다.

## 실패가 남긴 엔지니어링 교훈

첫째, state gate와 token gate를 분리해야 합니다. token이 같아도 내부 상태가 잘못되면 다음 요청에서 문제가 생길 수 있고, 상태가 같아도 acceptance quality나 latency를 말할 수는 없습니다.

둘째, 관찰자가 원래 commit보다 먼저 state를 읽거나 동기화하면 구현의 race를 가릴 수 있습니다. 최종 검증에서는 post-commit observer, exact request ID filtering, warmup/health traffic 제외를 사용해야 합니다.

셋째, 실패를 지원 범위에서 지우지 말아야 합니다. RTX5090 parity 실패와 SM120 disabled 상태를 문서에 남겨야 사용자가 어떤 조합을 선택하면 안 되는지 알 수 있습니다.

넷째, 실험 자산을 모두 백업할 필요는 없습니다. 작은 request, output, log, hash, summary는 보존하고, 대형 target weights와 반복 state capture는 복구 필요성을 따져 제외할 수 있습니다. 다만 제외했다는 사실과 그 결과의 범위를 함께 적어야 합니다.

## 이 숫자를 릴리스 문장으로 번역하는 법

좋은 문장은 “H100의 고정된 TP1 단일 요청 검증에서, 실제 accepted/rejected prefix를 포함한 67 committed rounds의 state parity와 별도 144/144 token parity를 확인했다”입니다. 나쁜 문장은 “Ling native replay가 모든 GPU에서 정확하고 빠르다”입니다.

마찬가지로 “기능 검증 pass”는 괜찮지만 “품질 향상”이나 “최대 몇 배 speedup”은 이 자료에서 나오지 않습니다. 학습 loss 추세, acceptance length, target-only token equality, state equality는 서로 다른 측정 축입니다. 하나의 축이 통과했다고 나머지 축이 자동으로 통과하지 않습니다.

이 정직한 문장 경계는 마케팅을 약하게 만드는 것이 아니라 재현 비용을 줄입니다. 독자는 어떤 revision, GPU, request shape, cache mode, state dtype로 같은 결과를 얻었는지 알 수 있고, 아직 필요한 추가 gate도 분명해집니다.

## 근거와 범위

수치와 지원 범위는 [현재 모델 검증 기록](../docs/MODEL_VALIDATION.md)을 기준으로 읽습니다. 과거 캠페인의 성공은 제품 이름 변경 후 모든 경로를 다시 검증했다는 뜻이 아닙니다. [Draftfit 연재 목차](README.md)로 돌아갈 수 있습니다.

- 로컬 문서: `NATIVE_LING_REPLAY.md`, `RUNTIME_PROFILES.md`, `PUBLIC_SUPPORT.md`
- 로컬 실험: H100 native replay repair report, H100 unhooked serving gate, H200 replay report
- 실패 근거: Blackwell campaign의 native ReplaySSM parity failure
- 미포함: 별도 장기 50K training, 품질·처리량·속도 향상, 다른 모델/하드웨어의 blanket support

## FAQ

### 67은 state snapshot 개수인가요?

정확히는 세 프롬프트에서 관찰한 실제 committed rounds의 합입니다. 각 commit 뒤 temporal/convolution state와 출력 token을 비교했습니다.

### 144/144는 67라운드와 같은 장비 결과인가요?

아닙니다. 67-round state gate는 H100 PCIe, 144/144 token-only gate는 별도의 H100 SXM 캠페인입니다.

### 이 결과만으로 native replay를 켜도 되나요?

테스트한 pinned TP1/single-request/static 조건 안에서만 판단해야 합니다. radix, graphs, multi-request, 다른 GPU와 target은 별도 검증이 필요합니다.

### 관련 글은 무엇인가요?

[GPU0 target/GPU1 draft topology](gpu0-teacher-gpu1-draft-online-offline.md), [타깃 온보딩과 chat mask](draftfit-target-onboarding-chat-masks.md), [정직한 release matrix](honest-release-matrix-packaging-upstream.md)를 함께 보세요.
