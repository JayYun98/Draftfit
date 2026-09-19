---
title: "Draftfit GPU0 Teacher, GPU1 Draft: Online과 Offline 학습을 같은 계약으로 비교하기"
description: "대여한 H200에서 확인한 GPU0 target capture와 GPU1 draft training, 그리고 online/offline feature parity의 의미와 한계를 설명합니다."
date: 2026-09-18
slug: "gpu0-teacher-gpu1-draft-online-offline"
categories: ["추측 디코딩", "GPU 시스템", "분산 학습"]
tags: ["H200", "GPU placement", "online training", "offline training", "Mooncake", "DSpark"]
draft: true
images: []
---

## 답부터 말하면

Draftfit — **Your workload. Your draft.** 사용자가 준비한 프롬프트·응답에 맞춰 draft를 학습하고 평가하려면, teacher feature를 언제 만들고 어디에 보관할지 선택해야 합니다. online과 offline은 이 선택을 위한 데이터 경로이지 속도 등급이 아닙니다.

GPU0에 frozen target을 두고 GPU1에서 draft를 학습하는 online topology는 제가 대여한 H200에서 기능적으로 통과했습니다. 같은 13-token Ling 입력에 대해 online capture와 offline feature tensor가 캠페인 설정의 taps 3, 7, 11, 15, 19에서 bitwise equal이었고, GPU0 target TP1과 GPU1 training을 동시에 유지한 100 optimizer steps도 finite하게 완료했습니다. 이것은 topology와 feature contract가 맞는다는 근거이지, 어떤 모델·GPU·동시 요청에서도 같은 결과가 난다는 보장이나 speedup benchmark는 아닙니다.

또 하나의 중요한 구분은 online과 offline이 “서로 다른 학습기”가 아니라 같은 typed training entry point가 선택하는 데이터 경로라는 점입니다. online은 target feature를 실행 중 받아오고, offline은 미리 저장한 feature checkpoint를 읽습니다.

## Online teacher를 별도 서비스로 분리하기

온라인 학습에서 target model은 trainer 프로세스 안에 복사되어 forward되는 teacher처럼 보일 수 있지만, 실제 런타임 경계는 조금 다릅니다. teacher service가 target inference와 feature materialization을 맡고, producer가 prompt를 보내며, Mooncake transport/store를 통해 feature reference를 consumer에 전달합니다. 아래의 역사적 Ling 캠페인은 patched SGLang 경로이고, 현재 Draftfit은 별도의 native HF/vLLM teacher 경로도 갖습니다. 각 backend의 지원과 parity는 따로 검증합니다. GPU0은 이 target service의 TP/EP placement를 소유하고, consumer trainer는 GPU1에서 draft를 업데이트합니다.

이 분리는 세 가지 이점을 줍니다. 첫째, target weights와 draft optimizer state의 메모리 요구를 분리할 수 있습니다. 둘째, producer와 consumer가 같은 run contract를 보고 있는지 manifest로 확인할 수 있습니다. 셋째, target inference가 실패하면 feature stream을 성공한 것처럼 가장하지 않고 prompt lease를 회수하거나 run을 실패로 끝낼 수 있습니다.

그 대신 online은 운영 요소가 늘어납니다. capture server, health endpoint, Mooncake metadata/RPC endpoint, control directory, consumer SQLite/WAL, request lifecycle이 모두 신뢰 경계에 들어옵니다. external-services mode에서는 CLI가 이 서비스들을 대신 시작하거나 GPU를 배정하지 않습니다. managed-local은 별도의 단일 노드 개발 편의 기능이며 remote scheduler나 resume path가 아닙니다.

## Offline은 target을 없애는 것이 아니라 앞당기는 방식이다

오프라인 학습에서는 `data.hidden_states_path`가 미리 생성된 feature checkpoint를 가리킵니다. trainer는 target weights를 로드하지 않고 draft만 학습하므로, 학습 중 target inference를 유지할 필요가 없습니다. 대신 feature 저장 공간과 사전 capture 시간이 필요합니다. 동일한 strategy와 draft config를 사용해 feature를 만들지 않으면 checkpoint schema, target layer, mask가 달라질 수 있습니다.

offline의 장점은 재현성입니다. 같은 manifest를 여러 epoch에 재사용할 수 있고, target server가 매번 살아 있어야 하는 것도 아닙니다. evaluation도 precomputed offline features를 통해 분리할 수 있습니다. 반면 feature가 오래되었거나 target revision이 달라지면 “offline이 안정적이다”라는 장점이 사라집니다. target, tokenizer, renderer, draft config의 provenance를 함께 보존해야 합니다.

## H200에서 비교한 것은 무엇인가

대여한 H200 기능 검증에서는 동일한 13-token Ling 입력을 online과 offline 경로로 흘려보냈습니다. 캠페인 설정의 target taps 3, 7, 11, 15, 19에 해당하는 feature tensor가 bitwise equal이고 최대 절대 오차는 0이었습니다. 같은 캠페인에서 GPU0에는 target TP1 capture, GPU1에는 draft training을 두었고, 100 finite optimizer steps와 128 produced references, zero failed prompts를 기록했습니다.

이 결과를 “online이 offline보다 빠르다” 또는 “GPU1이면 어떤 draft든 학습된다”로 번역하면 안 됩니다. 비교한 것은 하나의 짧은 exact-token 입력과 한 target/draft/backend 조합의 기능 일치입니다. feature parity는 학습 품질, held-out acceptance, 사용자 latency를 측정하지 않습니다. 실제 report도 loss와 teacher agreement를 기능적 learning evidence로만 설명하며, held-out quality improvement나 production acceptance benchmark로 취급하지 않습니다.

별도의 Blackwell 실험에서는 GPU0 Ling target과 GPU1 weights-only fine-tuning을 8 steps 수행했고, 새 export가 한 24-token request에서 target-only greedy output과 24/24 일치했습니다. 그러나 그 캠페인의 native ReplaySSM state parity는 실패했고, 합성 feature를 사용한 여러 CUDA update/reload 결과도 real target-family certification이 아닙니다. 하드웨어별 결과를 합쳐 “GPU0/GPU1 topology 전체가 지원된다”고 말할 수 없는 이유입니다.

## Native HF/vLLM 경로에서 추가로 확인한 것

Qwen3-0.6B와 RTX 3090 두 장을 사용한 후속 캠페인에서는 실제 Mooncake TCP로 teacher GPU0와 trainer GPU1을 연결했습니다. HF BF16은 네 프롬프트의 feature parity와 DSpark·DFlash2 각각 20회 업데이트, exact next-step resume, HF export/reload를 통과했습니다. direct gate에는 GPU1의 별도 frozen HF reference/head도 있으므로 GPU1에 draft만 존재했다고 설명하지 않습니다.

vLLM BF16 feature parity는 최대 절대 오차 32로 실패했습니다. FP32와 deterministic trainer 조건에서는 parity와 학습·재개를 통과했지만 BF16 실패를 지우지 않습니다. managed HF/vLLM CLI의 각각 두 번 업데이트 성공 역시 feature parity나 서비스 재시작 보장이 아닙니다. 자세한 pin과 조건은 [모델 검증 기록](../docs/MODEL_VALIDATION.md)에 있습니다.

이 글의 online은 live feature capture를 뜻합니다. 운영 트래픽 자동 수집·OpenTelemetry 연동·지속 재학습·자동 배포는 미래 방향이며 현재 미구현입니다.

## Online에서 실제로 자주 실패하는 지점

첫 번째는 이미지와 패치의 불일치입니다. 이전 Ling 실행에서는 사용한 이미지의 SGLang revision에 SpecForge capture sink와 `--enable-spec-capture`가 없어 online feature capture가 실행되지 않았습니다. 패치를 강제로 적용하지 않고 실패로 기록했습니다. target 이름과 이미지 tag가 같아 보여도 capture API와 backend revision을 독립적으로 확인해야 합니다.

두 번째는 lifecycle freshness입니다. 새 online attempt에서 이전 consumer database의 SQLite/WAL/SHM을 재사용하면 producer와 consumer의 ack 경계가 과거 run과 섞일 수 있습니다. 반대로 정확한 resume에서는 원래 control directory, store id, run id, consumer state, checkpoint를 함께 보존해야 합니다. producer 자체를 resume하는 계약이 아니라 consumer가 retained stream의 미확인 tail을 이어받는 계약입니다.

세 번째는 partial quantum입니다. producer가 보낸 참조 수가 world size, batch size, accumulation steps가 정한 global quantum을 채우지 못하면 incomplete optimizer step을 억지로 실행해서는 안 됩니다. typed launcher는 이 경계를 schedule과 EOF 처리에 반영합니다.

## 이 topology를 검증할 최소 매트릭스

먼저 target tokenizer와 renderer에서 exact token/mask parity를 검증합니다. 다음으로 동일 입력의 online/offline feature parity를 비교합니다. 그다음 GPU0 target/GPU1 training을 실제로 동시에 실행해 finite step과 prompt accounting을 확인합니다. checkpoint boundary에서 export하고, exported draft를 실제 backend에 다시 로드해 target-only tokens와 비교합니다. 마지막으로 multi-request, radix cache, CUDA graph, 다른 TP, 다른 model family를 지원한다고 말하려면 각각 별도 gate를 추가합니다.

이 매트릭스가 있는 경우에만 online/offline의 trade-off를 독자에게 정직하게 설명할 수 있습니다. online은 target inference를 함께 유지하는 대신 storage 부담이 낮고, offline은 target을 학습 시점에서 분리하는 대신 feature storage가 커집니다. 어느 쪽도 자동으로 더 좋은 품질이나 더 빠른 serving을 의미하지 않습니다.

## 근거와 범위

수치와 지원 범위는 [현재 모델 검증 기록](../docs/MODEL_VALIDATION.md)을 기준으로 읽습니다. 과거 캠페인의 성공은 제품 이름 변경 후 모든 경로를 다시 검증했다는 뜻이 아닙니다. [Draftfit 연재 목차](README.md)로 돌아갈 수 있습니다.

- 로컬 문서: `PUBLIC_WORKFLOW.md`, `PUBLIC_SUPPORT.md`, `basic_usage/disaggregated_training.md`, `RUNTIME_PROFILES.md`
- 로컬 실험: H200 one-hour functional validation, H200 replay report, Blackwell functional campaign
- 미포함: 별도 장기 50K training, 보편적인 GPU 지원, 시장 가격·비용 비교, speedup 또는 품질 향상 주장

## FAQ

### GPU0 target과 GPU1 draft가 항상 최선인가요?

아닙니다. 검증된 한 topology입니다. target TP/EP, draft 메모리, transport, request concurrency에 따라 다른 배치가 필요하며 새 조합은 별도 gate가 필요합니다.

### online feature와 offline feature가 같으면 serving도 같은가요?

아닙니다. feature parity는 학습 입력의 일치입니다. export reload, target token comparison, state replay, 실제 acceptance는 별도 검증입니다.

### Mooncake가 trainer 안에서 자동으로 실행되나요?

기본 external-services mode에서는 그렇지 않습니다. 서비스는 외부에서 준비하고, typed config는 endpoint와 lifecycle contract를 기록합니다.

### 다음에 읽을 글은 무엇인가요?

[타깃 온보딩과 chat mask](draftfit-target-onboarding-chat-masks.md), [H100 native replay](native-replay-h100-67-rounds.md), [checkpoint와 spot recovery](checkpoint-warmstart-resume-spot-backup.md)를 이어서 보세요.
