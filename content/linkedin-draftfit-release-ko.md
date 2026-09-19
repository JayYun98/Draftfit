---
title: "Draftfit: 내 워크로드에 맞는 speculative draft 만들기"
description: "개인·팀의 데이터로 draft를 학습·미세조정·평가하는 Draftfit의 방향과 실제 GPU 근거의 한계를 소개합니다."
date: 2026-09-18
slug: "draftfit-workload-specific-drafts-ko"
categories: ["speculative-decoding", "Draftfit"]
tags: ["Draftfit", "추측 디코딩", "draft 학습"]
draft: true
images: []
---

Draftfit — Your workload. Your draft.

내 워크로드를 반영한 프롬프트와 응답으로 speculative draft를 학습하고, 기존 draft를 미세조정하고, 결과를 평가하는 Draftfit을 만들고 있습니다.

출발점은 “모두에게 가장 빠른 draft는 무엇인가?”가 아닙니다. “내가 실제로 처리하는 요청에 이 draft가 맞는가?”입니다.

그래서 target revision, chat template, assistant mask, captured feature, 학습에 쓰지 않은 평가 데이터를 함께 관리해야 합니다. 학습 loss가 낮아졌다는 사실만으로 acceptance나 실제 serving latency가 좋아졌다고 말할 수는 없습니다.

제한된 실제 GPU 근거는 있습니다. Qwen3-0.6B와 RTX 3090 두 장에서 native HF teacher 경로는 네 프롬프트의 feature parity, DSpark·DFlash2 각각 20회 업데이트, 정확한 다음 step 재개, HF export/reload를 통과했습니다. vLLM은 FP32 parity와 deterministic-training gate를 통과했지만, BF16 feature parity는 실패했고 그대로 실패로 남깁니다. 이는 짧은 기능 검사이지 워크로드 품질이나 속도 벤치마크가 아닙니다.

Draftfit은 SpecForge에서 파생한 자체 학습 런타임을 포함하며 TorchSpec·AngelSpec의 설계를 참고했습니다. DSpark는 플랫폼 안의 draft 방법 중 하나이지 제품 이름이 아닙니다.

OpenTelemetry를 통한 운영 트래픽 자동 수집, 지속 재학습, 자동 배포는 앞으로 탐색할 방향이며 현재 구현된 제품 기능이 아닙니다. 지금은 사용자가 데이터를 준비하고 학습 실행과 평가 기준을 선택합니다.

작은 테스트의 성공을 범용 지원으로 확대하지 않으면서, 특정 워크로드에 맞는 draft를 만드는 과정을 기록하고 있습니다. 비슷한 문제를 다루는 분들과 경험을 나누고 싶습니다.

#Draftfit #SpeculativeDecoding #머신러닝

## 편집 참고 — 게시 본문에서 제외

[검증 근거와 한계](../docs/MODEL_VALIDATION.md), [연재 목차](README.md), [타깃과 마스크 준비](draftfit-target-onboarding-chat-masks.md), [실시간·저장 feature 비교](gpu0-teacher-gpu1-draft-online-offline.md).

아직 게시하지 않은 초안입니다. 공개 URL이나 릴리스 제공을 약속하지 않습니다.

## FAQ

**속도 향상을 보장하나요?** 아닙니다. held-out acceptance와 target-only 대비 draft-assisted serving 실측이 별도로 필요합니다.

**모든 아키텍처에서 사용할 수 있나요?** 아닙니다. 타깃·백엔드·방법·하드웨어 조합마다 검증해야 합니다.
