---
title: "Draftfit Checkpoint, Warm Start, Resume: GPU 실험을 복구 가능한 런으로 만들기"
description: "weights-only warm start와 full resume를 구분하고, H200 crash recovery와 spot backup에서 배운 최소 복구 계약을 정리합니다."
date: 2026-09-18
slug: "checkpoint-warmstart-resume-spot-backup"
categories: ["분산 학습", "MLOps", "릴리스 엔지니어링"]
tags: ["checkpoint", "warm start", "resume", "spot GPU", "Mooncake", "복구"]
draft: true
images: []
---

## 답부터 말하면

Draftfit — **Your workload. Your draft.** 내 데이터로 새 draft를 학습할 때, 기존 draft를 미세조정할 때, 중단된 동일 런을 복구할 때는 서로 다른 시작점이 필요합니다. 이 글은 그중 fine-tune과 resume의 경계를 다룹니다.

weights-only warm start와 training resume는 같은 기능이 아닙니다. `model.draft_checkpoint_path`는 새 런을 draft weights에서 시작할 뿐이고, `training.resume_from`은 optimizer/scheduler, step·epoch·data position, per-rank RNG까지 복원해 같은 런을 이어갑니다. 제가 대여한 H200에서 checkpoint 25에 consumer를 의도적으로 종료한 뒤 optimizer moments, FP32 master weights, scheduler, CPU/CUDA RNG를 정확히 복원하고 step 26–100을 한 번씩 완료했습니다.

하지만 이 결과는 원래 producer, target server, Mooncake store가 살아 있는 single-rank consumer crash recovery입니다. cold host, store 손실, historical ledger mismatch, cross-world optimizer resharding까지 증명한 것은 아닙니다. Spot 환경에서는 무엇을 백업했고 무엇을 복구하지 못했는지도 릴리스 결과의 일부로 기록해야 합니다.

## Warm start와 resume를 먼저 분리하라

호환되는 draft architecture를 기존 가중치로 초기화하거나, 이미 학습된 draft를 다른 데이터로 fine-tune하려면 warm start를 사용합니다. 이 경로는 draft weights만 읽습니다. 새 optimizer를 만들고, step counter와 schedule을 새로 시작하며, data progression과 RNG도 복원하지 않습니다. warm source에 `config.json`이 있으면 그 draft architecture를 가져올 수 있지만, 이것은 training state를 가져오는 것이 아닙니다.

반대로 중단된 동일한 training run을 이어가려면 full checkpoint가 필요합니다. `training.resume_from`은 runtime checkpoint를 읽고 optimizer/scheduler와 step/data position, rank별 RNG를 복원합니다. `model.draft_checkpoint_path`와 동시에 지정하지 않는 이유도 여기에 있습니다. 둘을 함께 쓰면 “새 런으로 시작할지, 기존 런을 재개할지”가 모호해집니다. 공개 CLI는 이 충돌을 early validation에서 거부합니다.

또한 세계 크기가 달라진다고 자동으로 안전해지지 않습니다. optimizer/FSDP checkpoint는 현재 같은 trainer world size가 필요하고, 2→1 또는 1→2 cross-world resharding은 지원 범위 밖입니다. control-plane ref redistribution이 optimizer state resharding을 대신하지 않습니다.

## H200 warm resume가 실제로 확인한 것

제가 대여한 H200 online run은 `max_steps=100`, `total_steps=100`, save interval 25를 사용했습니다. 실제 checkpoint 25 저장이 끝난 뒤 SQLite `global_step=25`, `optimizer_durable=true`, `acked_count=25`를 확인하고 consumer process에 의도적인 종료를 주입했습니다. 이 종료는 정상 completion/failure signaling을 우회해 producer가 아직 소비되지 않은 feature를 유지하도록 만든 테스트 조건입니다.

새 consumer는 동일한 ledger, references, live store, checkpoint 25로 시작했습니다. backend restoration 직후 Adam moment/step tensor, FP32 master weight, 완전한 scheduler payload, CPU RNG와 바인딩된 CUDA RNG를 checkpoint와 비교했습니다. 모두 exact equality를 통과했습니다. 이후 step 26부터 100까지 75 optimizer steps를 빠짐없이 한 번씩 기록했고, loss와 CE/L1/confidence component, gradient norm, learning rate가 모두 finite였습니다. 마지막 SQLite marker는 global step 100, durable true, acked count 100이었습니다.

이 결과는 “resume 파일을 읽었다”보다 강한 근거입니다. optimizer가 같은 다음 상태에서 출발하는지, checkpoint step과 durable acknowledgement가 맞는지, tail reference를 중복 소비하지 않는지를 함께 확인했기 때문입니다. 동시에 producer와 target/capture server와 Mooncake가 계속 살아 있어야 했다는 사실도 같은 크기로 기록해야 합니다.

## Online resume는 consumer-only 계약이다

온라인 disaggregated run의 producer는 training checkpoint를 resume하지 않습니다. 새 consumer가 retained control directory, store id, run id, consumer state, SQLite/WAL, Mooncake objects와 matching checkpoint를 사용해 확인된 reference는 건너뛰고 미확인 tail을 이어받습니다. rank 0의 durable optimizer marker가 checkpoint step과 다르면 재개를 허용하지 않는 이유는, 이미 소비한 샘플을 다시 학습하거나 optimizer state와 stream을 어긋나게 만들 수 있기 때문입니다.

새로운 online attempt에는 fresh control directory와 consumer-state가 필요합니다. 이전 SQLite/WAL/SHM을 새로운 run처럼 재사용하면 과거 ack와 현재 checkpoint가 섞입니다. 반대로 exact resume에서는 기존 state를 유지해야 합니다. “fresh start”와 “resume”의 저장소 규칙이 반대이므로, 운영 runbook에서 둘을 같은 cleanup 절차로 다루면 안 됩니다.

오프라인 disaggregated resume도 consumer-only입니다. producer가 고정 manifest와 feature store를 재생성하거나 재개하는 것이 아니라, retained feature stream을 같은 training contract로 읽습니다. 새로운 model revision이나 draft config로 기존 feature cache를 재사용하면 checkpoint restore가 성공해도 의미가 바뀔 수 있습니다.

## Spot 중단과 백업 경계

Spot run은 훈련 코드만큼 resource lifecycle을 기록해야 합니다. exact instance ID, target/draft hash, config, timestamps, checkpoint summary, exit code, small logs를 local manifest에 남기고, 그 ID만 destroy한 뒤 instances와 volumes가 비었는지 다시 조회하는 식입니다. broad cleanup이나 label 추정으로 다른 실험을 지우면 안 됩니다.

백업은 복구에 필요한 작은 자산을 우선합니다. summary, manifests, hashes, requests, server logs, checkpoint metadata는 빠르게 보관합니다. 대형 target weights, multi-gigabyte optimizer shards, 반복 state tensors는 저장·전송 비용과 복구 필요성을 비교해 제외할 수 있습니다. 단, “백업하지 않았다”는 사실은 결과의 한계입니다.

실제 Blackwell 캠페인에서는 spot interruption 중 약 370MB export transfer가 중단됐고 새 export weights는 회수하지 못했습니다. 성공적으로 확인된 serving 결과와 로컬에 retained된 이전 export는 구분해 기록했습니다. H200 campaign도 마지막 export inference가 중단되어 그 단계의 완료를 주장하지 않았습니다. 중단 원인을 확인하지 못했다면 outbid나 preemption으로 단정하지 않는 것이 원칙입니다.

## 실패에서 배운 복구 교훈

제가 겪은 가장 큰 실패는 “체크포인트가 있다”와 “복구 가능한 런이다”를 같은 말로 취급하는 것이었습니다. 새 export가 전송 중 사라지면 serving 결과와 파일 보존을 분리해서 말해야 하고, consumer가 살아난다고 해서 producer·store·ledger 경계가 자동으로 맞는 것도 아닙니다. 그래서 저는 checkpoint step, durable ack, store identity, world size, 백업된 작은 증거를 한 묶음으로 기록합니다.

## 복구를 설계하는 최소 체크리스트

1. save interval을 optimizer boundary와 맞추고, complete checkpoint만 `latest`로 가리킵니다.
2. checkpoint 저장 직후 durable marker, optimizer step, ack count를 함께 기록합니다.
3. warm start인지 resume인지 한 번에 하나만 선택합니다.
4. resume에는 동일 world size, 동일 run/control/store identity, matching draft config를 요구합니다.
5. online resume에서는 producer를 재시작하지 않고, retained references와 durable ack 경계를 확인합니다.
6. 새 attempt에는 fresh control/state path를 만들고, resume attempt에는 기존 path를 보존합니다.
7. spot 종료 뒤 작은 증거를 먼저 백업하고 exact resource ID만 teardown합니다.
8. 복구하지 못한 대형 자산과 아직 검증하지 않은 cold-store recovery를 release matrix에서 명시적으로 제외합니다.

이 체크리스트는 인프라를 복잡하게 만드는 장식이 아닙니다. GPU 시간이 비싸서가 아니라, 학습 결과와 데이터 소비 위치가 한 번 어긋나면 다시 재현하기 어려운 종류의 오류를 막기 위한 최소 계약입니다.

## 근거와 범위

수치와 지원 범위는 [현재 모델 검증 기록](../docs/MODEL_VALIDATION.md)을 기준으로 읽습니다. 과거 캠페인의 성공은 제품 이름 변경 후 모든 경로를 다시 검증했다는 뜻이 아닙니다. [Draftfit 연재 목차](README.md)로 돌아갈 수 있습니다.

- 로컬 문서: `PUBLIC_WORKFLOW.md`, `basic_usage/training.md`, `basic_usage/disaggregated_training.md`, `OPERATIONS_VAST.md`
- 로컬 실험: H200 warm resume report, H200 validation report, Blackwell campaign, public release preparation report
- 미포함: cold infrastructure-loss recovery, cross-world optimizer resharding, 별도 장기 50K training, 외부 클라우드 복구 보장

## FAQ

### warm start로 optimizer도 이어지나요?

아닙니다. warm start는 draft weights만 초기화합니다. optimizer, scheduler, counters, data position, RNG는 새로 시작합니다.

### resume checkpoint만 있으면 온라인 run을 어디서나 재개할 수 있나요?

아닙니다. 현재 근거는 원래 producer/server/Mooncake가 살아 있는 consumer crash recovery입니다. retained ledger/store와 matching checkpoint가 필요합니다.

### world size를 바꿔서 더 작은 GPU로 복구할 수 있나요?

현재 계약에서는 optimizer/FSDP cross-world resharding을 지원하지 않습니다. 같은 trainer world size를 유지해야 합니다.

### 관련 글은 무엇인가요?

[GPU0 target/GPU1 draft와 online/offline](gpu0-teacher-gpu1-draft-online-offline.md), [H100 native replay의 별도 상태 gate](native-replay-h100-67-rounds.md), [정직한 release matrix](honest-release-matrix-packaging-upstream.md)를 함께 보세요.
