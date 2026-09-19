---
title: "Draftfit 타깃 온보딩: 설정 파일보다 먼저 확인할 아키텍처와 Chat Mask"
description: "메타데이터 기반 타깃 점검, typed run config, chat template generation span, loss mask 검증을 로컬 근거로 설명합니다."
date: 2026-09-18
slug: "draftfit-target-onboarding-chat-masks"
categories: ["추측 디코딩", "개발 워크플로"]
tags: ["Draftfit", "target inspection", "chat template", "loss mask", "온보딩"]
draft: true
images: []
---

## 답부터 말하면

Draftfit — **Your workload. Your draft.** 내 에이전트가 반복해서 처리하는 요청에 맞는 draft를 만들 수 있을까요? Draftfit은 사용자가 준비한 프롬프트·응답으로 speculative draft를 학습하거나 미세조정하고, 별도 평가 데이터에서 확인하는 워크플로를 지향합니다. target 모델 자체의 답변 품질을 개인화하거나 속도를 보장한다는 뜻은 아닙니다.

새 타깃을 온보딩할 때 가장 먼저 필요한 것은 GPU를 빌리거나 예제 YAML을 복사하는 일이 아닙니다. 타깃 메타데이터, draft 구조, 데이터 스키마, chat template의 assistant generation span, feature tap을 하나의 실행 계약으로 맞추는 일입니다. 설정 계획이 통과해도 실제 타깃 feature/state parity나 serving compatibility가 증명되는 것은 아닙니다.

이 글은 Draftfit의 타깃 준비 워크플로와 보존된 Ling·Qwen 기능 검증을 바탕으로 한 초안입니다. 학습 런타임은 SpecForge에서 파생한 코드를 포함하며 출처를 유지합니다. config로 확장 가능한 구조와 모든 아키텍처의 자동 지원은 다릅니다.

## 1. 하나의 typed run config이 소유해야 하는 것

공개 진입점은 방법별 Python trainer가 아니라 `draftfit train --config run.yaml`입니다. YAML은 target model, draft method와 구조, data source, optimizer, output directory, tracking, runtime, deployment를 함께 기록합니다. online이면 producer가 타깃 feature를 캡처하고 consumer가 draft를 학습합니다. offline이면 미리 저장한 feature checkpoint를 읽고 trainer는 타깃을 로드하지 않습니다.

이 구성이 중요한 이유는 launch topology가 별도 셸 변수에 흩어지면 데이터와 모델 설정이 쉽게 어긋나기 때문입니다. online의 `data.train_data_path` 또는 `data.prompts_path`와 offline의 `data.hidden_states_path`가 모드를 결정합니다. 별도의 scenario flag를 중복으로 두지 않고, `deployment` 아래의 trainer world size와 disaggregated control/store 경로를 같은 실행 문서에 둡니다.

다만 config는 계약의 선언이지 인증서가 아닙니다. 예제에 Ling, Qwen, LFM, Granite가 보인다는 사실만으로 각 target/backend/GPU 조합이 지원되는 것은 아닙니다. 새 조합은 실제 capture, 짧은 training, export reload, serving 또는 state gate를 통과해야 합니다.

## 2. weight-free inspect는 시간을 아껴 주지만 무게를 대신하지 않는다

Draftfit의 target inspector는 configuration, tokenizer metadata, Hub model metadata를 읽고 remote modeling code나 weights를 실행·다운로드하지 않습니다. 이 단계에서 확인할 것은 다음과 같습니다.

- architecture가 dense, MoE, hybrid, recurrent 중 무엇인지
- hidden size, layer 수, embedding/head 이름
- draft가 요구하는 target layer id와 auxiliary tap
- tokenizer의 chat template와 generation span 사용 여부
- target backend가 요구하는 state 종류와 아직 모르는 항목

inspection JSON으로 scaffold를 만들 수 있지만 scaffold는 metadata-only입니다. feature capture, target backend, state replay는 별도의 실험 대상입니다. 이 경계를 문서에서 분리해 두면 “명령이 만들어졌다”와 “모델이 실행된다”를 혼동하지 않게 됩니다.

예를 들어 generic `hf` renderer는 tokenizer template 안에 native assistant `{% generation %}` span이 있어야 합니다. template 문자열이 존재한다는 사실만으로 assistant mask가 올바르게 생성된다고 볼 수 없습니다. generation span이 없는 타깃은 등록된 renderer를 선택하거나 올바르게 pre-tokenized된 입력을 제공해야 합니다.

## 3. 데이터 스키마와 mask는 모델 품질 이전의 경계다

기본 raw JSONL은 `id`와 `conversations` 배열을 사용합니다. 각 메시지는 `user` 또는 `assistant` role과 content를 가져야 합니다. preformatted text를 사용할 수도 있지만 그 경우에도 실제로 사용한 chat template을 명시하고, renderer가 assistant span을 찾아 loss mask를 만들어야 합니다.

mask가 틀리면 학습 loss가 계산된다는 사실만으로는 충분하지 않습니다. system/user 토큰을 target으로 학습하거나 assistant 응답 일부가 잘리면, draft가 target의 실제 생성 분포를 배우고 있다고 말하기 어렵습니다. 긴 입력에서 assistant 구간이 `max_length` 뒤로 밀려 mask token이 거의 남지 않는 문제도 있습니다. 데이터 준비 문서는 이런 행을 사전에 검사하고, 특히 마지막 `block_size`에 유효한 loss 위치가 있는지 확인하도록 설명합니다.

multi-turn reasoning도 같은 원칙을 따릅니다. 한 conversation에 여러 assistant target이 있어도 last-turn-only mask를 쓰면 마지막 응답만 학습될 수 있습니다. 이 경우 generation event 단위로 행을 확장해 각 assistant turn을 별도 학습 사례로 만들고, 이전 reasoning은 노출하지 않는 변환이 필요합니다. 이 절차는 단순한 문자열 변환이 아니라 serving boundary와 학습 target을 일치시키는 일입니다.

## 4. 내 에이전트의 추론도 개인화해야 할까

중요한 질문은 사용자 이름이 붙은 draft를 갖는지가 아니라, 사용자가 실제로 처리하는 요청 분포를 학습과 평가에 반영했는지입니다. 예를 들어 코드 도구 호출이 많은 에이전트와 한국어 문서 요약 중심의 에이전트는 요청 구성과 출력 길이가 다릅니다. 어떤 draft가 유리한지는 각각의 held-out 데이터와 serving 조건에서 측정해야 합니다.

현재 시작점은 사용자가 준비한 데이터입니다. 사용할 권한이 있는 프롬프트·응답만 선택하고 비밀값·개인정보를 검토하며, 대화 단위로 train/eval을 분리해 중복 유출을 줄입니다. 전체 내부 reasoning trace를 무조건 수집할 필요는 없습니다. 실제 모델의 입력·출력 계약을 반영하는 데이터가 우선입니다. 자동 비식별화나 동의 관리 기능이 있다는 뜻은 아닙니다.

OpenTelemetry로 에이전트 트래픽을 자동 수집하고, drift에 따라 지속 학습하고, 검증된 draft를 자동 배포하는 루프는 **미래 방향이며 현재 미구현**입니다. 지금의 train·fine-tune·evaluate 워크플로와 구분해야 합니다.

## 5. 실제 Ling 온보딩에서 드러난 것

로컬 Ling 경로는 renderer와 tokenizer parity를 별도 gate로 취급했습니다. 이후 H200 실험에서 동일한 13-token 입력을 online/offline으로 처리해 taps 3, 7, 11, 15, 19의 feature tensor가 bitwise equal이고 최대 오차가 0인 것을 확인했습니다. 이는 데이터·renderer·capture 계약이 한 짧은 샘플에서 일치했다는 뜻입니다.

반대로 이전 Ling 실행에서는 사용한 Ling 이미지에 SpecForge patched capture sink와 `--enable-spec-capture`가 없어 online capture가 막혔습니다. 소스 revision에 패치를 강제로 적용하지 않았고, 이 결과를 실패 trace로 남겼습니다. “같은 모델 이름”이나 “비슷한 SGLang 이미지”가 capture contract를 보장하지 않는다는 교훈입니다.

또 다른 주의점은 target layer id를 자동 추천값으로 그대로 믿지 않는 것입니다. draft JSON이 요구하는 auxiliary layer와 실제 target backend가 반환하는 feature tap은 명시적으로 비교해야 합니다. architecture scaffold는 시작점을 줄 뿐, real-weight tensor parity를 대신하지 않습니다.

## 6. 실패를 줄이는 최소 온보딩 순서

첫째, local-only inspect로 metadata를 고정합니다. 둘째, fresh output directory에 typed config를 생성하고 `--plan`으로 resolved topology와 redacted paths를 확인합니다. 셋째, CPU renderer test에서 exact token ids, assistant mask 길이, non-empty supervised span을 확인합니다. 넷째, 짧은 real target capture와 offline/online parity를 비교합니다. 다섯째, 한두 번의 finite optimizer update와 export reload를 수행합니다. 마지막으로 target-only와 speculative serving token을 비교하고, 필요한 경우 state replay를 별도 gate로 실행합니다.

이 순서는 “큰 학습을 먼저 하고 나중에 원인을 찾기”보다 싸고 빠릅니다. GPU에 들어가기 전에 잡을 수 있는 schema, template, layer, path 오류를 앞단에서 제거하기 때문입니다. 다만 비용이 낮다고 해서 검증을 생략하는 것은 아닙니다. trust boundary에서 input validation, unknown override rejection, overwrite refusal은 그대로 유지해야 합니다.

## 근거와 범위

수치와 지원 범위는 [현재 모델 검증 기록](../docs/MODEL_VALIDATION.md)을 기준으로 읽습니다. 과거 캠페인의 성공은 제품 이름 변경 후 모든 경로를 다시 검증했다는 뜻이 아닙니다. [Draftfit 연재 목차](README.md)로 돌아갈 수 있습니다.

- 로컬 문서: `PUBLIC_WORKFLOW.md`, `PUBLIC_SUPPORT.md`, `basic_usage/data_preparation.md`, `basic_usage/training.md`
- 로컬 실험: H200 feature parity와 GPU placement report, 2026-09-08 public release preparation report
- 미포함: 별도 장기 50K training, 보편적인 target 지원, 품질 향상·속도 향상 수치

## FAQ

### inspect가 성공하면 바로 학습할 수 있나요?

아닙니다. inspect는 metadata와 제안값을 만들 뿐입니다. 실제 target backend, feature capture, renderer mask, export와 serving gate가 필요합니다.

### chat template가 있으면 mask도 검증된 것 아닌가요?

아닙니다. assistant generation span이 실제로 노출되고, token ids와 loss mask 길이가 일치하며, 학습할 assistant 위치가 남는지 별도 검증해야 합니다.

### offline이면 target을 준비하지 않아도 되나요?

trainer가 target을 로드하지 않는다는 의미에서는 그렇습니다. 그러나 feature checkpoint를 만들 때는 동일한 target revision, renderer, layer contract가 필요합니다.

### 관련된 다음 글은 무엇인가요?

[GPU0 타깃/GPU1 draft topology](gpu0-teacher-gpu1-draft-online-offline.md), [native replay의 H100 근거와 한계](native-replay-h100-67-rounds.md), [정직한 release matrix](honest-release-matrix-packaging-upstream.md)를 이어서 읽을 수 있습니다.
