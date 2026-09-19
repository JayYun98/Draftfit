---
name: dspark-config
description: 사용자 타깃 모델을 DSpark, DFlash, DFlash2 등의 학습에 연결하기 위해 아키텍처 메타데이터, draft 설정과 호환 runtime을 선택할 때 사용합니다.
---

# 모델과 설정 준비

[환경 선택](../../docs/MODEL_ENVIRONMENTS.md), [설정 확장](../../docs/TARGET_EXTENSION.md),
[지원 경계](../../docs/PUBLIC_SUPPORT.md)를 읽는다. uv 실행/설치는
[환경 가이드](../../docs/ENVIRONMENT.md)를 따른다. 저장소의 수정된 SpecForge는
이미 플랫폼에 포함돼 있으므로 upstream을 별도 설치하지 않는다.

아래 전체 절차는 최초 타깃 준비 또는 타깃/backend 변경에 적용한다. 기존 설정의
학습률·step·데이터 경로만 바꾸는 요청은 해당 필드와 관련 검증만 수행하고,
불필요한 재-inspect·새 프로젝트 생성·runtime 재빌드를 하지 않는다.

1. 타깃 경로 또는 HF ID/정확한 revision, draft 방법, 데이터/feature 소스와 GPU
   조건을 확인한다. 사용자가 이미 고른 모델을 임의로 다른 모델로 바꾸지 않는다.
2. `dspark target inspect`로 config·tokenizer 메타데이터·weight key 후보를 확인한다.
   모델 카드의 주장과 실제 JSON을 구분하고 원격 custom code는 검토 없이 실행하지 않는다.
3. 깊이/hidden width/vocab/embedding·head, attention 종류, MoE 여부, linear/conv/
   recurrent state, chat template/assistant mask 요구사항을 정리한다.
   draft depth나 feature tap 추천은 가설이지 검증된 최적값이 아니다.
4. 기존 backend가 실행·capture·serving/state를 지원하면 환경을 재사용하고 config만
   바꾼다. 지원이 없으면 필요한 backend 변경을 명시한다. config가 새 아키텍처의
   Python/커널 구현을 대신한다고 주장하지 않는다. stock SGLang과 capture build를 구분한다.
5. `dspark algorithms`와 `target prepare --help`를 확인한 뒤 새로운 출력 디렉터리에
   `target prepare`로 초안을 생성한다. 개별 `--set`을 사용하고 데이터/기존 draft는
   전용 인자로 지정한다. 기존 파일을 덮어쓰거나 준비 단계 remote-code 제한을 우회하지 않는다.
6. `dspark train -c PATH --plan`으로 입력 경로·capture layer·runtime·GPU 배치를
   확인한다. online은 Mooncake와 capture 서버가 필요하다. offline은 캐시의 타깃/
   tokenizer/revision/taps 일치를 확인한다. 준비 성공은 실제 feature 검증이 아니다.

결과는 선택한 환경, 생성 config 경로, 수정한 필드와 이유, 남은 실제 검증을 보여준다.
템플릿·feature·state 호환성이 불명확하면 해당 gate를 pending으로 둔다. 모델 이름만
보고 dtype/커널/지원 여부를 확정하지 않는다. 상세 시행은 [훈련 스킬](../dspark-train-validate/SKILL.md).
