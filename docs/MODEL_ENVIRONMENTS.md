# 모델별 환경 선택: 재사용이 기본

커스텀 SpecForge는 특정 모델 한 개를 위한 별도 설치물이 아닙니다. 이 저장소의
수정된 학습 엔진은 `dspark-train-platform` 패키지에 포함됩니다. 다른 모델을
선택해도 같은 엔진을 사용합니다. 원래 SpecForge를 추가 설치하지 마세요.

커스텀 SGLang은 두 이유로 필요할 수 있습니다: 특정 타깃의 실행/state 지원,
그리고 학습용 hidden feature를 Mooncake에 전달하는 capture 기능입니다.
일반 chat API가 동작한다고 capture와 speculative serving까지 지원되는 것은 아닙니다.

| 변경 | 기본 처리 | 확인할 것 |
| --- | --- | --- |
| 데이터·학습률·step만 변경 | 기존 환경 재사용 | 데이터 형식, loss mask, holdout 분리 |
| 같은 지원 아키텍처의 다른 크기/체크포인트 | 환경 재사용 후보; 모델 config 수정 | VRAM, tokenizer/template, 깊이·feature taps·head 경로 |
| draft 방법 DSpark → DFlash2 | 엔진 재설치보다 backend 기능부터 확인 | capture layout, selector/conv export 로딩, 실제 serving 지원 |
| KV → linear/conv/recurrent/hybrid 타깃 | backend 지원과 state 계약 확인 | rollback/replay, capture 경계, 지원되지 않으면 구현 필요 |
| GPU 세대·CUDA·드라이버 변경 | 호환 runtime profile 선택 | 커널 지원, Mooncake wheel, attention backend; config만으로 보장 불가 |
| SGLang source/patch 변경 | 별도 버전 고정 환경 | patch 적용 검사, 관련 회귀 검사와 영향받는 실기능 |

## 현재 환경 선택

- CPU 설정·데이터·테스트: [uv 환경 가이드](ENVIRONMENT.md)의 해시 lock.
- Ling 온라인 학습: [runtime profiles](RUNTIME_PROFILES.md)와
  [정확한 Ling capture revision](../patches/sglang/ling-8ba213f/README.md).
  기록된 base image/source/patch 조합을 재사용합니다. 원본 PyPI SGLang으로
  덮어쓰지 않습니다. 별도 Dockerfile 빌드와 과거 base-image 실험은 다른 증거입니다.
- DFlash2: [지원 경계](PUBLIC_SUPPORT.md)의 serving 요구사항을 확인합니다.
  기존 Ling 이미지가 DFlash2도 지원한다고 가정하지 않습니다.
- 그 외 타깃: [설정 확장 가이드](TARGET_EXTENSION.md)를 따라 backend가 이미
  실행할 수 있는 타깃에 config를 추가합니다. 없는 아키텍처 구현은 설정의 역할이 아닙니다.

## 새 환경이 필요한 경우

기존 환경을 수정하지 말고 별도 venv 또는 별도 이미지 태그를 사용합니다.
타깃 revision, tokenizer/template, 플랫폼 wheel SHA256, backend source SHA,
patch SHA256, 이미지 digest, Torch/CUDA/Mooncake 버전을 기록합니다.
기존 패치와 정확한 source가 맞는지 확인한 뒤에만 설치합니다. 패치 실패를
무시하거나 다른 모델용 패치를 억지로 적용하지 않습니다.

이미 provision된 환경에서는 uv로 플랫폼 wheel만 `--no-deps` 설치합니다.
새 환경에는 먼저 backend의 정확한 의존성을 준비해야 합니다. `--no-deps`는
누락된 의존성을 해결하는 옵션이 아닙니다. CPU lock을 GPU 환경에 sync하지 마세요.
설정 변경과 backend 변경을 구분하고, 과거 통과 항목 전체가 아니라 영향받는
capture/학습/export/serving/state 검사를 다시 실행합니다.

## 저장소의 에이전트 스킬 사용

요청한 `.skill/` 아래에 다음 세 스킬이 있습니다:

- [인스턴스 대여](../.skill/dspark-rent/SKILL.md)
- [모델·config·환경 구성](../.skill/dspark-config/SKILL.md)
- [훈련·검증](../.skill/dspark-train-validate/SKILL.md)

`.skill/`은 저장 위치이며 모든 에이전트의 자동 검색 표준은 아닙니다.
저장소를 연 에이전트에 “`.skill/dspark-config/SKILL.md`를 읽고 진행해줘”처럼
명시적으로 요청할 수 있습니다. 도구별 자동 등록은 그 도구의 skill 경로 설정을
따릅니다. 폴더만 다른 곳에 복사하면 문서 상대 링크가 깨지므로 저장소와 함께
사용하거나 참조 경로를 함께 조정하세요. 이 작업은 전역 스킬을 설치하지 않습니다.
