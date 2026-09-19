# Draftfit editorial drafts

**Your workload. Your draft.**

사용자가 준비한 개인·팀 워크로드의 프롬프트와 응답으로 speculative draft를
학습하고, 기존 draft를 미세조정하고, 학습에 쓰지 않은 데이터로 평가하는
Draftfit의 방향을 설명하는 초안 모음입니다. DSpark는 제품명이 아니라 플랫폼
안의 draft 방법 중 하나입니다. SpecForge는 파생 코드의 출처로 구분합니다.

게시 가능한 글 7편은 모두 `draft: true`입니다. 이 목차와 복사용 LinkedIn
모음은 편집 자료이지 추가 게시글이 아닙니다. 외부 게시·사이트 생성·배포는 하지
않았으며, 도메인이나 canonical URL, 공개 릴리스 주소를 임의로 정하지 않았습니다.

## 한국어 기술 연재

1. [내 데이터와 타깃 준비: 아키텍처·chat mask·개인화 방향](draftfit-target-onboarding-chat-masks.md)
2. [GPU0 teacher / GPU1 draft: online과 offline feature 경로](gpu0-teacher-gpu1-draft-online-offline.md)
3. [Native replay: 학습과 분리한 67 state rounds / 144 tokens 검증](native-replay-h100-67-rounds.md)
4. [기존 draft 미세조정과 중단된 학습 복구의 차이](checkpoint-warmstart-resume-spot-backup.md)
5. [내 워크로드 평가와 정직한 검증 매트릭스](honest-release-matrix-packaging-upstream.md)

## LinkedIn

- [English draft](linkedin-draftfit-release-en.md)
- [한국어 초안](linkedin-draftfit-release-ko.md)
- [언어별 복사용 본문](LINKEDIN_READY.md)

두 LinkedIn 본문은 같은 메시지의 언어별 버전입니다. 게시할 때는 한 언어를
선택하고 YAML, 편집 참고, FAQ는 제외합니다. 실제 다국어 사이트나 hreflang을
구성한 것은 아닙니다.

## 공통 편집 경계

- 실제 수치와 실패는 [MODEL_VALIDATION](../docs/MODEL_VALIDATION.md)을 기준으로
  대조합니다. 과거 캠페인은 새 브랜드·현재 배포물의 GPU 재검증이 아닙니다.
- train·fine-tune·evaluate는 사용자 데이터와 명시적 실행·평가 조건을 전제로
  합니다. 데이터 자동 수집이나 자동 개인정보 정제를 뜻하지 않습니다.
- OpenTelemetry 연동, 운영 트래픽 자동 수집, 지속 재학습, 자동 배포·롤백은
  미래 방향이며 현재 미구현입니다.
- 범용 아키텍처 지원, 품질·속도 보장, production readiness를 주장하지 않습니다.
  특히 vLLM BF16 feature parity와 RTX 5090 native replay 실패를 유지합니다.
- 실제 게시 전 author/contact, canonical URL, 이미지·alt, 사이트 메타데이터와
  모바일 렌더링을 확인해야 합니다. 현재 상대 링크는 저장소 안에서 읽기 위한
  링크이며, 게시 사이트의 최종 URL로 변환해야 합니다.
