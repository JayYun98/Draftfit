# 사람이 설정으로 타깃을 추가하는 방법

타깃 실행은 SGLang 등 추론 백엔드가 담당합니다. 이 프로젝트에서는 사람이
타깃 메타데이터, 캡처 레이어, 기존 draft 구조와 데이터 설정을 지정합니다.
설정을 추가한다고 백엔드에 없는 새 아키텍처의 실행 코드가 생기지는 않습니다.

## 기존 구현을 사용하는 프로젝트 준비

`dspark algorithms`로 등록된 알고리즘, draft 아키텍처와 허용 override를 확인합니다.
아래 예시는 타깃 깊이가 6 이상이고 호환되는 오프라인 feature를 준비할 경우입니다.
`/models/my-target`에는 실제 타깃의 `config.json`과 tokenizer 메타데이터가 있어야 합니다.

```sh
dspark target prepare /models/my-target --local-only \
  --output-dir ./projects/my-target --strategy dspark \
  --hidden-states ./features/my-target \
  --set 'model.target_layer_ids=[1,3,5]' \
  --set model.draft_num_hidden_layers=2
```

기존 draft를 수정해서 쓰려면 `--draft-config ./my-draft.json`을 추가합니다.
JSON의 `architectures`에는 `dspark algorithms`에 표시된 호환 아키텍처 하나를
지정해야 합니다. 기존 draft의 깊이 등은 명시적으로 override한 값만 바뀝니다.
출력 `draft.json`과 `train.json`을 직접 편집할 수도 있습니다.

온라인 데이터는 `--hidden-states` 대신 `--train-data ./train.jsonl`로 지정합니다.
`hf` 렌더러에는 native `{% generation %}` 구간이 필요하며 실제 assistant mask는
별도로 검증해야 합니다. 필요한 경우 등록된 렌더러를 `--set data.chat_template=NAME`으로
선택합니다. 원격 타깃은 저장소 ID와 `--revision`을 사용하며 준비 단계에서 commit SHA를 고정합니다.

`--set`에는 개별 필드 경로를 사용합니다. `model={...}`, `data={...}`처럼 소스가
포함된 부모 객체를 교체하거나 draft/data 소스를 override하면 거부합니다.
소스는 `--draft-config`, `--draft-checkpoint`, `--train-data`, `--hidden-states`로 선택합니다.

`inspect`는 JSON/템플릿 메타데이터만 읽습니다. `prepare`도 원격 모델 코드를
실행하지 않으므로 `--set model.trust_remote_code=true`를 거부합니다.
원격 custom draft config를 실행해야만 읽을 수 있다면 먼저 호환되는 로컬 JSON을
작성합니다. 학습에 원격 코드 실행이 필요한 경우, 코드를 검토한 뒤 생성된
`train.json`에서 `model.trust_remote_code`를 명시적으로 켤 수 있습니다.
학습 설정 자체의 기존 opt-in 기능은 유지됩니다.

## 사람이 작성한 inspection으로 scaffold 생성

```sh
dspark target inspect /models/my-target --local-only > inspection.json
# inspection.json의 recommendations를 검토하고 필요한 후보 값을 수정합니다.
dspark target scaffold --inspection inspection.json --output ./custom-run.json
```

입력 형식은 `target_inspect_v1`입니다. `target_spec_v1` sidecar를 직접 입력하는
명령은 없습니다. `recommendations.target_tap_candidates`, `draft_depth_candidates`,
`train_block_candidates`에서 사람이 선택한 값으로 scaffold를 만들 수 있습니다.
`scaffold`는 설정 초안이며 `prepare`의 draft/타깃 일치 검사를 대신하지 않습니다.
지원 근거가 없는 facts를 임의로 바꾸어 호환성 인증처럼 사용하면 안 됩니다.

## 검증 범위

준비 단계는 등록된 구현, 필드/소스, draft 크기, 캡처 인덱스 등을 검사하지만
feature 파일 내용이나 GPU 실행을 검증하지 않습니다. `manifest.json`의
capture/training/export/serving 상태는 계속 `not_run`입니다.
실제 타깃에서 tokenizer mask, hidden feature의 경계·정규화, greedy token parity,
학습·export·serving을 검증해야 합니다. recurrent/hybrid 타깃은 state snapshot,
rollback, replay까지 실제 백엔드에서 검증해야 합니다.
