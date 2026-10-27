#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
# Keep the optional HTTP service separate from the provisioned trainer runtime.
web_python="${DRAFTFIT_WEB_PYTHON:-$project_root/.venv-web/bin/python}"
if [[ ! -x "$web_python" ]]; then
  if [[ -n "${DRAFTFIT_WEB_PYTHON:-}" ]]; then
    echo "DRAFTFIT_WEB_PYTHON must name an executable Python interpreter." >&2
    exit 1
  fi
  uv venv --python 3.12 .venv-web
  uv pip sync --python "$web_python" requirements-web.lock --require-hashes
fi
if [[ ! -d web/node_modules ]]; then npm ci --prefix web --no-audit --no-fund; fi
npm run build --prefix web
if [[ ! -f .env && -f "${CODEX_HOME:-$HOME/.codex}/global.env" ]]; then
  export DRAFTFIT_FALLBACK_ENV="${DRAFTFIT_FALLBACK_ENV:-${CODEX_HOME:-$HOME/.codex}/global.env}"
fi
exec "$web_python" -m draftfit.web.server --project "$project_root" "$@"
