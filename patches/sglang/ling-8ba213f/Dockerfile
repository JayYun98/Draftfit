# Build from the Draftfit repository root. No CUDA/torch/SGLang upgrades.
FROM lmsysorg/sglang:dev-cu12-Ling-3.0-tiny@sha256:3076afb2abf0e927663e9b7a475277a8b1e87cbd7ae175e22a54ade33aef3e89
WORKDIR /opt/draftfit
COPY pyproject.toml setup.py MANIFEST.in version.txt README.md LICENSE THIRDPARTY_NOTICES.md ./
COPY licenses ./licenses
COPY docs ./docs
COPY draftfit ./draftfit
COPY configs ./configs
COPY examples ./examples
COPY scripts ./scripts
COPY tests ./tests
COPY patches/sglang/ling-8ba213f ./patches/sglang/ling-8ba213f
RUN bash scripts/apply_sglang_spec_capture_patch.sh --target ling-8ba213f \
    && python -m pip install --no-deps uv==0.9.18 \
    && uv pip install --python "$(command -v python)" --no-deps accelerate==1.14.0 yunchang==0.6.4 \
    && uv pip install --python "$(command -v python)" --no-deps --no-build-isolation -e . \
    && python -c "import importlib.metadata as m; from pathlib import Path; assert m.version('draftfit') == Path('version.txt').read_text().strip()" \
    && python -m draftfit.assets list \
    && python -m unittest tests.test_runtime.test_ling_capture_patch -q
