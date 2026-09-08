# Build from the SpecForge repository root. No CUDA/torch/SGLang upgrades.
FROM lmsysorg/sglang:dev-cu12-Ling-3.0-tiny@sha256:3076afb2abf0e927663e9b7a475277a8b1e87cbd7ae175e22a54ade33aef3e89
WORKDIR /opt/specforge
COPY pyproject.toml README.md ./
COPY specforge ./specforge
COPY configs ./configs
COPY examples ./examples
COPY scripts ./scripts
COPY tests ./tests
COPY patches/sglang/ling-8ba213f ./patches/sglang/ling-8ba213f
RUN bash scripts/apply_sglang_spec_capture_patch.sh --target ling-8ba213f \
    && python -m pip install --no-deps accelerate==1.14.0 yunchang==0.6.4 \
    && python -m pip install --no-deps --no-build-isolation -e . \
    && python -m unittest tests.test_runtime.test_ling_capture_patch -q
