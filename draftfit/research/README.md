# Draftfit research

Choose an experiment below. Each linked guide names the implementation, how to select it, and its tests or recorded results. The verification column describes **what was exercised**; the last column states what that evidence does not establish. Historical model and hardware results apply to the revisions and workloads in their reports.

Run the [research CPU suite](tests/cpu.py) from a source checkout with `python -m draftfit.research.tests.cpu`. For a runnable example, pick a [recipe](../../examples/configs/research/README.md) and run `python -m draftfit.research -c <recipe.yaml> --plan`. Platform experiments use the commands in their own guides.

## Paper methods and measurements

| Research | Verification evidence | Limit or open question |
| --- | --- | --- |
| [Acceptance by output position](papers/acceptance_position/README.md) | CPU tests check position bins and proposal counts. | A diagnostic; it does not measure throughput. |
| [ALR/IRA](papers/alr_ira/README.md) | Regression fixtures exercise rollout preparation and cache reuse. | Full-scale distillation quality is unverified. |
| [Paired reference timing](papers/apple_silicon_measurement/README.md) | Reference tests check token equality before pairing timings. | These checks do not measure native Apple Silicon serving. |
| [ARR](papers/arr/README.md) | Replay and budget tests; the recorded math workload showed no quality gain. | Results do not establish a general improvement. |
| [CAST](papers/cast/README.md) | Reference tree and verifier tests cover temperature and end-of-sequence handling. | No production serving parity claim. |
| [D3D](papers/d3d/README.md) | Corrected targets, loss and gradients checked on CPU/CUDA; small-model training results were mixed. | Full recovery training and asynchronous serving are unverified. |
| [DForkTree](papers/dforktree/README.md) | CPU scoring matched a CUDA formula on fixtures and small-model inputs. | No GPU tree builder, verifier or speed measurement. |
| [Multi-Future Drafting](papers/mfd/README.md) | CPU tests check prefix ranking, ties and the installed recipe. | No mode training, GPU tree construction or target verification. |
| [MoE-SpAc](papers/moe_spac/README.md) | CPU tests check policy decisions on traces. | Expert execution and offload are unverified. |
| [NgramRL](papers/ngramrl/README.md) | Memory tests and an A100 reference run found fewer target calls on repeated text. | Cold and changed inputs did not show the same benefit. |
| [Osprey](papers/osprey/README.md) | Tests cover conversion, tokenizer alignment and training fixtures. | No general model-quality or serving result. |
| [Query layouts](papers/query_layout/README.md) | CPU/CUDA tensor attention comparisons passed. | Native GGML execution is unverified. |
| [REAL loss](papers/real/README.md) | CPU/CUDA loss and gradients checked; one 100-step comparison favored REAL at the endpoint. | One seed does not establish quality or acceptance gains. |
| [Redline](papers/redline/README.md) | CPU tests exercise selection from supplied measurements and recipe output. | Selection does not benchmark the chosen configuration. |
| [RIFT](papers/rift/README.md) | The A100 reference run reduced active cache size but ran slower. | Peak allocation did not fall; no serving speed gain established. |
| [Session lookup](papers/session_lookup/README.md) | CPU tests compare 9,180 small cases with an independent oracle; the installed recipe checks the proposal. | No target verifier, energy or throughput measurement. |
| [SpecMem](papers/specmem/README.md) | Memory isolation and reload tests; an A100 reference run reduced calls on a repeat case. | Lexical embeddings and a small workload limit the result. |
| [TableTune](papers/tabletune/README.md) | CPU recipe checks table selection; recorded CUDA forward profiles exercised raw draft-length selection. | Predicted goodput is not measured serving throughput. |
| [TreeWY](papers/treewy/README.md) | CPU/CUDA comparisons checked Gated DeltaNet state. | Full cache rollback and serving speed are unverified. |
| [ValuSpec](papers/valuspec/README.md) | In the recorded CAST run, fallback matched baseline tokens after all candidates were rejected. | No speed gain was demonstrated. |

See the [reference-run measurements](../../docs/research/results/latest-methods-effects-20261028.json) and the [100-step comparison](../../docs/research/results/paper-components-small100-20261113.md) for workloads, revisions and controls. The [source catalog](catalog.py) also lists papers with no executable integration.

## Platform experiments

| Research | Verification evidence | Limit or open question |
| --- | --- | --- |
| [Asynchronous joint capture](experiments/async_joint_capture/README.md) | Tiny CPU training, queue credit, producer failures and checkpoint resume tested. | CUDA overlap, peak memory and speed are unverified. |
| [Prism GGUF capture](experiments/prism_gguf/README.md) | Fake native API and prefix replay tests cover the bridge contract. | These tests do not exercise the pinned Bonsai target. |
| [MLX offline capture](experiments/mlx_capture/README.md) | Portable tests cover artifact identity and persistence. | Real capture needs the matching Metal runtime and model pack. |
| [MLX evaluation](experiments/mlx_evaluation/README.md) | Report tests reject missing trials and token mismatches. | Aggregation does not run an MLX engine. |
| [Native vLLM teacher](experiments/vllm_teacher/README.md) | Recorded FP32 parity passed; Qwen3-0.6B BF16 parity failed. | No general precision or serving parity. |
| [Quantized-target adaptation](experiments/quantized_target/README.md) | CPU cache isolation and export metadata tested. | The GPU gate did not reach model construction. |
| [Spark prefix verification](experiments/spark_verification/README.md) | Reference fixtures check target routing and prefix stability. | Native parity and serving speed are unverified. |
| [Resident target generation](experiments/resident_generation/README.md) | CPU generation and batch-policy tests include simulated out-of-memory handling. | Actual CUDA tuning performance is unverified. |
| [SGLang milestone verification](experiments/sglang_milestone/README.md) | Controller and gate tests; recorded long-output token parity failures. | The single-device dense-target gate does not certify other targets. |
| [Ling on Ampere](experiments/ling_ampere/README.md) | Profile resolution and a basic SM80 Torch probe. | Fused top-k failed; no SM80 training or replay pass. |
| [EAGLE4](experiments/eagle4/README.md) | The MiniCPM5 extension passed FP32 reference checks but failed strict BF16 parity. | Native EAGLE4 matched 7/8 cases in the recorded gate. |
| [DFlash-family ablations](experiments/family_ablation/README.md) | Family tests cover held-out features, plan overrides and sequential handoff. | Transfer and proxy losses do not establish acceptance. |
| [Raw-text initialization](experiments/raw_text_pretraining/README.md) | Tiny models exercise pretraining and projection widening. | No GPU acceptance or serving comparison against scratch. |
| [Optuna screening](experiments/training_search/README.md) | Controller and SQLite tests exercise study validation and persistence. | The score compares one family; it does not measure speed. |
| [MiniCPM5-2B family](experiments/minicpm5_family/README.md) | Warm-start and family tests; measured exports had token mismatches. | No compatible DFlash2 serving path on the tested A100. |
| [Qwen3.5 hybrid text](experiments/qwen35_text/README.md) | Tiny hybrid text capture through DSpark and DFlash2 updates. | Vision, hybrid rollback and equivalent vLLM features remain open. |
| [Xing4.0](experiments/xing4/README.md) | Capture reduction and YaRN rotary fixtures tested. | Original-weight capture and serving parity are unverified. |
| [ZDTaichu5.0](experiments/taichu5/README.md) | Fixtures check embedding/head transfer and text-decoder selection. | No full-weight, multimodal or serving parity result. |
| [Hemmingway-1](experiments/hemmingway1/README.md) | Recipe metadata and package architecture checked. | Original-weight training and exact serving are unverified. |
| [Agnes-3.0-Flash](experiments/agnes3/README.md) | Text-decoder selection tested on a fixture. | Original weights, hybrid replay and serving remain unverified. |
| [LFM2.5 warm starts](experiments/lfm25/README.md) | Official draft tensors loaded; tiny dense and MoE capture tested. | Original-target capture and served token equality remain open. |
| [Humanizer](experiments/humanizer/README.md) | Input preparation and tiny joint-training tests cover the text path. | No 12B capture, export parity or serving-speed result. |
