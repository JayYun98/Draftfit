Draftfit Documentation
======================

**Your workload. Your draft.**

Train and fine-tune speculative decoding drafts on your own data. Draftfit
builds on the SpecForge engine; it is maintained independently, not by the
SGLang team. Export compatibility and inference speedups require validation
for the particular model, backend and workload.

Start with the maintained workflow and support boundaries below. Older
backend-specific tutorials in this documentation tree are reference material,
not a blanket compatibility or production-readiness claim.

.. toctree::
   :maxdepth: 1
   :caption: Start here

   get_started/about
   get_started/installation
   ENVIRONMENT
   PUBLIC_WORKFLOW
   PUBLIC_SUPPORT

.. toctree::
   :maxdepth: 1
   :caption: Validation and runtime boundaries

   RUNTIME_PROFILES
   MODEL_VALIDATION
   RELEASE_GATES
   PERFORMANCE_GATE
   PUBLIC_RELEASE

.. toctree::
   :maxdepth: 1
   :caption: Ownership and attribution

   CODE_OWNERSHIP
   SOURCE_ATTRIBUTION

.. toctree::
   :maxdepth: 1
   :caption: Usage and extension reference

   basic_usage/data_preparation
   basic_usage/training
   basic_usage/disaggregated_training
   advanced_features/customization
   examples/llama3-eagle3-online
   examples/llama3-eagle3-offline

.. toctree::
   :maxdepth: 1
   :caption: Inherited concepts, tutorials and historical results

   concepts/speculative_decoding
   concepts/EAGLE3
   basic_usage/AMD/amd_rocm
   benchmarks/benchmark
   benchmarks/eagle3-disaggregated-parity
   benchmarks/domino-disaggregated-performance
   community_resources/specbundle
   community_resources/dashboard

Source and issues: `Draftfit repository <https://github.com/JayYun98/dspark-train-platform>`_.
