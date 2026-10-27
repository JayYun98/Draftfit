Architecture diagrams

Editable sources: *.mmd (Mermaid)
Published images: *.svg (rendered, no browser-side JavaScript required)
Renderer used: @mermaid-js/mermaid-cli 12.0.0

Render each source with:
  mmdc -i NAME.mmd -o NAME.svg -c mermaid-config.json -b white

Use the renderer's -p option to point to an installed Chrome executable when
needed. Keep machine-specific browser paths outside the repository. The docs
build consumes the committed SVGs and does not require this renderer.

The family-* diagrams preserve the independent training/conversion routes.
The joint-lifecycle and milestone-verification diagrams describe the current
joint/unattended paths. Native diagnostic timings and bounded serving smoke
are not benchmark-quality speed policy inputs.
