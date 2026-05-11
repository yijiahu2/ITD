# Project Agent Instructions

This repository keeps Everything Claude Code (ECC) installed in-project.
Keep this root `AGENTS.md` minimal so Codex gets routing rules without loading
the full ECC instruction surface on every run.

## Default Search Policy

For code modification, code review, and debugging tasks, start in:

- `ITD_agent/`
- `scripts/`
- `tests/`
- `input_layer/`
- `output_layer/`
- `tools/`
- `runtime_entrypoints/`

## Default Docs Policy

Only escalate to docs or config indexes when source inspection is insufficient.
Read in this order:

1. `docs/codemap.md`
2. `docs/README.md`
3. `configs/README.md`
4. `configs/examples/`
5. `configs/templates/`

## Do Not Scan By Default

Do not read these paths unless the task explicitly requires runtime artifacts,
historical experiments, dataset inspection, or Codex/ECC configuration work:

- `outputs/`
- `tmp_debug/`
- `data/`
- `.agents/`
- `.codex/`

## Targeted Exceptions

Access to any default-blocked path is allowed when the task explicitly requires
it, but use targeted reads instead of broad scans.

- If runtime debugging needs artifacts, read only the named run directory or
  specific summary/trace file under `outputs/`.
- If temporary debug evidence is required, read only the named file or
  subdirectory under `tmp_debug/`.
- If dataset inspection is required, read only the named file or subdirectory
  under `data/`.
- If ECC skill behavior must be inspected, read only the specific skill file
  under `.agents/skills/`.
- If Codex configuration or agent behavior must be inspected, read only the
  specific file under `.codex/`.
- Never start with repo-wide search across blocked paths.
- When escalating into a blocked path, state which path is being opened and why.

## ECC And Skills

- ECC is installed in-project under `.agents/`.
- Read `.agents/skills/*.md` only when a skill is explicitly triggered by the
  user or clearly required by the task.
- Use `.codex/AGENTS.md` only when Codex-specific supplement rules are needed.

## Runtime Artifacts

- Prefer source files over generated summaries, traces, and runtime configs.
- Read outputs, traces, JSON summaries, and generated YAML only when the task is
  specifically about experiment results or runtime debugging.
- Prefer targeted paths over repo-wide scans.

## Working Style

- Use targeted search and file reads instead of broad scans.
- If source inspection is enough, do not escalate to long docs or runtime
  artifacts.
- When asking for more context, first report which paths were already checked.
