# kernel-cost-analysis (placeholder)

This directory marks a Claude Code **skill** the KDA-for-Trainium workflow uses,
but its contents are intentionally omitted from this repository — the skill lives
in a separate NKI kernel library that is not part of this open-source release.

To use it, drop the skill's `SKILL.md` (and any supporting files) into this
folder, or symlink it from your local NKI kernel library, e.g.:

    ln -sfn /path/to/nki-kernel-library/.claude/skills/kernel-cost-analysis .claude/skills/kernel-cost-analysis

See the top-level README (Setup) and CLAUDE.md (Skills) for what each skill
provides and which are safe to use without a local build workflow.
