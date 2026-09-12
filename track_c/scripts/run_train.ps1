# Convenience launcher for Track C training (Windows/PowerShell).
# Run from the repo root: .\track_c\scripts\run_train.ps1
# NOTE: development/training now happens on a Linux GPU box - scripts/*.sh
# are the maintained entry points; this is kept only for local Windows dev.

python -m track_c.src.train $args
