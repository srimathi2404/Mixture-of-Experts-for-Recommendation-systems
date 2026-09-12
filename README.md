# MoE for Recommendation Systems

Team project on whether Mixture-of-Experts helps recommendation systems. See
`docs/MoE_for_RecSys_Project_Tracks.md` for the full team plan (Tracks A-D) and
`docs/extra_experiments.md` for the add-on experiments referenced by each track.

This repository currently implements **Track C**: self-attention experts for MoSE
(a multi-task sequential recommender), as a follow-up to Track B's LSTM-expert version.
See [`track_c/README.md`](track_c/README.md) for the model, how to run it, and status.

## Setup

```bash
pip install -r requirements.txt   # see requirements.txt for the torch/CUDA install caveat
```

Each track is self-contained under its own `track_*/` directory (independence rule from the
project plan — no track waits on another's code or results).
