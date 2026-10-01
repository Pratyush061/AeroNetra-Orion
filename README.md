<div align="center">

# AeroNetra-Orion

### Event-driven predictive perception for UAVs

**Fuse neuromorphic events with conventional frames, forecast the future, and avoid collisions _before_ they happen.**

[![CI](https://github.com/Pratyush061/AeroNetra-Orion/actions/workflows/ci.yml/badge.svg)](https://github.com/Pratyush061/AeroNetra-Orion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

</div>

> **New here?** Read the **[Friendly Guide](GUIDE.md)** — a plain-language walkthrough of what this does, how to run it, how to test it, and what to do with it.

---

## Why this exists

Almost every UAV perception stack is **reactive**: look at the current frame, detect what is
there now, act now. By the time a fast intruder is large enough to be a confident detection, the
time budget to react is nearly gone — and a rolling-shutter RGB frame at speed is smeared by
motion blur exactly when clarity matters most.

**AeroNetra-Orion is anticipatory.** It couples a neuromorphic **event camera** (microsecond
latency, no motion blur, ~120 dB dynamic range) with a conventional frame, uses the event signal
to *restore the detail the frame lost to motion blur*, and drives a **world model** that forecasts
where every object will be over the next second. That forecast becomes a **time-to-collision** and
a **predictive occupancy map** — so the platform can decide before the object is in the way.

> This is the first of four planned breakthroughs. It is a complete, runnable system: clone it and
> it runs end to end with **zero downloads**.

---

## The pipeline

```
   sharp scene ──► DVS event simulation ─┐
                                        ├─► event-guided fusion ─► detector ─► tracker
   blurred frame ───────────────────────┘                                   (Kalman)
                                                                                │
                                                                                ▼
                     risk + occupancy ◄── time-to-collision ◄── world-model forecast
```

| Stage | Module | What it does |
|---|---|---|
| Events | `orion.events` | DVS model (`log`-intensity threshold), event frames, time surfaces |
| Fusion | `orion.fusion` | event-modulated restoration of a motion-blurred frame |
| Detect | `orion.detect` | weight-free motion detector **or** optional YOLOv8 ONNX detector |
| Track | `orion.track` | greedy-IoU association + 8-state Kalman (position **and** size velocity) |
| Predict | `orion.predict` | constant-velocity Kalman forecast **or** optional GRU world model |
| Risk | `orion.ttc` | forward-corridor time-to-collision + predictive occupancy |
| Orchestrate | `orion.pipeline` | threads it all together and writes metrics |

Tracking *size velocity* is the detail that makes anticipation work: it lets the predictor foresee
**looming** — the apparent growth of an object on a collision course.

---

## Quickstart

```bash
git clone https://github.com/Pratyush061/AeroNetra-Orion.git
cd AeroNetra-Orion

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# run the download-free demo
python -m orion.cli demo --out outputs
# or:  orion demo
# or:  python scripts/run_demo.py
```

You will get an annotated video, a metrics file, and a printed summary:

```
AeroNetra-Orion — demo complete
  frames: 72
  events_total: 113,325
  mean_detections: 9.58
  mean_tracks: 9.19
  critical_alerts: 52
  min_ttc_s: 0.033
  mean_recall: 0.702
  elapsed_s: 2.62
  video: outputs/orion_demo.mp4
```

> Numbers are from the built-in synthetic scene and are illustrative of the *pipeline*, not a
> benchmark claim. Reproducibility is enforced: the scene is seeded.

```bash
orion list-datasets          # supported event & aerial datasets
orion run --config configs/default.yaml
```

---

## Datasets — no single benchmark is privileged

The demo needs no data at all. For real work, Orion speaks to both **event** and **aerial**
datasets, deliberately spanning domains rather than leaning on any one benchmark:

| Kind | Datasets |
|---|---|
| Event / neuromorphic | **DSEC**, **MVSEC**, **VisEvent**, DAVIS 240C |
| Aerial / UAV frames | **AU-AIR**, **UAVDT**, **SeaDronesSee**, DroneCrowd, DOTA, CARPK, VisDrone |

Event datasets (DSEC, MVSEC) exercise fusion and flow; **VisEvent** exercises event-based
tracking; aerial datasets (AU-AIR, UAVDT, SeaDronesSee) exercise the detector. See
[`docs/datasets.md`](docs/datasets.md) for the reasoning and the exact folder layout each loader
expects. Loaders are pure parsing functions plus a `require()` guard that fails with a download
link rather than a stack trace.

---

## The optional learned world model

By default the forecast is a physics-based constant-velocity Kalman predictor — always available,
no training. If you want a learned dynamics model:

```bash
pip install -e ".[learn]"                 # adds PyTorch
python scripts/train_world_model.py --epochs 30 --out outputs/world_model.pt
# then in configs/default.yaml:
#   predict:
#     backend: torch
#     torch_model: outputs/world_model.pt
```

The GRU predicts future box offsets from a short window of past boxes. If PyTorch is missing,
Orion silently falls back to Kalman — nothing breaks.

---

## Configuration

Everything is a typed dataclass (`orion.config`) and overridable from YAML
(`configs/default.yaml`). Key knobs: event contrast threshold, scene/ego parameters, fusion
strength, detector thresholds, tracker noise, prediction horizon, and the forward keep-out
corridor (`predict.corridor_*`).

---

## Project layout

```
AeroNetra-Orion/
├── src/orion/          # the package (events, fusion, detect, track, predict, ttc, pipeline, cli)
├── configs/            # YAML configuration
├── scripts/            # run_demo.py, train_world_model.py
├── tests/              # pytest suite (runs without downloads)
├── docs/               # architecture, datasets, innovation notes
└── .github/workflows/  # CI: lint + tests + demo smoke run
```

---

## Testing & quality

```bash
pip install -e ".[dev]"
pytest -q        # 50 tests
ruff check .
```

CI runs lint, the full test suite and a demo smoke run on Python 3.10–3.12.

---

## Relationship to AeroNetra-computervision

[`AeroNetra-computervision`](https://github.com/Pratyush061/AeroNetra-computervision) is the
research **detection & counting** platform (static images, model comparison). Orion is a
*sibling* — same reproducibility values, different question: not *what is here now*, but
*what is about to happen*. The two share nothing at runtime and can evolve independently.

---

## Roadmap — four breakthroughs

1. **AeroNetra-Orion** — event-driven predictive perception ← *you are here*
2. Monocular geo-registration (single-frame object geolocation)
3. Swarm consensus perception (occlusion-resilient multi-UAV counting)
4. Adverse-condition robust fusion (thermal + RGB + dehazing with test-time adaptation)

---

## License

MIT © 2026 Pratyush Jain. See [LICENSE](LICENSE).
