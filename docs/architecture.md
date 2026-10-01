# Architecture

## Data flow

```
SceneFrame ─┬─► sharp ─► simulate_events ─► EventStream ─┐
            │                                             ├─► EventFrameFusion.fuse
            └─► gray (blurred) ──────────────────────────┘        │
                                                                    ▼
                                                        FusedFrame.sharpened
                                                                    │
                                                          Detector.detect ─► [Detection]
                                                                    │
                                                      IoUKalmanTracker.update ─► [Track]
                                                                    │
                                                     Predictor.predict ─► (T,4) boxes
                                                                    │
                                       assess_tracks ─► [RiskAssessment] ─┬─► occupancy_map
                                                                          └─► min_ttc
```

## Modules

| Module | Responsibility | Key types |
|---|---|---|
| `config` | Typed, YAML-overridable configuration | `OrionConfig`, `load_config` |
| `events` | DVS simulation and event representations | `EventStream`, `simulate_events` |
| `scene` | Deterministic, download-free aerial scene | `SyntheticAerialScene`, `SceneFrame` |
| `fusion` | Event-guided restoration of blurred frames | `EventFrameFusion`, `FusedFrame` |
| `detect` | Weight-free and ONNX detectors | `Detection`, `MotionDetector`, `OnnxDetector` |
| `track` | IoU association + Kalman smoothing | `KalmanBox`, `Track`, `IoUKalmanTracker` |
| `predict` | Short-horizon forecasting | `KalmanPredictor`, `TorchWorldModel` |
| `ttc` | Collision risk + predictive occupancy | `RiskAssessment`, `assess_tracks` |
| `pipeline` | Orchestration and metrics | `OrionPipeline`, `run_demo` |
| `viz` | Drawing (no GUI dependency) | `annotate` |
| `cli` | Command line | `main` |

## Design decisions

**One detection contract.** Every detector returns `list[Detection]`, so the tracker, predictor and
risk layers never branch on which model produced a box. Swapping in a learned detector is a
one-line config change.

**Size velocity in the Kalman state.** The state is `[cx, cy, vx, vy, w, h, vw, vh]`. Carrying
`vw, vh` is what allows the predictor to extrapolate looming — without it, a monocular
time-to-collision is impossible.

**Graceful degradation everywhere.** No ONNX model → classical detector. No PyTorch → Kalman
predictor. No dataset → synthetic scene. No video codec → skip the video, keep the metrics. The
project always runs.

**Determinism.** The scene is seeded, so the demo and its metrics are reproducible run to run.
