# The innovation, stated honestly

"Never done before" is a strong claim, and it deserves a careful, honest answer. Each *ingredient*
below has prior art in isolation. The contribution of AeroNetra-Orion is the **coherent,
reproducible system** that couples them for UAVs, and the **anticipatory safety output** it makes
available as a runnable artefact.

## The idea

A reactive perception stack answers *"what is here now?"*. Orion answers *"what is about to
happen, and will it collide with me?"* — by fusing an event camera's blur-free, microsecond signal
with a conventional frame and running a world model over the result.

## Ingredients and prior art (be candid)

| Ingredient | Prior art exists | Orion's contribution |
|---|---|---|
| Event-based motion deblurring | Yes (event-guided deblurring literature) | A lightweight, event-*activity*-modulated restoration, fast and dependency-free |
| Event + frame fusion | Yes | One fused tensor feeding a single detection contract |
| Kalman / constant-velocity tracking | Yes | Size-velocity state enabling monocular looming |
| Trajectory forecasting / world models | Yes (large-scale driving world models) | A tiny, optional GRU with a physics fallback, trained on-device |
| Monocular time-to-collision | Yes | Corridor-intrusion TTC wired to a **predictive** trajectory |
| Predictive occupancy | Yes | Derived directly from forecast boxes |

## What is genuinely new here

1. **The coupling.** Event-guided restoration feeds a size-aware tracker whose forecast feeds a
   corridor-based TTC — as one reproducible pipeline, for UAVs, with no external model weights
   required to run.
2. **Anticipation as the safety primitive.** The primary output is a *future* collision risk, not
   a present-frame detection.
3. **Runnable by default.** The whole system executes with zero downloads and zero GPU, which is
   unusual for anything event-based.

## Falsifiable claims

- A size-velocity Kalman state produces finite monocular TTC on looming objects; a position-only
  state does not. (See `tests/test_predict.py::test_predictor_foresees_looming`.)
- Predicted trajectories intruding into the forward corridor are flagged *before* the object
  arrives. (See `tests/test_ttc.py`.)

## What this is not

- Not a SOTA deblurring method; the fusion stage is deliberately lightweight.
- Not validated on a public benchmark in this repository; the demo is synthetic and seeded.
- Not a flight controller; it emits risk, not actuator commands.

Honesty about the boundary is part of the design. The next three breakthroughs extend it.
