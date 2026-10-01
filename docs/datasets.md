# Datasets

AeroNetra-Orion runs with **zero downloads** (the synthetic scene). For real work it targets both
**event** and **aerial** data, spanning domains on purpose. No single benchmark is privileged.

## Why these datasets

An event-driven, predictive UAV stack has two data needs that are usually served by disjoint
benchmarks:

1. **Event data** — to develop and validate fusion, event representations, and event-based
   tracking. Served by DSEC, MVSEC, VisEvent, DAVIS.
2. **Aerial imagery** — to develop and validate the detector in the UAV domain. Served by AU-AIR,
   UAVDT, SeaDronesSee, DroneCrowd, DOTA, CARPK, VisDrone.

Using only VisDrone would cover (2) narrowly and (1) not at all. Orion therefore treats event and
aerial datasets as first-class and complementary.

## Registry

| Key | Kind | Modality | Notes |
|---|---|---|---|
| `dsec` | event | stereo event + frame | high-res driving event data; fusion & depth |
| `mvsec` | event | event + frame + IMU + LiDAR | event optical flow, odometry |
| `visevent` | event | event + frame tracking | event-based single-object tracking |
| `davis240c` | event | event + frame | classic DAVIS recordings |
| `auair` | aerial | aerial RGB | low-altitude UAV traffic; detector training |
| `uavdt` | aerial | aerial RGB video | UAV detection & tracking benchmark |
| `seadronessee` | aerial | maritime UAV RGB | small-object, search-and-rescue |
| `dronecrowd` | aerial | UAV crowd counting | dense counting; predictive occupancy |
| `dota` | aerial | aerial oriented boxes | large oriented-object benchmark |
| `carpk` | aerial | aerial parking counting | clean counting benchmark |
| `visdrone` | aerial | aerial detection/tracking | widely used — one option among many |

```bash
orion list-datasets
```

## Expected layout

Datasets live under `data/<key>/` (configurable via `ORION_DATA_ROOT`). `require(key)` raises a
`FileNotFoundError` carrying the download URL if the folder is missing or empty.

```
data/
  dsec/
  auair/
  uavdt/
  visdrone/
  ...
```

## Parsers

Label parsing is exposed as pure functions so it is unit-testable without the data:

- `parse_visdrone_det(lines)` — `left,top,w,h,score,category,…` (skips category `0`)
- `parse_uavdt_gt(lines)` — `frame,id,x,y,w,h,out_of_view,occlusion,…`
- `parse_auair(json_obj)` — AU-AIR annotation JSON
- `load_events_npz(path)` — `x, y, t, p` (+ optional `width, height`)
