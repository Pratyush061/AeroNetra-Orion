"""End-to-end orchestration for AeroNetra-Orion.

The pipeline threads the whole idea together for every frame:

    sharp scene ──► DVS events ─┐
                                ├─► event-guided fusion ─► detector ─► tracker
    blurred frame ──────────────┘                                       │
                                                                        ▼
                        risk + occupancy ◄── TTC ◄── world-model forecast
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import viz
from .config import OrionConfig
from .detect import build_detector
from .events import EventStream, simulate_events
from .fusion import EventFrameFusion
from .predict import build_predictor
from .scene import SyntheticAerialScene
from .track import IoUKalmanTracker, iou
from .ttc import assess_tracks, min_ttc, occupancy_map


class OrionPipeline:
    """Event-driven predictive perception pipeline."""

    def __init__(self, cfg: OrionConfig):
        self.cfg = cfg
        self.fusion = EventFrameFusion(cfg.fusion)
        self.detector = build_detector(cfg.detect)
        self.tracker = IoUKalmanTracker(cfg.track)
        self.predictor = build_predictor(cfg.predict)

    def reset(self) -> None:
        self.fusion.reset()
        self.detector.reset()
        self.tracker.reset()

    @staticmethod
    def _events_for_frame(
        events: EventStream, t: float, width: int, height: int
    ) -> EventStream:
        sel = events.t == t
        return EventStream(events.x[sel], events.y[sel], events.t[sel], events.p[sel], width, height)

    def run(self, frames, out_dir=None, save_video=None, save_frames=None) -> dict:
        cfg = self.cfg
        out_dir = Path(out_dir or cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_video = cfg.save_video if save_video is None else save_video
        save_frames = cfg.save_frames if save_frames is None else save_frames

        frames = list(frames)
        if not frames:
            raise ValueError("no frames to process")

        H, W = frames[0].gray.shape[:2]
        sharps = [f.sharp for f in frames]
        times = [f.time for f in frames]
        events = simulate_events(
            sharps, times, threshold=cfg.events.threshold, eps=cfg.events.contrast_eps
        )

        writer = None
        if save_video:
            vpath = out_dir / "orion_demo.mp4"
            writer = cv2.VideoWriter(
                str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), cfg.scene.fps, (W, H)
            )
            if not writer.isOpened():
                writer = None  # codec unavailable in this environment

        frames_dir = out_dir / "frames"
        if save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)

        det_counts, trk_counts, ttcs, recalls, precisions = [], [], [], [], []
        alerts = 0
        warnings = 0
        t0 = time.perf_counter()

        for f in frames:
            ev = self._events_for_frame(events, f.time, W, H)
            fused = self.fusion.fuse(f.gray, ev)
            dets = self.detector.detect(fused.sharpened)
            tracks = self.tracker.update(dets)
            preds = [self.predictor.predict(tr) for tr in tracks]
            assessments = assess_tracks(tracks, preds, cfg.predict, (H, W))
            occ = occupancy_map(assessments, W, H) if assessments else None
            mttc = min_ttc(assessments)
            alerts += sum(1 for a in assessments if a.risk == "critical")
            warnings += sum(1 for a in assessments if a.risk == "warning")

            if f.boxes is not None and len(f.boxes):
                gt = f.boxes
                tp = sum(1 for b in gt if any(iou(b, d.box) >= 0.3 for d in dets))
                recalls.append(tp / len(gt))
                matched = sum(1 for d in dets if any(iou(d.box, b) >= 0.3 for b in gt))
                precisions.append(matched / len(dets) if dets else 0.0)

            erate = len(ev) * cfg.scene.fps
            annotated = viz.annotate(
                f.rgb, dets, tracks, assessments, occ, mttc, alerts, event_rate=erate
            )
            if writer is not None:
                writer.write(annotated)
            if save_frames:
                cv2.imwrite(str(frames_dir / f"frame_{f.index:04d}.png"), annotated)

            det_counts.append(len(dets))
            trk_counts.append(len(tracks))
            if np.isfinite(mttc):
                ttcs.append(float(mttc))

        if writer is not None:
            writer.release()
        elapsed = time.perf_counter() - t0

        summary = {
            "frames": len(frames),
            "fps_processed": round(len(frames) / elapsed, 2) if elapsed > 0 else None,
            "events_total": int(len(events)),
            "mean_detections": round(float(np.mean(det_counts)), 2) if det_counts else 0.0,
            "mean_tracks": round(float(np.mean(trk_counts)), 2) if trk_counts else 0.0,
            "critical_alerts": alerts,
            "warning_alerts": warnings,
            "min_ttc_s": round(min(ttcs), 3) if ttcs else None,
            "mean_recall": round(float(np.mean(recalls)), 3) if recalls else None,
            "mean_precision": round(float(np.mean(precisions)), 3) if precisions else None,
            "elapsed_s": round(elapsed, 3),
            "video": str(out_dir / "orion_demo.mp4") if (save_video and writer) else None,
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        return summary


def run_demo(cfg: OrionConfig | None = None, out_dir=None) -> dict:
    """Run the download-free synthetic demo end to end."""
    cfg = cfg or OrionConfig()
    scene = SyntheticAerialScene(cfg.scene)
    pipeline = OrionPipeline(cfg)
    return pipeline.run(scene, out_dir=out_dir)
