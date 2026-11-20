#!/usr/bin/env python3
"""
Shared helpers for the hockey highlight scripts.

  - device selection (CUDA / Apple MPS / CPU)
  - BoT-SORT tracker config with appearance re-ID turned on
  - geometry + HSV colour histograms (same mask for reference and candidates)
  - jersey-number reading (PARSeq, falling back to EasyOCR) with per-track voting
  - box smoothing (One-Euro filter) and gap filling for the offline renderer
  - overlay drawing and a robust FFmpeg writer
"""

import math
import os
import platform
import subprocess
import tempfile
from collections import defaultdict

import cv2
import numpy as np

try:
    import torch
except Exception:  # torch is optional for the pure-geometry helpers
    torch = None


# ============================================================
# DEVICE
# ============================================================
def pick_device():
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ============================================================
# TRACKER CONFIG
# ============================================================
def make_tracker_yaml(kind="botsort", with_reid=True, track_buffer=90, reid_model="auto"):
    """
    Write a tracker YAML based on the one shipped with ultralytics, with
    hockey-friendly settings:
      - BoT-SORT with appearance re-ID (model="auto" reuses YOLO's own features,
        so there is no extra model to download)
      - longer track_buffer so a player hidden behind others for ~3s keeps the same ID
      - camera-motion compensation stays on (handles panning/zooming cameras)
    Returns the path to the YAML file.
    """
    import ultralytics
    import yaml

    base_dir = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers")
    src = os.path.join(base_dir, f"{kind}.yaml")
    if not os.path.exists(src):
        print(f"[WARN] {kind}.yaml not found, falling back to bytetrack.")
        return os.path.join(base_dir, "bytetrack.yaml")

    with open(src, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["track_buffer"] = int(track_buffer)
    if kind == "botsort":
        cfg["with_reid"] = bool(with_reid)
        cfg["model"] = reid_model

    fd, path = tempfile.mkstemp(prefix=f"hockey_{kind}_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def run_track(model, frame, tracker_yaml, conf, imgsz, device):
    """YOLO + tracker on one frame. Returns (boxes_xyxy[N,4] float32, ids[N] int32)."""
    res = model.track(
        frame, persist=True, tracker=tracker_yaml, conf=conf, imgsz=imgsz,
        classes=[0], device=device, verbose=False,
    )
    b = res[0].boxes
    if b is None or b.id is None:
        return np.empty((0, 4), np.float32), np.empty((0,), np.int32)
    return b.xyxy.cpu().numpy().astype(np.float32), b.id.int().cpu().numpy().astype(np.int32)


# ============================================================
# VIDEO PROPS
# ============================================================
def detect_video_props(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0:
        fps = 30.0
    return w, h, fps


def autotune(w, h, fps, base_conf):
    is_4k = w * h >= int(3840 * 2160 * 0.85)
    return {
        "imgsz": 1024 if is_4k else 960,
        "conf": max(base_conf, 0.20) if is_4k else base_conf,
        "ocr_every": 18 if fps >= 55 else 10,
    }


# ============================================================
# GEOMETRY
# ============================================================
def clamp_box(box, W, H):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(W - 1.0, x1)); y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(0.0, min(W - 1.0, x2)); y2 = max(0.0, min(H - 1.0, y2))
    if x2 <= x1:
        x2 = min(W - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(H - 1.0, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_h(box):
    return float(box[3] - box[1])


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def torso_crop(frame, box, top=0.12, bottom=0.60, side=0.18):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    cy1, cy2 = int(y1 + top * h), int(y1 + bottom * h)
    cx1, cx2 = int(x1 + side * w), int(x2 - side * w)
    cy1 = max(0, cy1); cy2 = min(frame.shape[0], cy2)
    cx1 = max(0, cx1); cx2 = min(frame.shape[1], cx2)
    if cy2 <= cy1 or cx2 <= cx1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    return crop if crop.size else None


# ============================================================
# COLOUR HISTOGRAMS
# The old code built the reference histogram with a circular mask but the
# candidate histograms without one, so a candidate never matched as well as it
# should. Both now go through the same function.
# ============================================================
def hsv_hist(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), max(1, int(min(h, w) * 0.4)), 255, -1)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [90, 128], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def get_color_hist(frame, box):
    return hsv_hist(torso_crop(frame, box))


def compare_hist(crop_bgr, ref_hist):
    if ref_hist is None:
        return 0.0
    h = hsv_hist(crop_bgr)
    if h is None:
        return 0.0
    return float(cv2.compareHist(ref_hist, h, cv2.HISTCMP_CORREL))


# ============================================================
# BOX SMOOTHING
# ============================================================
class OneEuro:
    """One-Euro filter (Casiez et al.): smooths jitter when still, follows fast moves."""

    def __init__(self, freq, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = float(freq), min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x.astype(np.float32)
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) * self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat.astype(np.float32)


def smooth_series_zero_phase(boxes, freq, min_cutoff=1.2, beta=0.02):
    """Offline smoothing: One-Euro forward + backward, averaged, so there is no lag."""
    if len(boxes) < 3:
        return [np.asarray(b, np.float32) for b in boxes]
    fwd_f, bwd_f = OneEuro(freq, min_cutoff, beta), OneEuro(freq, min_cutoff, beta)
    fwd = [fwd_f(b) for b in boxes]
    bwd = [bwd_f(b) for b in reversed(boxes)][::-1]
    return [((f + b) / 2.0).astype(np.float32) for f, b in zip(fwd, bwd)]


def fill_and_smooth_tracks(tracking_db, fps, max_gap_frames=12, smooth=True):
    """
    tracking_db: {frame_idx: [ {slot, box, label, color, ...}, ... ]}
    Returns a new db where, per slot:
      - gaps of <= max_gap_frames between two tracked frames are filled by linear interpolation
        (stops the overlay blinking off for a frame or two)
      - each continuous run of boxes is smoothed with a zero-lag One-Euro filter
    """
    per_slot = defaultdict(dict)
    meta = {}
    for f, entries in tracking_db.items():
        for e in entries:
            per_slot[e["slot"]][int(f)] = np.asarray(e["box"], np.float32)
            meta[e["slot"]] = {k: v for k, v in e.items() if k not in ("box", "selected")}

    out = defaultdict(list)
    for slot, series in per_slot.items():
        frames = sorted(series)
        # 1) fill short gaps
        filled = {}
        for i, f in enumerate(frames):
            filled[f] = series[f]
            if i + 1 < len(frames):
                nf = frames[i + 1]
                gap = nf - f - 1
                if 0 < gap <= max_gap_frames:
                    a, b = series[f], series[nf]
                    for k in range(1, gap + 1):
                        t = k / (gap + 1)
                        filled[f + k] = (1 - t) * a + t * b
        # 2) smooth each continuous run
        fr = sorted(filled)
        runs, cur = [], [fr[0]]
        for f in fr[1:]:
            if f == cur[-1] + 1:
                cur.append(f)
            else:
                runs.append(cur); cur = [f]
        runs.append(cur)
        for run in runs:
            boxes = [filled[f] for f in run]
            if smooth:
                boxes = smooth_series_zero_phase(boxes, fps)
            for f, b in zip(run, boxes):
                e = dict(meta[slot]); e["box"] = [float(v) for v in b]
                out[f].append(e)
    return dict(out)


# ============================================================
# JERSEY NUMBER READING
# ============================================================
class JerseyReader:
    """
    Reads jersey numbers from a player box.

    Pipeline (after Koshkina & Elder, "A General Framework for Jersey Number
    Recognition in Sports Video", which reports 91.4% on hockey images):
      1. torso localisation with a YOLO pose model (shoulders -> hips); falls back to a fixed crop
      2. legibility gate: skip crops that are too small or too blurry to read
      3. scene-text recognition with PARSeq (torch.hub); EasyOCR is used if PARSeq can't load
    Only 1-2 digit strings are returned.

    PARSEQ_WEIGHTS can point at a fine-tuned hockey checkpoint (e.g. from
    github.com/mkoshkina/jersey-number-pipeline) for better accuracy.
    """

    def __init__(self, device="cpu", backend="parseq", pose_weights="yolo26n-pose.pt",
                 parseq_weights=None, min_crop_h=22, min_sharpness=25.0):
        self.device = device
        self.min_crop_h = min_crop_h
        self.min_sharpness = min_sharpness
        self.backend = None
        self.pose = None
        self.parseq = None
        self.easy = None

        if pose_weights:
            try:
                from ultralytics import YOLO
                self.pose = YOLO(pose_weights)
            except Exception as e:
                print(f"[JerseyReader] pose model unavailable ({e}); using fixed torso crop.")

        if backend == "parseq":
            try:
                self.parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
                if parseq_weights and os.path.exists(parseq_weights):
                    ckpt = torch.load(parseq_weights, map_location="cpu")
                    sd = ckpt.get("state_dict", ckpt)
                    missing, unexpected = self.parseq.load_state_dict(sd, strict=False)
                    print(f"[JerseyReader] loaded {parseq_weights} (missing={len(missing)}, unexpected={len(unexpected)})")
                # PARSeq is small and only runs every few frames; keep it off MPS to avoid unsupported-op errors
                self.parseq_device = "cuda" if device == "cuda" else "cpu"
                self.parseq = self.parseq.eval().to(self.parseq_device)
                self.img_size = tuple(self.parseq.hparams.img_size)  # (H, W) = (32, 128)
                self.backend = "parseq"
            except Exception as e:
                print(f"[JerseyReader] PARSeq unavailable ({e}); falling back to EasyOCR.")
                backend = "easyocr"

        if backend == "easyocr" and self.backend is None:
            try:
                import easyocr
                self.easy = easyocr.Reader(["en"], gpu=(device == "cuda"))
                self.backend = "easyocr"
            except Exception as e:
                print(f"[JerseyReader] EasyOCR unavailable ({e}); number reading disabled.")

        print(f"[JerseyReader] backend={self.backend}  pose={'on' if self.pose else 'off'}")

    @property
    def ok(self):
        return self.backend is not None

    # ---- torso localisation ----
    def _torso(self, frame, box):
        if self.pose is not None:
            x1, y1, x2, y2 = map(int, box)
            pad = int(0.08 * max(1, x2 - x1))
            px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
            px2, py2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            person = frame[py1:py2, px1:px2]
            if person.size:
                try:
                    r = self.pose.predict(person, imgsz=256, conf=0.25, device=self.device, verbose=False)[0]
                    if r.keypoints is not None and len(r.keypoints) > 0:
                        kp = r.keypoints.xy[0].cpu().numpy()
                        kc = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(kp))
                        idx = [5, 6, 11, 12]  # L/R shoulder, L/R hip (COCO)
                        if all(kc[i] > 0.3 for i in idx):
                            pts = kp[idx]
                            tx1, tx2 = pts[:, 0].min(), pts[:, 0].max()
                            ty1, ty2 = pts[:2, 1].min(), pts[2:, 1].max()
                            w, h = tx2 - tx1, ty2 - ty1
                            if w > 4 and h > 4:
                                tx1, tx2 = int(tx1 - 0.10 * w), int(tx2 + 0.10 * w)
                                ty1, ty2 = int(ty1 - 0.05 * h), int(ty2 + 0.05 * h)
                                tx1, ty1 = max(0, tx1), max(0, ty1)
                                tx2, ty2 = min(person.shape[1], tx2), min(person.shape[0], ty2)
                                c = person[ty1:ty2, tx1:tx2]
                                if c.size:
                                    return c
                except Exception:
                    pass
        return torso_crop(frame, box)

    def _legible(self, crop):
        if crop is None or crop.size == 0 or crop.shape[0] < self.min_crop_h:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() >= self.min_sharpness

    # ---- recognition ----
    def _read_parseq(self, crop):
        H, W = self.img_size
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        t = (t - 0.5) / 0.5
        with torch.inference_mode():
            probs = self.parseq(t.unsqueeze(0).to(self.parseq_device)).softmax(-1)
            labels, confs = self.parseq.tokenizer.decode(probs)
        txt = labels[0]
        conf = float(confs[0].prod().item()) if len(confs[0]) else 0.0
        return [(txt, conf)]

    def _read_easy(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        h = gray.shape[0]
        if h < 50:
            gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        elif h > 300:
            gray = cv2.resize(gray, (0, 0), fx=300.0 / h, fy=300.0 / h, interpolation=cv2.INTER_AREA)
        res = self.easy.readtext(gray, allowlist="0123456789", detail=1, paragraph=False)
        return [(t, float(c)) for _, t, c in res]

    def read(self, frame, box):
        """Returns (number_str, confidence) or None if nothing legible."""
        if not self.ok:
            return None
        crop = self._torso(frame, box)
        if not self._legible(crop):
            return None
        try:
            items = self._read_parseq(crop) if self.backend == "parseq" else self._read_easy(crop)
        except Exception:
            return None
        best = None
        for t, c in items:
            t = "".join(ch for ch in (t or "") if ch.isdigit())
            if 1 <= len(t) <= 2 and (best is None or c > best[1]):
                best = (t, c)
        return best


class NumberVoter:
    """
    Accumulates jersey-number reads per track ID and only commits to an answer
    once several frames agree. One lucky (or unlucky) frame no longer decides
    whether a candidate is "your" player.
    """

    def __init__(self, min_conf=0.35, min_votes=3, min_share=0.6, decay=0.97):
        self.min_conf, self.min_votes, self.min_share, self.decay = min_conf, min_votes, min_share, decay
        self.votes = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(int)

    def add(self, tid, number, conf):
        if number is None or conf < self.min_conf:
            return
        v = self.votes[tid]
        for k in list(v):
            v[k] *= self.decay
        v[number] += conf
        self.counts[tid] += 1

    def forget(self, tid):
        self.votes.pop(tid, None)
        self.counts.pop(tid, None)

    def verdict(self, tid):
        """Returns (number, share) once confident, else (None, 0)."""
        v = self.votes.get(tid)
        if not v or self.counts[tid] < self.min_votes:
            return None, 0.0
        total = sum(v.values())
        num, w = max(v.items(), key=lambda kv: kv[1])
        share = w / max(total, 1e-9)
        return (num, share) if share >= self.min_share else (None, share)

    def score(self, tid, hint):
        """+1 confident match, -1 confident mismatch, 0 unknown."""
        if not hint:
            return 0
        num, _ = self.verdict(tid)
        if num is None:
            return 0
        return 1 if num == hint else -1


# ============================================================
# OVERLAYS
# ============================================================
def draw_feet_marker(frame, box, color, thickness=4):
    x1, y1, x2, y2 = map(int, box)
    cx = (x1 + x2) // 2
    w, h = max(10, x2 - x1), max(10, y2 - y1)
    fy = max(0, min(frame.shape[0] - 1, int(y2 + 0.03 * h)))
    cv2.ellipse(frame, (cx, fy), (max(6, int(w * 0.42)), max(4, int(w * 0.15))),
                0, 0, 360, color, thickness, cv2.LINE_AA)


def draw_plate(frame, box, text, border_color, font_scale=1.0, thickness=3, pad_x=16, pad_y=12,
               bg_alpha=0.6, border=3, y_gap=14, text_color=(255, 255, 255)):
    x1, y1 = int(box[0]), int(box[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    rx1, ry2 = max(0, x1), max(0, y1 - y_gap)
    ry1 = max(0, ry2 - (th + pad_y * 2))
    rx2 = min(frame.shape[1] - 1, rx1 + tw + pad_x * 2)
    # blend only the plate region (much faster than copying a whole 4K frame)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size:
        roi[:] = (roi * (1 - bg_alpha)).astype(np.uint8)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), border_color, border)
    cv2.putText(frame, text, (rx1 + pad_x, ry2 - pad_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_bottom_banner(frame, text, font_scale=1.2, thickness=3, pad_x=26, pad_y=16,
                       y_margin=70, alpha=0.6, border=3):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bw, bh = tw + pad_x * 2, th + pad_y * 2
    x1 = max(0, (w - bw) // 2); x2 = min(w - 1, x1 + bw)
    y2 = max(0, h - y_margin); y1 = max(0, y2 - bh)
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        roi[:] = (roi * (1 - alpha)).astype(np.uint8)
    if border > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), border)
    cv2.putText(frame, text, (x1 + pad_x, y2 - pad_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ============================================================
# PREVIEW
# ============================================================
def preview_scale(W, H, max_w=1920, max_h=1080):
    return float(min(max_w / float(W), max_h / float(H), 1.0))


def resize_for_preview(frame, scale):
    if scale >= 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def candidates_near_roi(boxes, ids, roi_xywh, margin=60):
    rx, ry, rw, rh = roi_xywh
    rcx, rcy = rx + rw / 2.0, ry + rh / 2.0
    out = []
    for b, i in zip(boxes, ids):
        cx, cy = box_center(b)
        if rx - margin < cx < rx + rw + margin and ry - margin < cy < ry + rh + margin:
            out.append({"id": int(i), "box": np.asarray(b, np.float32), "dist": math.hypot(cx - rcx, cy - rcy)})
    out.sort(key=lambda d: d["dist"])
    return out


# ============================================================
# FFMPEG WRITER
# ============================================================
def default_video_codec_args(bitrate="8M", maxrate="10M", bufsize="16M"):
    if platform.system() == "Darwin":
        return ["-c:v", "hevc_videotoolbox", "-tag:v", "hvc1",
                "-b:v", bitrate, "-maxrate", maxrate, "-bufsize", bufsize]
    return ["-c:v", "libx265", "-tag:v", "hvc1", "-preset", "veryfast", "-crf", "20"]


class FFmpegVideoWriter:
    """
    Pipe raw frames into ffmpeg, copy audio from the source, write to a temp file
    and rename on success. ffmpeg's stderr goes to a log file (never fills a pipe).
    """

    def __init__(self, src_video, out_video, width, height, fps, codec_args=None, audio_bitrate="192k"):
        self.width = int(width) - (int(width) % 2)
        self.height = int(height) - (int(height) % 2)
        self.final_path = out_video
        root, ext = os.path.splitext(out_video)
        self.tmp_path = f"{root}.tmp{ext or '.mp4'}"
        self.log_path = f"{root}.ffmpeg.log"
        self.logf = open(self.log_path, "wb")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}",
            "-r", f"{float(fps):.03f}", "-i", "-",
            "-i", src_video, "-map", "0:v:0", "-map", "1:a:0?", "-shortest",
            *(codec_args or default_video_codec_args()),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", audio_bitrate, "-movflags", "+faststart",
            self.tmp_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self.logf, bufsize=0)
        self.broken = False

    def write(self, frame_bgr):
        if self.broken:
            return
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = frame_bgr[: self.height, : self.width]
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except (BrokenPipeError, OSError):
            self.broken = True

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        rc = self.proc.wait()
        self.logf.close()
        self.proc = None
        if rc != 0:
            try:
                with open(self.log_path, "rb") as f:
                    tail = "\n".join(f.read().decode("utf-8", "replace").strip().splitlines()[-30:])
            except Exception as e:
                tail = f"(could not read ffmpeg log: {e})"
            raise RuntimeError(f"ffmpeg failed (rc={rc}). Last log lines:\n{tail}")
        os.replace(self.tmp_path, self.final_path)
