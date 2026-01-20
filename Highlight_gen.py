#!/usr/bin/env python3
"""
Hockey single-player tracker + feet marker + nameplate + bottom broadcast caption

PREVIEW/OUTPUT SPLIT:
  - Preview window shows DEBUG/HUD text (LOST/reacquiring, relock HUD, etc.)
  - Output video does NOT include debug text (clean overlays only)

PATCHES IN THIS VERSION
  ✅ Preview forced to 1080p (or smaller) while output stays native resolution
  ✅ ROI locking works on the downscaled preview (mouse coords mapped back to full-res)
  ✅ Candidate boxes + ROI rectangle render correctly in preview (scaled)
  ✅ Dark-profile denoise remains OFF (prevents big slowdowns)
  ✅ Robust FFmpeg writer finalization WITHOUT communicate()
  ✅ Nameplate bigger (more readable)
  ✅ Bottom caption draws true broadcast-style bar near bottom center

Keys (main playback):
  q = quit
  r = relock (enter relock mode, scrub if needed)

Keys (relock mode):
  Draw ROI with mouse (on the scaled preview)
  n = cycle candidates
  SPACE / ENTER = lock
  . = next frame (step forward 1 frame)
  , = previous frame (step back 1 frame)
  ] = jump forward 1 second
  [ = jump back 1 second
  p = toggle play/pause while in relock mode
  s = skip locking (return to main with NO LOCK; overlays hidden until you relock)
  q = cancel relock (returns to main; keeps existing lock if you had one)

Requirements:
  pip install ultralytics opencv-python easyocr torch torchvision
  brew install ffmpeg
"""

import os
import cv2
import math
import numpy as np
import easyocr
import torch
import ultralytics
import subprocess
from ultralytics import YOLO

# ============================================================
# USER CONFIG
# ============================================================
VIDEO_PATH  = "/Users/deckj/Documents/Game_Processor/Noah_powerplay_cloud_apo8_alq13.mp4"
OUTPUT_PATH = "Noah_point_powerplay_marked.mp4"

PLAYER_LABEL = "Decker"
SHOW_NAMEPLATE = True

BOTTOM_CAPTION_TEXT = "Noah #6 black at point on powerplay"
SHOW_BOTTOM_BANNER = True   # <-- turned ON so it actually writes

TARGET_NUMBER = "6"
USE_OCR = True

YOLO_WEIGHTS = "yolo11s.pt"   # or your best.pt
BASE_CONF = 0.18

# ============================================================
# PREVIEW SIZE CONTROL (output stays native)
# ============================================================
FORCE_PREVIEW_1080P = True
PREVIEW_MAX_W = 1920
PREVIEW_MAX_H = 1080

# ============================================================
# TRACKING / REACQUIRE TUNING
# ============================================================
REACQUIRE_THRESHOLD   = 80
OCR_CHECK_THRESHOLD   = 15
GRACE_PERIOD_FRAMES   = 20
SEARCH_RADIUS_BASE    = 200
SEARCH_RADIUS_GROWTH  = 15
SWITCH_CONFIRM_FRAMES = 20

# ============================================================
# VISUALS
# ============================================================
DRAW_FEET_MARKER = True
FEET_COLOR = (0, 0, 255)
FEET_THICKNESS = 4

# --- Nameplate (bigger) ---
NAMEPLATE_COLOR = (255, 255, 255)
NAMEPLATE_BG_ALPHA = 0.60
NAMEPLATE_FONT_SCALE = 1.00   # was ~0.8
NAMEPLATE_THICKNESS  = 3      # was 2
NAMEPLATE_PAD_X = 16          # was 10
NAMEPLATE_PAD_Y = 12          # was 8
NAMEPLATE_BORDER_THICKNESS = 3  # was 2
NAMEPLATE_Y_GAP = 14          # vertical gap above bbox

# --- Broadcast banner (bigger & true "lower third") ---
BANNER_ALPHA = 0.60
BANNER_Y_MARGIN = 70
BANNER_PAD_X = 26
BANNER_PAD_Y = 16
BANNER_BORDER = 3

BANNER_FONT_SCALE = 1.20
BANNER_THICKNESS  = 3

# ============================================================
# VIDEO ENCODE (HEVC VT + audio)
# ============================================================
HEVC_BITRATE   = "8M"
HEVC_MAXRATE   = "10M"
HEVC_BUFSIZE   = "16M"
AUDIO_BITRATE  = "192k"

# ============================================================
# OPTIONAL: AUTO RINK PRESET
# ============================================================
ENABLE_RINK_PRESET_AUTO = True
ENABLE_DENOISE = False  # keep OFF unless you really need it (slow at 4K)

# ============================================================
# ROI STATE (ROI is stored in PREVIEW coordinates)
# ============================================================
roi_selected = False
bbox_roi = (0, 0, 0, 0)  # x, y, w, h   (PREVIEW COORDS)

# preview = full * PREVIEW_SCALE
PREVIEW_SCALE = 1.0


def reset_roi():
    global roi_selected, bbox_roi
    roi_selected = False
    bbox_roi = (0, 0, 0, 0)


def select_roi(event, x, y, flags, param):
    """Mouse callback runs on the PREVIEW image, so x/y are PREVIEW coords."""
    global roi_selected, bbox_roi
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_roi = (x, y, 0, 0)
        roi_selected = False
    elif event == cv2.EVENT_LBUTTONUP:
        x1, y1 = bbox_roi[0], bbox_roi[1]
        w, h = x - x1, y - y1
        if w < 0:
            x1 += w
            w = abs(w)
        if h < 0:
            y1 += h
            h = abs(h)
        bbox_roi = (x1, y1, w, h)
        roi_selected = True


# ============================================================
# DEVICE (MPS/CPU)
# ============================================================
def pick_device():
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE = pick_device()


# ============================================================
# PREVIEW RESIZE + MAPPING HELPERS
# ============================================================
def compute_preview_scale(full_w, full_h):
    if not FORCE_PREVIEW_1080P:
        return 1.0
    return float(min(PREVIEW_MAX_W / float(full_w), PREVIEW_MAX_H / float(full_h), 1.0))


def resize_for_preview(frame_bgr):
    if PREVIEW_SCALE >= 1.0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    nw = int(round(w * PREVIEW_SCALE))
    nh = int(round(h * PREVIEW_SCALE))
    return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def roi_preview_to_full(roi_box_preview):
    x, y, w, h = roi_box_preview
    s = max(1e-9, PREVIEW_SCALE)
    return (x / s, y / s, w / s, h / s)


def box_full_to_preview_xyxy(box_xyxy_full):
    x1, y1, x2, y2 = box_xyxy_full
    s = PREVIEW_SCALE
    return (x1 * s, y1 * s, x2 * s, y2 * s)


# ============================================================
# VIDEO PROPS + AUTOTUNE
# ============================================================
def detect_video_props(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0:
        fps = 30.0
    return w, h, fps


def autotune(w, h, fps):
    px = w * h
    is_4k = px >= int(3840 * 2160 * 0.85)
    is_60 = fps >= 55.0
    is_30 = 25.0 <= fps < 55.0

    imgsz = 960 if not is_4k else 1024
    conf = BASE_CONF if not is_4k else max(BASE_CONF, 0.20)

    if is_60:
        ocr_every = 18
    elif is_30:
        ocr_every = 10
    else:
        ocr_every = 12

    return {"imgsz": imgsz, "conf": conf, "ocr_every": ocr_every}


# ============================================================
# BUILT-IN BYTETRACK YAML
# ============================================================
def get_builtin_bytetrack_yaml():
    return os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")


# ============================================================
# ROBUST FFmpeg writer (NO communicate(), stderr -> log file)
# ============================================================
class FFmpegVideoWriter:
    """
    Robust pipe writer:
      - write frames to ffmpeg stdin
      - ffmpeg stderr -> log file (no communicate())
      - encode to temp file then rename on success
    """
    def __init__(self, src_video, out_video, width, height, fps):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)

        self.final_path = out_video
        root, ext = os.path.splitext(out_video)
        self.tmp_path = f"{root}.tmp{ext if ext else '.mp4'}"
        self.log_path = f"{root}.ffmpeg.log"

        for p in (self.tmp_path, self.log_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        self.logf = open(self.log_path, "wb")

        cmd = [
            "ffmpeg", "-y",
            "-hide_banner",
            "-loglevel", "error",

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.03f}",
            "-i", "-",

            "-i", src_video,

            "-map", "0:v:0",
            "-map", "1:a?",
            "-shortest",

            "-c:v", "hevc_videotoolbox",
            "-tag:v", "hvc1",
            "-b:v", HEVC_BITRATE,
            "-maxrate", HEVC_MAXRATE,
            "-bufsize", HEVC_BUFSIZE,

            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,

            "-movflags", "+faststart",

            self.tmp_path
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.logf,
            bufsize=0
        )
        self._broken = False

    def write(self, frame_bgr):
        if self._broken:
            return
        try:
            if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            self._broken = True
        except Exception:
            self._broken = True

    def close(self):
        if self.proc is None:
            return

        try:
            if self.proc.stdin:
                self.proc.stdin.flush()
                self.proc.stdin.close()
        except Exception:
            pass

        rc = self.proc.wait()

        try:
            self.logf.flush()
            self.logf.close()
        except Exception:
            pass

        self.proc = None

        if rc != 0:
            tail = ""
            try:
                with open(self.log_path, "rb") as f:
                    data = f.read()
                text = data.decode("utf-8", errors="replace")
                tail = "\n".join(text.strip().splitlines()[-30:])
            except Exception as e:
                tail = f"(could not read ffmpeg log: {e})"
            raise RuntimeError(f"ffmpeg failed (rc={rc}). Last log lines:\n{tail}")

        os.replace(self.tmp_path, self.final_path)


# ============================================================
# RINK AUTO PRESET
# ============================================================
def estimate_rink_brightness(frame_bgr, sample_every=10):
    h, w = frame_bgr.shape[:2]
    y1 = int(h * 0.55)
    y2 = int(h * 0.90)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    roi = frame_bgr[y1:y2:sample_every, x1:x2:sample_every]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    return float(np.percentile(v, 60))


def select_rink_profile(brightness_p60):
    if brightness_p60 < 105:
        return "dark"
    elif brightness_p60 > 140:
        return "bright"
    else:
        return "neutral"


def rink_preset_auto(frame_bgr, profile):
    f = frame_bgr.astype(np.float32) / 255.0

    if profile == "dark":
        contrast, brightness = 1.18, 0.03
        warm_b, warm_r = 0.97, 1.03
        denoise, sharp = ENABLE_DENOISE, 0.12
    elif profile == "bright":
        contrast, brightness = 1.08, -0.02
        warm_b, warm_r = 0.96, 1.02
        denoise, sharp = False, 0.10
    else:
        contrast, brightness = 1.12, -0.01
        warm_b, warm_r = 0.96, 1.03
        denoise, sharp = False, 0.10

    f = (f - 0.5) * contrast + 0.5 + brightness

    b = f[:, :, 0] * warm_b
    g = f[:, :, 1] * 1.00
    r = f[:, :, 2] * warm_r
    f = np.stack([b, g, r], axis=2)

    f = np.clip(f, 0.0, 1.0)
    out = (f * 255.0).astype(np.uint8)

    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)

    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    out = cv2.addWeighted(out, 1.0 + sharp, blur, -sharp, 0)
    return out


# ============================================================
# Geometry helpers
# ============================================================
def clamp_box_xyxy(box, W, H):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(W - 1.0, x1))
    y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(0.0, min(W - 1.0, x2))
    y2 = max(0.0, min(H - 1.0, y2))
    if x2 <= x1:
        x2 = min(W - 1.0, x1 + 1.0)
    if y2 <= y1:
        y2 = min(H - 1.0, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_h(box):
    return float(box[3] - box[1])


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def torso_crop(frame, box_xyxy):
    x1, y1, x2, y2 = map(int, box_xyxy)
    h = y2 - y1
    w = x2 - x1
    cy1 = int(y1 + 0.15 * h)
    cy2 = int(y1 + 0.55 * h)
    cx1 = int(x1 + 0.25 * w)
    cx2 = int(x2 - 0.25 * w)
    cy1 = max(0, cy1); cy2 = min(frame.shape[0], cy2)
    cx1 = max(0, cx1); cx2 = min(frame.shape[1], cx2)
    if cy2 <= cy1 or cx2 <= cx1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    return crop if crop is not None and crop.size else None


def get_color_hist(frame, box_xyxy):
    crop = torso_crop(frame, box_xyxy)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def compare_color(crop_bgr, ref_hist):
    if crop_bgr is None or ref_hist is None:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return float(cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL))


# ============================================================
# Overlay helpers
# ============================================================
def draw_feet_marker(frame, box_xyxy):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cx = int((x1 + x2) / 2)
    w = max(10, x2 - x1)
    h = max(10, y2 - y1)

    feet_y = int(y2 + 0.03 * h)
    feet_y = max(0, min(frame.shape[0] - 1, feet_y))

    ax = int(w * 0.42)
    ay = int(w * 0.15)

    cv2.ellipse(frame, (cx, feet_y), (max(6, ax), max(4, ay)),
                0, 0, 360, FEET_COLOR, FEET_THICKNESS, cv2.LINE_AA)


def draw_nameplate(frame, box_xyxy, text):
    x1, y1, x2, y2 = map(int, box_xyxy)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = NAMEPLATE_FONT_SCALE
    thickness = NAMEPLATE_THICKNESS

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    rx1 = max(0, x1)
    ry2 = max(0, y1 - NAMEPLATE_Y_GAP)
    ry1 = max(0, ry2 - (th + NAMEPLATE_PAD_Y * 2))
    rx2 = min(frame.shape[1] - 1, rx1 + tw + NAMEPLATE_PAD_X * 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, NAMEPLATE_BG_ALPHA, frame, 1 - NAMEPLATE_BG_ALPHA, 0, frame)

    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), NAMEPLATE_BORDER_THICKNESS)

    tx = rx1 + NAMEPLATE_PAD_X
    ty = ry2 - NAMEPLATE_PAD_Y
    cv2.putText(frame, text, (tx, ty), font, font_scale, NAMEPLATE_COLOR, thickness, cv2.LINE_AA)


def draw_bottom_banner(frame, text):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = BANNER_FONT_SCALE
    thickness = BANNER_THICKNESS

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    box_w = tw + (BANNER_PAD_X * 2)
    box_h = th + (BANNER_PAD_Y * 2)

    x1 = int((w - box_w) / 2)
    x2 = int(x1 + box_w)

    y2 = int(h - BANNER_Y_MARGIN)
    y1 = int(y2 - box_h)

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, BANNER_ALPHA, frame, 1 - BANNER_ALPHA, 0, frame)

    if BANNER_BORDER > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), BANNER_BORDER)

    tx = x1 + BANNER_PAD_X
    ty = y2 - BANNER_PAD_Y
    cv2.putText(frame, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ============================================================
# YOLO+ByteTrack wrapper
# ============================================================
def run_track(model, frame, tracker_yaml, cfg):
    res = model.track(
        frame,
        persist=True,
        tracker=tracker_yaml,
        conf=cfg["conf"],
        imgsz=cfg["imgsz"],
        classes=[0],
        device=DEVICE,
        verbose=False
    )
    if res[0].boxes.id is None:
        return np.empty((0, 4), np.float32), np.empty((0,), np.int32)
    boxes = res[0].boxes.xyxy.cpu().numpy().astype(np.float32)
    ids = res[0].boxes.id.int().cpu().numpy().astype(np.int32)
    return boxes, ids


def candidates_near_roi_fullres(boxes_full, ids, roi_box_full_xywh):
    ux, uy, uw, uh = roi_box_full_xywh
    roi_c = (ux + uw / 2.0, uy + uh / 2.0)
    cand = []
    for box, tid in zip(boxes_full, ids):
        cx, cy = box_center(box)
        if (ux - 80) <= cx <= (ux + uw + 80) and (uy - 80) <= cy <= (uy + uh + 80):
            dist = ((cx - roi_c[0]) ** 2 + (cy - roi_c[1]) ** 2) ** 0.5
            cand.append({"id": int(tid), "box": box, "dist": dist})
    cand.sort(key=lambda d: d["dist"])
    return cand


def seek_cap_to_frame(cap, target_frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(target_frame_idx)))


def relock_mode(cap, current_frame_full, current_frame_idx, model, tracker_yaml, cfg, fps):
    global roi_selected, bbox_roi
    reset_roi()

    candidate_list = []
    selected_idx = 0
    frozen = False
    paused = True

    frame_full = current_frame_full
    frame_idx = int(current_frame_idx)

    print("\n⏸ RELOCK MODE")
    print("Draw ROI. n=cycle, SPACE=lock, p=play/pause, . , step, [ ] jump sec, s=skip lock, q=cancel\n")

    while True:
        frame_disp = resize_for_preview(frame_full.copy())
        temp = frame_disp.copy()

        hud = f"RELOCK: frame {frame_idx}  {'PAUSED' if paused else 'PLAY'}  (n cycle, SPACE lock, s skip, q cancel)"
        cv2.putText(temp, hud, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if bbox_roi[2] > 0 and bbox_roi[3] > 0:
            x, y, ww, hh = bbox_roi
            cv2.rectangle(temp, (x, y), (x + ww, y + hh), (255, 0, 0), 2)

        if roi_selected and not frozen:
            boxes_full, ids = run_track(model, frame_full, tracker_yaml, cfg)
            roi_full = roi_preview_to_full(bbox_roi)
            candidate_list = candidates_near_roi_fullres(boxes_full, ids, roi_full)
            selected_idx = 0
            frozen = True

        if candidate_list:
            curr = candidate_list[selected_idx]
            x1, y1, x2, y2 = curr["box"]
            px1, py1, px2, py2 = box_full_to_preview_xyxy((x1, y1, x2, y2))
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
            cv2.rectangle(temp, (px1, py1), (px2, py2), (0, 255, 0), 3)
            cv2.putText(temp, f"Sel ID: {curr['id']} (n=next)", (px1, max(20, py1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif roi_selected:
            cv2.putText(temp, "No candidates near ROI. Redraw ROI.", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Video", temp)
        k = cv2.waitKey(15) & 0xFF

        if k == ord("p"):
            paused = not paused

        if not roi_selected:
            frozen = False
            candidate_list = []

        if k == ord("n") and candidate_list:
            selected_idx = (selected_idx + 1) % len(candidate_list)

        if k in (32, 13) and candidate_list:
            curr = candidate_list[selected_idx]
            tid = int(curr["id"])
            lock_box = clamp_box_xyxy(curr["box"], frame_full.shape[1], frame_full.shape[0])
            ref_hist = get_color_hist(frame_full, lock_box)
            return ("lock", tid, lock_box, ref_hist, frame_full, frame_idx)

        if k == ord("s"):
            return ("skip", None, None, None, frame_full, frame_idx)

        if k == ord("q"):
            return ("cancel", None, None, None, current_frame_full, current_frame_idx)

        step = 0
        if k == ord("."):
            step = 1
        elif k == ord(","):
            step = -1
        elif k == ord("]"):
            step = int(round(fps))
        elif k == ord("["):
            step = -int(round(fps))

        if step != 0:
            frame_idx = max(0, frame_idx + step)
            seek_cap_to_frame(cap, frame_idx)
            ok, newf = cap.read()
            if ok:
                frame_full = newf
            reset_roi()
            candidate_list = []
            frozen = False
            paused = True

        if not paused:
            ok, newf = cap.read()
            if not ok:
                paused = True
            else:
                frame_full = newf
                frame_idx += 1


# ============================================================
# MAIN
# ============================================================
def main():
    global PREVIEW_SCALE

    print("YOLO device selected:", DEVICE)
    tracker_yaml = get_builtin_bytetrack_yaml()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: could not open video:", VIDEO_PATH)
        return

    W, H, FPS = detect_video_props(cap)
    cfg = autotune(W, H, FPS)

    PREVIEW_SCALE = compute_preview_scale(W, H)

    print(f"Video: {W}x{H} @ {FPS:.2f}fps")
    print("Autotune:", cfg)
    print("ByteTrack YAML:", tracker_yaml)
    print("YOLO weights:", YOLO_WEIGHTS)
    print(f"Preview scaling: {'ON' if FORCE_PREVIEW_1080P else 'OFF'} (scale={PREVIEW_SCALE:.3f})")

    print("Loading YOLO...")
    model = YOLO(YOLO_WEIGHTS)
    reader = easyocr.Reader(["en"], gpu=False) if USE_OCR else None

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Video", select_roi)

    ret, first = cap.read()
    if not ret:
        print("Error: could not read first frame")
        return

    rink_profile = "neutral"
    if ENABLE_RINK_PRESET_AUTO:
        bscore = estimate_rink_brightness(first)
        rink_profile = select_rink_profile(bscore)
        print(f"[INFO] Rink profile: {rink_profile} (p60={bscore:.1f})  denoise={'ON' if ENABLE_DENOISE else 'OFF'}")

    first_proc = rink_preset_auto(first.copy(), rink_profile) if ENABLE_RINK_PRESET_AUTO else first

    action, tid, lock_box, ref_hist, _, _ = relock_mode(cap, first_proc, 0, model, tracker_yaml, cfg, FPS)

    TARGET_ID = None
    if action == "lock":
        TARGET_ID = tid
        print(f"✅ Locked TARGET_ID={TARGET_ID}")
    elif action == "skip":
        print("⚠️ Skipping lock: running with NO LOCK (no overlays) until you press 'r' to relock.")
    else:
        cap.release()
        cv2.destroyAllWindows()
        return

    seek_cap_to_frame(cap, 0)

    writer = None
    try:
        writer = FFmpegVideoWriter(VIDEO_PATH, OUTPUT_PATH, W, H, FPS)

        last_box = lock_box.copy() if lock_box is not None else None
        last_center = box_center(last_box) if last_box is not None else None
        last_h = box_h(last_box) if last_box is not None else None
        velocity = (0.0, 0.0)
        frames_since_lost = 0
        pending_id = None
        pending_count = 0
        frame_idx = -1

        while True:
            ret, frame_in = cap.read()
            if not ret:
                break
            frame_idx += 1

            frame_proc = frame_in
            if ENABLE_RINK_PRESET_AUTO:
                if frame_idx % int(max(1, round(FPS) * 4)) == 0:
                    bscore = estimate_rink_brightness(frame_proc)
                    new_profile = select_rink_profile(bscore)
                    if new_profile != rink_profile:
                        rink_profile = new_profile
                        print(f"[INFO] Rink profile -> {rink_profile} (p60={bscore:.1f})  denoise={'ON' if ENABLE_DENOISE else 'OFF'}")
                frame_proc = rink_preset_auto(frame_proc, rink_profile)

            frame_out = frame_proc.copy()
            frame_prev = frame_out.copy()

            # Banner is "clean overlay" => goes to output and preview
            if SHOW_BOTTOM_BANNER and BOTTOM_CAPTION_TEXT:
                draw_bottom_banner(frame_out, BOTTOM_CAPTION_TEXT)
                draw_bottom_banner(frame_prev, BOTTOM_CAPTION_TEXT)

            if TARGET_ID is None:
                cv2.putText(frame_prev, "NO LOCK: press 'r' to lock", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Video", resize_for_preview(frame_prev))
                writer.write(frame_out)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    action, tid, lock_box, ref_hist, _, _ = relock_mode(
                        cap, frame_proc, frame_idx, model, tracker_yaml, cfg, FPS
                    )
                    if action == "lock":
                        TARGET_ID = tid
                        last_box = lock_box.copy()
                        last_center = box_center(last_box)
                        last_h = box_h(last_box)
                        velocity = (0.0, 0.0)
                        frames_since_lost = 0
                        pending_id = None
                        pending_count = 0
                        print(f"✅ Re-locked TARGET_ID={TARGET_ID}")
                    elif action == "skip":
                        TARGET_ID = None
                continue

            boxes, ids = run_track(model, frame_proc, tracker_yaml, cfg)

            target_found = False
            target_box = None

            if len(ids) > 0 and TARGET_ID in ids:
                idx = int(np.where(ids == TARGET_ID)[0][0])
                target_box = clamp_box_xyxy(boxes[idx], W, H)
                target_found = True

            if not target_found and last_center is not None and len(ids) > 0:
                frames_since_lost += 1

                drag = 0.95 ** frames_since_lost
                pred_x = float(last_center[0] + (velocity[0] * drag * frames_since_lost))
                pred_y = float(last_center[1] + (velocity[1] * drag * frames_since_lost))
                search_radius = float(SEARCH_RADIUS_BASE + frames_since_lost * SEARCH_RADIUS_GROWTH)

                best_score = -1e9
                best_id = None
                best_box = None

                for box, tid_ in zip(boxes, ids):
                    box = clamp_box_xyxy(box, W, H)
                    cx, cy = box_center(box)
                    dist = math.hypot(cx - pred_x, cy - pred_y)
                    if dist > search_radius:
                        continue

                    score = 0.0
                    score += 50.0 * (1.0 - dist / search_radius)

                    bh = box_h(box)
                    if last_h and last_h > 0 and abs(bh - last_h) / last_h < 0.25:
                        score += 25.0

                    if last_box is not None:
                        score += 60.0 * iou(last_box, box)

                    crop = torso_crop(frame_proc, box)
                    if crop is not None and ref_hist is not None:
                        score += 40.0 * compare_color(crop, ref_hist)

                    if USE_OCR and reader is not None and score > OCR_CHECK_THRESHOLD:
                        if (frame_idx % cfg["ocr_every"]) == 0 and crop is not None:
                            try:
                                ocr_res = reader.readtext(crop, allowlist="0123456789")
                                for _, text, prob in ocr_res:
                                    t = (text or "").strip()
                                    if TARGET_NUMBER and (TARGET_NUMBER in t) and prob >= 0.30:
                                        score += 120.0
                                    elif TARGET_NUMBER and prob >= 0.85 and t != TARGET_NUMBER:
                                        score -= 120.0
                            except Exception:
                                pass

                    if score > best_score:
                        best_score = score
                        best_id = int(tid_)
                        best_box = box

                if best_id is not None and best_score >= REACQUIRE_THRESHOLD:
                    if pending_id == best_id:
                        pending_count += 1
                    else:
                        pending_id = best_id
                        pending_count = 1

                    if pending_count >= SWITCH_CONFIRM_FRAMES:
                        TARGET_ID = best_id
                        target_box = best_box
                        target_found = True
                        frames_since_lost = 0
                        pending_id = None
                        pending_count = 0

            if target_found and target_box is not None:
                cx, cy = box_center(target_box)
                if last_center is not None:
                    vx = cx - last_center[0]
                    vy = cy - last_center[1]
                    velocity = (0.6 * velocity[0] + 0.4 * vx, 0.6 * velocity[1] + 0.4 * vy)
                last_center = (cx, cy)
                last_h = box_h(target_box)
                last_box = target_box.copy()
                frames_since_lost = 0
            else:
                frames_since_lost += 1

            if target_found and target_box is not None:
                if DRAW_FEET_MARKER:
                    draw_feet_marker(frame_out, target_box)
                    draw_feet_marker(frame_prev, target_box)
                if SHOW_NAMEPLATE and PLAYER_LABEL:
                    draw_nameplate(frame_out, target_box, PLAYER_LABEL)
                    draw_nameplate(frame_prev, target_box, PLAYER_LABEL)
            else:
                if frames_since_lost <= GRACE_PERIOD_FRAMES:
                    cv2.putText(frame_prev, "LOST: reacquiring...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame_prev, "LOST: press 'r' to relock", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Video", resize_for_preview(frame_prev))
            writer.write(frame_out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                action, tid, lock_box, ref_hist, _, _ = relock_mode(
                    cap, frame_proc, frame_idx, model, tracker_yaml, cfg, FPS
                )
                if action == "lock":
                    TARGET_ID = tid
                    last_box = lock_box.copy()
                    last_center = box_center(last_box)
                    last_h = box_h(last_box)
                    velocity = (0.0, 0.0)
                    frames_since_lost = 0
                    pending_id = None
                    pending_count = 0
                    print(f"✅ Re-locked TARGET_ID={TARGET_ID}")
                elif action == "skip":
                    TARGET_ID = None
                    print("⚠️ Switched to NO LOCK (no overlays) — press 'r' to relock anytime.")
                else:
                    print("Relock cancelled.")

    except Exception as e:
        print("\n[ERROR] Writer/processing failed:")
        print(str(e))
        print("If ffmpeg failed, check the generated .ffmpeg.log next to your output name.")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.close()
        except Exception as e:
            print("\n[ERROR] Writer failed to finalize output:")
            print(str(e))
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("Done!")
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()