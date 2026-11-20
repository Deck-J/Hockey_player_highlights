#!/usr/bin/env python3
"""
Hockey single-player tracker + feet marker + nameplate + bottom broadcast caption

PREVIEW/OUTPUT SPLIT:
  - Preview window shows DEBUG/HUD text (LOST/reacquiring, relock HUD, etc.)
  - Output video does NOT include debug text (clean overlays only)

WHAT'S NEW (modernize-tracking)
  ✅ YOLO26 detector by default (NMS-free, faster on CPU/MPS)
  ✅ BoT-SORT with appearance re-ID instead of plain ByteTrack -> far fewer ID swaps
  ✅ Longer track memory (track_buffer=90) so short occlusions keep the same ID
  ✅ Jersey numbers read with PARSeq + pose-based torso crop, and VOTED over several
     frames (EasyOCR fallback). The target is also checked while tracked: if its
     number is confidently wrong, the lock is dropped and reacquired.
  ✅ Overlay smoothed with a One-Euro filter (no jitter, no lag on fast skating)
  ✅ Fixed: lost-frame counter was incremented twice per frame
  ✅ Fixed: colour histogram for candidates now uses the same mask as the reference
  For the most accurate (but slower) option, see sam_track.py.

Keys (main playback):
  q = quit
  r = relock (enter relock mode, scrub if needed)

Keys (relock mode):
  Draw ROI with mouse (on the scaled preview)
  n = cycle candidates
  SPACE / ENTER = lock
  . = next frame        , = previous frame
  ] = +1 second         [ = -1 second
  p = toggle play/pause while in relock mode
  s = skip locking (return with NO LOCK; overlays hidden until you relock)
  q = cancel relock (keeps existing lock if you had one)

Requirements:
  pip install -r requirements.txt
  brew install ffmpeg
"""

import math

import cv2
import numpy as np
from ultralytics import YOLO

from hockey_common import (
    FFmpegVideoWriter, JerseyReader, NumberVoter, OneEuro, autotune, box_center, box_h,
    candidates_near_roi, clamp_box, compare_hist, default_video_codec_args, detect_video_props,
    draw_bottom_banner, draw_feet_marker, draw_plate, get_color_hist, iou, make_tracker_yaml,
    pick_device, preview_scale, resize_for_preview, run_track, torso_crop,
)

# ============================================================
# USER CONFIG
# ============================================================
VIDEO_PATH  = "/Users/deckj/Documents/Game_Processor/Noah_powerplay_cloud_apo8_alq13.mp4"
OUTPUT_PATH = "Noah_point_powerplay_marked.mp4"

PLAYER_LABEL = "Decker"
SHOW_NAMEPLATE = True

BOTTOM_CAPTION_TEXT = "Noah #6 black at point on powerplay"
SHOW_BOTTOM_BANNER = True

TARGET_NUMBER = "6"          # jersey number used to confirm / reacquire ("" to disable)
USE_OCR = True
OCR_BACKEND = "parseq"       # "parseq" (better) or "easyocr"
PARSEQ_WEIGHTS = None        # optional fine-tuned hockey checkpoint path
POSE_WEIGHTS = "yolo26n-pose.pt"  # torso finder for number reading; None to disable

YOLO_WEIGHTS = "yolo26s.pt"  # or yolo26m.pt (more accurate), or your fine-tuned best.pt
BASE_CONF = 0.18

TRACKER = "botsort"          # "botsort" (with re-ID) or "bytetrack"
TRACKER_REID = True
TRACK_BUFFER = 90            # frames a hidden player keeps their ID

SMOOTH_OVERLAY = True

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
GRACE_PERIOD_FRAMES   = 20
SEARCH_RADIUS_BASE    = 200
SEARCH_RADIUS_GROWTH  = 15
SWITCH_CONFIRM_FRAMES = 20
NUMBER_BONUS          = 120.0  # score for a voted jersey-number match (penalty for mismatch)
REJECT_SECONDS        = 2.0    # after dropping a wrong-number ID, ignore it this long

# ============================================================
# VISUALS
# ============================================================
DRAW_FEET_MARKER = True
FEET_COLOR = (0, 0, 255)
FEET_THICKNESS = 4

NAMEPLATE_FONT_SCALE = 1.00
NAMEPLATE_THICKNESS  = 3
NAMEPLATE_PAD_X = 16
NAMEPLATE_PAD_Y = 12
NAMEPLATE_BG_ALPHA = 0.60
NAMEPLATE_BORDER_THICKNESS = 3
NAMEPLATE_Y_GAP = 14

BANNER_ALPHA = 0.60
BANNER_Y_MARGIN = 70
BANNER_PAD_X = 26
BANNER_PAD_Y = 16
BANNER_BORDER = 3
BANNER_FONT_SCALE = 1.20
BANNER_THICKNESS  = 3

# ============================================================
# VIDEO ENCODE (HEVC VideoToolbox on Mac, libx265 elsewhere) + audio
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

DEVICE = pick_device()

# ============================================================
# ROI STATE (ROI is stored in PREVIEW coordinates)
# ============================================================
roi_selected = False
bbox_roi = (0, 0, 0, 0)
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
            x1 += w; w = -w
        if h < 0:
            y1 += h; h = -h
        bbox_roi = (x1, y1, w, h)
        roi_selected = True


def preview(frame):
    return resize_for_preview(frame, PREVIEW_SCALE)


def roi_preview_to_full(roi):
    s = max(1e-9, PREVIEW_SCALE)
    return tuple(v / s for v in roi)


# ============================================================
# RINK AUTO PRESET
# ============================================================
def estimate_rink_brightness(frame_bgr, sample_every=10):
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[int(h * 0.55):int(h * 0.90):sample_every, int(w * 0.10):int(w * 0.90):sample_every]
    if roi.size == 0:
        return 0.0
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    return float(np.percentile(v, 60))


def select_rink_profile(p60):
    if p60 < 105:
        return "dark"
    if p60 > 140:
        return "bright"
    return "neutral"


_LUT_CACHE = {}


def rink_preset_auto(frame_bgr, profile):
    """Contrast/warmth via per-channel lookup tables (same look as before, much faster at 4K)."""
    if profile == "dark":
        contrast, brightness, warm_b, warm_r, denoise, sharp = 1.18, 0.03, 0.97, 1.03, ENABLE_DENOISE, 0.12
    elif profile == "bright":
        contrast, brightness, warm_b, warm_r, denoise, sharp = 1.08, -0.02, 0.96, 1.02, False, 0.10
    else:
        contrast, brightness, warm_b, warm_r, denoise, sharp = 1.12, -0.01, 0.96, 1.03, False, 0.10

    if profile not in _LUT_CACHE:
        x = np.arange(256, dtype=np.float32) / 255.0
        base = (x - 0.5) * contrast + 0.5 + brightness
        luts = [np.clip(base * m, 0, 1) * 255.0 for m in (warm_b, 1.0, warm_r)]
        _LUT_CACHE[profile] = [l.astype(np.uint8) for l in luts]
    lb, lg, lr = _LUT_CACHE[profile]
    b, g, r = cv2.split(frame_bgr)
    out = cv2.merge([cv2.LUT(b, lb), cv2.LUT(g, lg), cv2.LUT(r, lr)])

    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.0 + sharp, blur, -sharp, 0)


# ============================================================
# OVERLAYS
# ============================================================
def draw_target(frame, box):
    if DRAW_FEET_MARKER:
        draw_feet_marker(frame, box, FEET_COLOR, FEET_THICKNESS)
    if SHOW_NAMEPLATE and PLAYER_LABEL:
        draw_plate(frame, box, PLAYER_LABEL, (255, 255, 255), NAMEPLATE_FONT_SCALE, NAMEPLATE_THICKNESS,
                   NAMEPLATE_PAD_X, NAMEPLATE_PAD_Y, NAMEPLATE_BG_ALPHA, NAMEPLATE_BORDER_THICKNESS,
                   NAMEPLATE_Y_GAP)


def draw_banner(frame):
    draw_bottom_banner(frame, BOTTOM_CAPTION_TEXT, BANNER_FONT_SCALE, BANNER_THICKNESS, BANNER_PAD_X,
                       BANNER_PAD_Y, BANNER_Y_MARGIN, BANNER_ALPHA, BANNER_BORDER)


# ============================================================
# RELOCK MODE
# ============================================================
def relock_mode(cap, current_frame_full, current_frame_idx, track_fn, fps, rink_profile):
    reset_roi()
    candidate_list, selected_idx, frozen, paused = [], 0, False, True
    frame_full, frame_idx = current_frame_full, int(current_frame_idx)

    print("\n⏸ RELOCK MODE")
    print("Draw ROI. n=cycle, SPACE=lock, p=play/pause, . , step, [ ] jump sec, s=skip lock, q=cancel\n")

    def load(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx)))
        ok, f = cap.read()
        if ok and ENABLE_RINK_PRESET_AUTO:
            f = rink_preset_auto(f, rink_profile)
        return ok, f

    while True:
        temp = preview(frame_full).copy()
        hud = f"RELOCK: frame {frame_idx}  {'PAUSED' if paused else 'PLAY'}  (n cycle, SPACE lock, s skip, q cancel)"
        cv2.putText(temp, hud, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if bbox_roi[2] > 0 and bbox_roi[3] > 0:
            x, y, ww, hh = bbox_roi
            cv2.rectangle(temp, (x, y), (x + ww, y + hh), (255, 0, 0), 2)

        if roi_selected and not frozen:
            boxes, ids = track_fn(frame_full)
            candidate_list = candidates_near_roi(boxes, ids, roi_preview_to_full(bbox_roi), margin=80)
            selected_idx, frozen = 0, True

        if candidate_list:
            c = candidate_list[selected_idx]
            px1, py1, px2, py2 = (int(v * PREVIEW_SCALE) for v in c["box"])
            cv2.rectangle(temp, (px1, py1), (px2, py2), (0, 255, 0), 3)
            cv2.putText(temp, f"Sel ID: {c['id']} (n=next)", (px1, max(20, py1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif roi_selected:
            cv2.putText(temp, "No candidates near ROI. Redraw ROI.", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Video", temp)
        k = cv2.waitKey(15) & 0xFF

        if k == ord("p"):
            paused = not paused
        if not roi_selected:
            frozen, candidate_list = False, []
        if k == ord("n") and candidate_list:
            selected_idx = (selected_idx + 1) % len(candidate_list)
        if k in (32, 13) and candidate_list:
            c = candidate_list[selected_idx]
            lock_box = clamp_box(c["box"], frame_full.shape[1], frame_full.shape[0])
            return ("lock", int(c["id"]), lock_box, get_color_hist(frame_full, lock_box), frame_idx)
        if k == ord("s"):
            return ("skip", None, None, None, frame_idx)
        if k == ord("q"):
            return ("cancel", None, None, None, current_frame_idx)

        step = {ord("."): 1, ord(","): -1, ord("]"): int(round(fps)), ord("["): -int(round(fps))}.get(k, 0)
        if step:
            frame_idx = max(0, frame_idx + step)
            ok, f = load(frame_idx)
            if ok:
                frame_full = f
            reset_roi()
            candidate_list, frozen, paused = [], False, True

        if not paused:
            ok, f = cap.read()
            if not ok:
                paused = True
            else:
                frame_full = rink_preset_auto(f, rink_profile) if ENABLE_RINK_PRESET_AUTO else f
                frame_idx += 1


# ============================================================
# MAIN
# ============================================================
def main():
    global PREVIEW_SCALE

    print("Device:", DEVICE)
    tracker_yaml = make_tracker_yaml(TRACKER, with_reid=TRACKER_REID, track_buffer=TRACK_BUFFER)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: could not open video:", VIDEO_PATH)
        return

    W, H, FPS = detect_video_props(cap)
    cfg = autotune(W, H, FPS, BASE_CONF)
    PREVIEW_SCALE = preview_scale(W, H, PREVIEW_MAX_W, PREVIEW_MAX_H) if FORCE_PREVIEW_1080P else 1.0

    print(f"Video: {W}x{H} @ {FPS:.2f}fps   Autotune: {cfg}")
    print(f"Tracker: {TRACKER} (re-ID={'on' if TRACKER_REID else 'off'})   Weights: {YOLO_WEIGHTS}")

    model = YOLO(YOLO_WEIGHTS)
    reader = JerseyReader(DEVICE, OCR_BACKEND, POSE_WEIGHTS, PARSEQ_WEIGHTS) if (USE_OCR and TARGET_NUMBER) else None
    voter = NumberVoter()

    def track_fn(frame):
        return run_track(model, frame, tracker_yaml, cfg["conf"], cfg["imgsz"], DEVICE)

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
        print(f"[INFO] Rink profile: {rink_profile} (p60={bscore:.1f})")
        first = rink_preset_auto(first, rink_profile)

    action, tid, lock_box, ref_hist, _ = relock_mode(cap, first, 0, track_fn, FPS, rink_profile)
    if action == "cancel":
        cap.release(); cv2.destroyAllWindows()
        return

    TARGET_ID = tid if action == "lock" else None
    print(f"✅ Locked TARGET_ID={TARGET_ID}" if TARGET_ID is not None else "⚠️ NO LOCK — press 'r' to lock.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    smoother = OneEuro(FPS)
    state = {}

    def reset_state(box):
        state.update(last_box=None if box is None else box.copy(),
                     last_center=None if box is None else box_center(box),
                     last_h=None if box is None else box_h(box),
                     velocity=(0.0, 0.0), lost=0, pending_id=None, pending_count=0)
        smoother.reset()

    reset_state(lock_box)
    rejected = {}  # tid -> frame index it was rejected at

    writer = None
    try:
        writer = FFmpegVideoWriter(VIDEO_PATH, OUTPUT_PATH, W, H, FPS,
                                   default_video_codec_args(HEVC_BITRATE, HEVC_MAXRATE, HEVC_BUFSIZE),
                                   AUDIO_BITRATE)
        frame_idx = -1

        while True:
            ret, frame_in = cap.read()
            if not ret:
                break
            frame_idx += 1

            frame_proc = frame_in
            if ENABLE_RINK_PRESET_AUTO:
                if frame_idx % int(max(1, round(FPS) * 4)) == 0:
                    new_profile = select_rink_profile(estimate_rink_brightness(frame_proc))
                    if new_profile != rink_profile:
                        rink_profile = new_profile
                        print(f"[INFO] Rink profile -> {rink_profile}")
                frame_proc = rink_preset_auto(frame_proc, rink_profile)

            frame_out = frame_proc.copy()
            if SHOW_BOTTOM_BANNER and BOTTOM_CAPTION_TEXT:
                draw_banner(frame_out)
            frame_prev = frame_out.copy()

            target_found, target_box = False, None

            if TARGET_ID is None:
                cv2.putText(frame_prev, "NO LOCK: press 'r' to lock", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                boxes, ids = track_fn(frame_proc)
                ocr_frame = reader is not None and frame_idx % cfg["ocr_every"] == 0

                if len(ids) and TARGET_ID in ids:
                    target_box = clamp_box(boxes[int(np.where(ids == TARGET_ID)[0][0])], W, H)
                    target_found = True
                    # Keep checking the number while tracked: catches silent ID swaps
                    if ocr_frame:
                        r = reader.read(frame_proc, target_box)
                        if r:
                            voter.add(TARGET_ID, *r)
                        if voter.score(TARGET_ID, TARGET_NUMBER) < 0:
                            print(f"[WARN] ID {TARGET_ID} reads as #{voter.verdict(TARGET_ID)[0]}, "
                                  f"not #{TARGET_NUMBER} — dropping lock and reacquiring.")
                            rejected[TARGET_ID] = frame_idx
                            voter.forget(TARGET_ID)
                            target_found, target_box = False, None

                if not target_found and state["last_center"] is not None and len(ids):
                    lost = state["lost"] + 1
                    drag = 0.95 ** lost
                    pred_x = state["last_center"][0] + state["velocity"][0] * drag * lost
                    pred_y = state["last_center"][1] + state["velocity"][1] * drag * lost
                    radius = SEARCH_RADIUS_BASE + lost * SEARCH_RADIUS_GROWTH

                    best_score, best_id, best_box = -1e9, None, None
                    for box, tid_ in zip(boxes, ids):
                        tid_ = int(tid_)
                        if tid_ in rejected and frame_idx - rejected[tid_] < REJECT_SECONDS * FPS:
                            continue
                        box = clamp_box(box, W, H)
                        cx, cy = box_center(box)
                        dist = math.hypot(cx - pred_x, cy - pred_y)
                        if dist > radius:
                            continue
                        score = 50.0 * (1.0 - dist / radius)
                        bh = box_h(box)
                        if state["last_h"] and abs(bh - state["last_h"]) / state["last_h"] < 0.25:
                            score += 25.0
                        score += 60.0 * iou(state["last_box"], box)
                        score += 40.0 * compare_hist(torso_crop(frame_proc, box), ref_hist)
                        if ocr_frame:
                            r = reader.read(frame_proc, box)
                            if r:
                                voter.add(tid_, *r)
                        if reader is not None:
                            score += NUMBER_BONUS * voter.score(tid_, TARGET_NUMBER)
                        if score > best_score:
                            best_score, best_id, best_box = score, tid_, box

                    if best_id is not None and best_score >= REACQUIRE_THRESHOLD:
                        if state["pending_id"] == best_id:
                            state["pending_count"] += 1
                        else:
                            state["pending_id"], state["pending_count"] = best_id, 1
                        # A voted number match is strong evidence: confirm faster
                        need = SWITCH_CONFIRM_FRAMES // 2 if voter.score(best_id, TARGET_NUMBER) > 0 else SWITCH_CONFIRM_FRAMES
                        if state["pending_count"] >= need:
                            TARGET_ID, target_box, target_found = best_id, best_box, True
                            smoother.reset()
                            print(f"↺ Reacquired as ID {TARGET_ID} (score {best_score:.0f})")

                # Update motion state — exactly one increment of `lost` per missed frame
                if target_found:
                    cx, cy = box_center(target_box)
                    if state["last_center"] is not None:
                        vx, vy = cx - state["last_center"][0], cy - state["last_center"][1]
                        state["velocity"] = (0.6 * state["velocity"][0] + 0.4 * vx,
                                             0.6 * state["velocity"][1] + 0.4 * vy)
                    state.update(last_center=(cx, cy), last_h=box_h(target_box), last_box=target_box.copy(),
                                 lost=0, pending_id=None, pending_count=0)
                else:
                    state["lost"] += 1

                if target_found:
                    draw_box = smoother(target_box) if SMOOTH_OVERLAY else target_box
                    draw_target(frame_out, draw_box)
                    draw_target(frame_prev, draw_box)
                elif state["lost"] <= GRACE_PERIOD_FRAMES:
                    cv2.putText(frame_prev, "LOST: reacquiring...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame_prev, "LOST: press 'r' to relock", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Video", preview(frame_prev))
            writer.write(frame_out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                resume_pos = frame_idx + 1
                action, tid, lock_box, new_hist, _ = relock_mode(cap, frame_proc, frame_idx, track_fn, FPS, rink_profile)
                # relock mode may have scrubbed; put the reader back where the output left off
                cap.set(cv2.CAP_PROP_POS_FRAMES, resume_pos)
                if action == "lock":
                    TARGET_ID, ref_hist = tid, new_hist
                    reset_state(lock_box)
                    print(f"✅ Re-locked TARGET_ID={TARGET_ID}")
                elif action == "skip":
                    TARGET_ID = None
                    print("⚠️ Switched to NO LOCK — press 'r' to relock anytime.")
                else:
                    print("Relock cancelled.")

    except Exception as e:
        print("\n[ERROR] Writer/processing failed:\n" + str(e))
        print("If ffmpeg failed, check the .ffmpeg.log next to your output.")
    finally:
        cap.release()
        try:
            if writer is not None:
                writer.close()
        except Exception as e:
            print("\n[ERROR] Writer failed to finalize output:\n" + str(e))
        cv2.destroyAllWindows()

    print("Done!\nOutput:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
