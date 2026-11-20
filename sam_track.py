#!/usr/bin/env python3
"""
SAM click-to-track editor (most accurate option)

Instead of "detect every person, link boxes, then guess which one is yours",
you draw a box around each player once and Meta's Segment Anything model
(SAM 3, or SAM 2.1 as a fallback) follows that exact player through the clip
using its video memory. No colour histograms, no ID switches to undo.

It writes the SAME tracking-data JSON as Multiplayer_Highlight.py, so you then
render with:   python Multiplayer_Highlight.py  ->  (r)ender

VIDEO_PATH / DATA_FILE / slot colours are taken from Multiplayer_Highlight.py,
so you only set your video in one place.

WORKFLOW
  1. Scrub to where you want to start, pick a slot (1-5), drag a box around the player.
     Repeat for up to 5 players.  t = edit label of the active slot.
  2. ENTER = track forward from this frame.
     While tracking:  r = stop here and fix (returns to step 1 on the current frame,
                          boxes pre-filled; fix the bad one and press ENTER again)
                      q = stop and save
  3. Data is merged into the JSON (only the slots you tracked, only the frames you covered).

KEYS (picker)
  1-5 select slot | drag = box for active slot | x = clear slot | t = label
  . / ,  = +/- 1 frame | ] / [ = +/- 1 second | SPACE = play/pause
  ENTER = start tracking | q = save + quit

WEIGHTS
  sam3.pt is gated: request access at huggingface.co/facebook/sam3, download sam3.pt and
  put it next to this script. If it isn't found, SAM 2.1 (sam2.1_l.pt, auto-download) is used.

SPEED
  SAM runs on a downscaled copy (WORK_HEIGHT) and boxes are scaled back to full res.
  On Apple Silicon expect a few frames per second at 1080p work size; 720p is ~2x faster.
"""

import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

import Multiplayer_Highlight as MH
from hockey_common import detect_video_props, draw_feet_marker, draw_plate, pick_device, preview_scale, resize_for_preview

# ============================================================
# CONFIG
# ============================================================
VIDEO_PATH = MH.VIDEO_PATH
DATA_FILE = MH.DATA_FILE
SLOT_COLORS = MH.SLOT_COLORS
MAX_SLOTS = MH.MAX_HIGHLIGHTS

SAM_WEIGHTS = "sam3.pt"            # gated; see header
SAM_FALLBACK_WEIGHTS = "sam2.1_l.pt"  # auto-downloads (sam2.1_b.pt is faster)
WORK_HEIGHT = 1080                 # SAM processing height (720 = faster, 1080 = sharper)
MAX_SEGMENT_SECONDS = None         # e.g. 60 to process at most a minute per run
SAM_CONF = 0.25

DEVICE = pick_device()


# ============================================================
# HELPERS
# ============================================================
def load_db():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_db(db):
    if os.path.exists(DATA_FILE):
        shutil.copyfile(DATA_FILE, DATA_FILE + ".bak")
    with open(DATA_FILE, "w") as f:
        json.dump({str(k): v for k, v in sorted(db.items())}, f)
    print(f">>> Saved {len(db)} frames to {DATA_FILE} (previous version: .bak)")


def mask_to_box(mask):
    """
    Tight player box from a SAM mask, trimming thin outliers (the stick) with
    percentiles so the feet marker stays centred on the skates.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    x1, x2 = np.percentile(xs, [4, 96])
    y1, y2 = np.percentile(ys, [1, 99.5])
    return [float(x1), float(y1), float(x2), float(y2)]


def extract_segment(src, start0, end0, work_h, out_path):
    """Frame-exact cut of frames [start0, end0] (0-based), scaled to work_h, for SAM."""
    vf = f"select='between(n\\,{start0}\\,{end0})',scale=-2:{work_h}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src, "-vf", vf,
           "-fps_mode", "passthrough", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", out_path]
    subprocess.run(cmd, check=True)


def build_predictor():
    weights = SAM_WEIGHTS
    if not os.path.exists(weights) and weights.startswith("sam3"):
        print(f"[SAM] {weights} not found (gated download, see header). Falling back to {SAM_FALLBACK_WEIGHTS}.")
        weights = SAM_FALLBACK_WEIGHTS
    overrides = dict(conf=SAM_CONF, task="segment", mode="predict", model=weights, save=False, verbose=False)
    if DEVICE == "cuda":
        overrides["half"] = True
    if weights.startswith("sam3"):
        from ultralytics.models.sam import SAM3VideoPredictor as P
    else:
        from ultralytics.models.sam import SAM2VideoPredictor as P
    print(f"[SAM] Using {weights} on {DEVICE}")
    return P(overrides=overrides)


# ============================================================
# PICKER
# ============================================================
def picker(cap, start_idx, prompts, W, H, fps, pscale):
    """
    Interactive prompt picker. prompts: {slot: {"box": [x1,y1,x2,y2] full-res, "label": str}}
    Returns ("track", frame_idx, prompts) or ("quit", ...).
    """
    state = {"drag": None, "cur": None, "active": min(prompts) if prompts else 0}
    editing = {"on": False, "text": ""}

    def mouse(event, x, y, flags, param):
        fx, fy = x / pscale, y / pscale
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (fx, fy); state["cur"] = (fx, fy)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["cur"] = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            (ax, ay), (bx, by) = state["drag"], (fx, fy)
            x1, x2 = sorted((ax, bx)); y1, y2 = sorted((ay, by))
            if x2 - x1 > 6 and y2 - y1 > 6:
                s = state["active"]
                label = prompts.get(s, {}).get("label", f"Player {s + 1}")
                prompts[s] = {"box": [x1, y1, x2, y2], "label": label}
            state["drag"] = state["cur"] = None

    cv2.setMouseCallback("SAM", mouse)

    def read_at(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        return f if ok else None

    idx = start_idx
    frame = read_at(idx)
    playing = False
    while True:
        disp = frame.copy()
        for s, p in prompts.items():
            c = SLOT_COLORS[s]
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(disp, (x1, y1), (x2, y2), c, 4 if s == state["active"] else 2)
            draw_plate(disp, p["box"], f"{s + 1}: {p['label']}", c, 1.0, 2, 12, 8, 0.6, 2, 10)
        if state["drag"] and state["cur"]:
            (ax, ay), (bx, by) = state["drag"], state["cur"]
            cv2.rectangle(disp, (int(ax), int(ay)), (int(bx), int(by)), SLOT_COLORS[state["active"]], 2)

        hud = (f"SAM PICKER | frame {idx + 1} | slot {state['active'] + 1} | 1-5 slot, drag box, x clear, "
               f"t label, . , [ ] scrub, ENTER track, q quit")
        if editing["on"]:
            hud = f"LABEL slot {state['active'] + 1}: {editing['text']}_   (ENTER ok, ESC cancel)"
        pv = resize_for_preview(disp, pscale)
        cv2.rectangle(pv, (0, 0), (pv.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(pv, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("SAM", pv)
        k = cv2.waitKey(15 if not playing else max(1, int(1000 / fps))) & 0xFF

        if editing["on"]:
            if k == 13:
                if state["active"] in prompts:
                    prompts[state["active"]]["label"] = editing["text"]
                editing["on"] = False
            elif k == 27:
                editing["on"] = False
            elif k in (8, 127):
                editing["text"] = editing["text"][:-1]
            elif 32 <= k <= 126:
                editing["text"] += chr(k)
            continue

        if k == ord("q"):
            return "quit", idx, prompts
        if k == 13:
            if prompts:
                return "track", idx, prompts
            print("Draw at least one box first.")
        if ord("1") <= k < ord("1") + MAX_SLOTS:
            state["active"] = k - ord("1")
        if k == ord("x"):
            prompts.pop(state["active"], None)
        if k == ord("t") and state["active"] in prompts:
            editing.update(on=True, text=prompts[state["active"]]["label"])
        if k == 32:
            playing = not playing

        step = {ord("."): 1, ord(","): -1, ord("]"): int(round(fps)), ord("["): -int(round(fps))}.get(k, 0)
        if playing and not step:
            step = 1
        if step:
            nidx = int(np.clip(idx + step, 0, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)))
            f = read_at(nidx)
            if f is not None and nidx != idx:
                idx, frame = nidx, f
            elif playing:
                playing = False


# ============================================================
# TRACK ONE SEGMENT
# ============================================================
def track_segment(start0, prompts, W, H, fps, total, pscale, db):
    """Runs SAM from frame start0 (0-based). Returns ("done"|"fix"|"quit", last_frame0, last_boxes)."""
    end0 = total - 1
    if MAX_SEGMENT_SECONDS:
        end0 = min(end0, start0 + int(MAX_SEGMENT_SECONDS * fps))

    work_h = min(WORK_HEIGHT, H)
    scale = H / float(work_h)
    tmpdir = tempfile.mkdtemp(prefix="sam_seg_")
    seg_path = os.path.join(tmpdir, "segment.mp4")
    print(f">>> Preparing frames {start0 + 1}..{end0 + 1} at {work_h}p for SAM...")
    extract_segment(VIDEO_PATH, start0, end0, work_h, seg_path)

    slots = sorted(prompts)
    bboxes = [[v / scale for v in prompts[s]["box"]] for s in slots]

    predictor = build_predictor()
    results = predictor(source=seg_path, bboxes=bboxes, stream=True)

    src = cv2.VideoCapture(VIDEO_PATH)
    src.set(cv2.CAP_PROP_POS_FRAMES, start0)

    status, last0, last_boxes = "done", start0, {s: prompts[s]["box"] for s in slots}
    try:
        for i, r in enumerate(results):
            f0 = start0 + i
            ok, full = src.read()
            if not ok:
                break

            entries, boxes_now = [], {}
            if r.boxes is not None and len(r.boxes):
                objs = r.boxes.cls.int().cpu().numpy()       # object index == prompt order
                xyxy = r.boxes.xyxy.cpu().numpy()
                masks = r.masks.data.cpu().numpy() if r.masks is not None else None
                for j, oi in enumerate(objs):
                    if oi >= len(slots):
                        continue
                    box = mask_to_box(masks[j]) if masks is not None else None
                    if box is None:
                        box = [float(v) for v in xyxy[j]]
                    box = [b * scale for b in box]
                    box = [max(0.0, min(W - 1.0, box[0])), max(0.0, min(H - 1.0, box[1])),
                           max(0.0, min(W - 1.0, box[2])), max(0.0, min(H - 1.0, box[3]))]
                    s = slots[oi]
                    boxes_now[s] = box
                    entries.append({"slot": s, "box": box, "label": prompts[s]["label"],
                                    "color": list(SLOT_COLORS[s])})

            # Merge into db (1-based frame keys): replace only the slots we're tracking
            key = f0 + 1
            kept = [e for e in db.get(key, []) if e["slot"] not in prompts]
            if kept or entries:
                db[key] = kept + entries
            else:
                db.pop(key, None)

            last0 = f0
            last_boxes.update(boxes_now)

            disp = full.copy()
            for e in entries:
                draw_feet_marker(disp, e["box"], tuple(e["color"]))
                draw_plate(disp, e["box"], e["label"], tuple(e["color"]), 1.0, 2, 12, 8, 0.6, 2, 10)
            pv = resize_for_preview(disp, pscale)
            missing = [str(s + 1) for s in slots if s not in boxes_now]
            msg = f"SAM TRACKING frame {key}/{end0 + 1} | r = stop & fix | q = stop & save"
            if missing:
                msg += f" | not visible: slot {','.join(missing)}"
            cv2.rectangle(pv, (0, 0), (pv.shape[1], 44), (0, 0, 0), -1)
            cv2.putText(pv, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("SAM", pv)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("r"):
                status = "fix"
                break
            if k == ord("q"):
                status = "quit"
                break
    finally:
        src.release()
        shutil.rmtree(tmpdir, ignore_errors=True)
    return status, last0, last_boxes


# ============================================================
# MAIN
# ============================================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open", VIDEO_PATH)
        return
    W, H, fps = detect_video_props(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pscale = preview_scale(W, H)
    db = load_db()
    print(f"Video: {W}x{H} @ {fps:.2f}fps, {total} frames. Existing data: {len(db)} frames.")

    cv2.namedWindow("SAM", cv2.WINDOW_NORMAL)

    # Start with whatever is already saved for the first frame
    prompts = {e["slot"]: {"box": e["box"], "label": e["label"]} for e in db.get(1, [])}
    idx = 0
    while True:
        action, idx, prompts = picker(cap, idx, prompts, W, H, fps, pscale)
        if action == "quit":
            break
        status, last0, last_boxes = track_segment(idx, prompts, W, H, fps, total, pscale, db)
        save_db(db)
        if status == "quit":
            break
        # "fix" or finished: go back to the picker on the last tracked frame with boxes pre-filled
        idx = last0
        prompts = {s: {"box": last_boxes[s], "label": prompts[s]["label"]} for s in prompts if s in last_boxes}
        if status == "done":
            print(">>> Reached the end of the segment. Adjust and ENTER to continue, or q to finish.")

    cap.release()
    cv2.destroyAllWindows()
    print(">>> Done. Render with: python Multiplayer_Highlight.py  ->  r")


if __name__ == "__main__":
    main()
