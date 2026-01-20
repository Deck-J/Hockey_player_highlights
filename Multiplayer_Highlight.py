#!/usr/bin/env python3
"""
Multiplayer Hockey Highlighter 

WORKFLOW:
1. Run script -> Select (e)ditor.
2. Track players (Space to pause, 'a' to add, 'r' to relock).
3. Quit (auto-saves).
4. Run script -> Select (r)ender to burn final video.

ADVANCED FEATURES:
✅ 5 Slots with Distinct Colors
✅ Anti-Drift Histogram (Prevents learning wrong colors during blocks)
✅ Center-Weighted Masking (Focuses on jersey logo, ignores arms)
✅ Back-Facing Logic (Trusts physics more when logo disappears)
✅ Staggered OCR (Prevents CPU crashes)
"""

import os
import cv2
import json
import math
import time
import sys
import numpy as np
import torch
import ultralytics
import subprocess
from ultralytics import YOLO

# Optional OCR
try:
    import easyocr
except Exception:
    easyocr = None

# ============================================================
# USER CONFIG
# ============================================================
VIDEO_PATH  = "/Users/deckj/Documents/Game_Processor/Noah_powerplay_cloud_apo8_alq13.mp4"
OUTPUT_PATH = "Noah_point_powerplay_marked.mp4"
DATA_FILE   = VIDEO_PATH + ".tracking_data.json"

YOLO_WEIGHTS = "yolo11s.pt"
BASE_CONF = 0.18
MAX_HIGHLIGHTS = 5

# ============================================================
# VISUALS CONFIG
# ============================================================
DRAW_FEET_MARKER = True
FEET_THICKNESS = 4
SLOT_COLORS = [
    (0, 0, 255),     # Slot 1: Red
    (0, 255, 255),   # Slot 2: Yellow
    (255, 0, 255),   # Slot 3: Magenta
    (255, 255, 0),   # Slot 4: Cyan
    (0, 255, 0),     # Slot 5: Green
]

PLATE_FONT_SCALE = 1.25
PLATE_THICKNESS  = 3
PLATE_PAD_X, PLATE_PAD_Y = 22, 14
PLATE_BG_ALPHA = 0.65
PLATE_BORDER_THICKNESS = 3
PLATE_TEXT_COLOR = (255, 255, 255)

SHOW_BOTTOM_BANNER = False
BOTTOM_CAPTION_TEXT = "Bottom caption (press 'b' to toggle)"
BANNER_ALPHA = 0.55
BANNER_Y_MARGIN = 60
BANNER_PAD_X, BANNER_PAD_Y = 18, 14
BANNER_BORDER = 3

HUD_FONT_SCALE = 0.95
HUD_THICKNESS = 2
HUD_BG_ALPHA = 0.55
HUD_BG_COLOR = (90, 40, 10)
HUD_TEXT_COLOR = (255, 255, 255)

# ============================================================
# OCR CONFIG (Staggered for Stability)
# ============================================================
USE_OCR = True
OCR_GPU = False
OCR_EVERY_N_FRAMES_WHEN_LOST = 45  # Check every 1.5s
OCR_MIN_LOST_FRAMES = 30           # Wait 1s before checking
OCR_ALLOWLIST = "0123456789"
DEFAULT_NUMBER_HINTS = ["", "", "", "", ""]

# ============================================================
# TRACKING TUNING (Hockey Optimized)
# ============================================================
REACQUIRE_THRESHOLD = 135
MAX_LOST_BEFORE_SWITCH = 60
SEARCH_RADIUS_BASE = 250       # WIDER for hockey gliding
SEARCH_RADIUS_GROWTH = 12      # Grow faster
SWITCH_CONFIRM_FRAMES = 30     # Lock on faster (was 45)
MIN_IOU_TO_KEEP = 0.06
MAX_HEIGHT_RATIO_DELTA = 0.30
MIN_HIST_SIM_TO_KEEP = 0.18

# ============================================================
# RENDERER: FFmpeg Writer
# ============================================================
HEVC_BITRATE   = "8M"
HEVC_MAXRATE   = "10M"
HEVC_BUFSIZE   = "16M"
AUDIO_BITRATE  = "192k"

class FFmpegVideoWriter:
    def __init__(self, src_video, out_video, width, height, fps):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.cmd = [
            "ffmpeg", "-y",
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
            out_video,
        ]
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=10**7)

    def write(self, frame_bgr):
        if self.proc.poll() is not None: raise RuntimeError("FFmpeg exited unexpectedly.")
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        try: self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError: raise RuntimeError("FFmpeg pipe broke.")

    def close(self):
        if self.proc.stdin: self.proc.stdin.close()
        self.proc.wait()

# ============================================================
# CORE UTILS
# ============================================================
def pick_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"
DEVICE = pick_device()

def get_builtin_bytetrack_yaml():
    return os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "bytetrack.yaml")

def detect_video_props(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0: fps = 30.0
    return w, h, fps

def autotune(w, h, fps):
    px = w * h
    is_4k = px >= int(3840 * 2160 * 0.85)
    return {"imgsz": 960 if not is_4k else 1024, "conf": BASE_CONF if not is_4k else max(BASE_CONF, 0.20)}

def clamp_box(box, W, H):
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def box_center(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def box_h(box):
    return float(box[3]-box[1])

def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def torso_crop(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2-y1, x2-x1
    cy1, cy2 = int(y1+0.12*h), int(y1+0.60*h)
    cx1, cx2 = int(x1+0.18*w), int(x2-0.18*w)
    cy1=max(0,cy1); cy2=min(frame.shape[0],cy2)
    cx1=max(0,cx1); cx2=min(frame.shape[1],cx2)
    if cy2<=cy1 or cx2<=cx1: return None
    return frame[cy1:cy2, cx1:cx2]

# [IMPROVEMENT] Center-Weighted Mask for Histograms
def get_color_hist(frame, box):
    crop = torso_crop(frame, box)
    if crop is None or crop.size == 0: return None
    
    # Create circular mask to focus on chest/logo
    h, w = crop.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(w / 2), int(h / 2))
    radius = int(min(h, w) * 0.4) # Inner 40%
    cv2.circle(mask, center, radius, 255, -1)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [90, 128], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def compare_hist(crop_bgr, ref_hist):
    if crop_bgr is None or ref_hist is None: return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [90, 128], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return float(cv2.compareHist(ref_hist, hist, cv2.HISTCMP_CORREL))

def safe_ocr(reader, crop):
    if reader is None or crop is None or crop.size < 100: return []
    try:
        h, w = crop.shape[:2]
        if max(h, w) > 220:
            s = 220.0/max(h, w)
            crop = cv2.resize(crop, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        res = reader.readtext(crop, allowlist=OCR_ALLOWLIST, detail=1, paragraph=False)
        return [(txt.strip(), float(conf)) for _, txt, conf in res if txt.strip()]
    except: return []

def run_track(model, frame, tracker_yaml, cfg):
    res = model.track(frame, persist=True, tracker=tracker_yaml, conf=cfg["conf"], imgsz=cfg["imgsz"], classes=[0], device=DEVICE, verbose=False)
    if res[0].boxes.id is None: return np.empty((0, 4), np.float32), np.empty((0,), np.int32)
    return res[0].boxes.xyxy.cpu().numpy().astype(np.float32), res[0].boxes.id.int().cpu().numpy().astype(np.int32)

# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_scene(frame, highlights_data, show_banner, banner_text):
    if show_banner and banner_text: draw_bottom_banner(frame, banner_text)
    for h in highlights_data:
        box, color, label, is_selected = h['box'], h['color'], h['label'], h.get('selected', False)
        if DRAW_FEET_MARKER: draw_feet_marker(frame, box, color)
        draw_plate_name(frame, box, label, is_selected, color)

def draw_feet_marker(frame, box, color):
    x1, y1, x2, y2 = map(int, box)
    cx, fy = (x1+x2)//2, min(frame.shape[0]-1, int(y2 + 0.03*max(10, y2-y1)))
    w = max(10, x2-x1)
    cv2.ellipse(frame, (cx, fy), (max(6, int(w*0.42)), max(4, int(w*0.15))), 0, 0, 360, color, FEET_THICKNESS, cv2.LINE_AA)

def draw_plate_name(frame, box, text, selected, color):
    x1, y1, _, _ = map(int, box)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, PLATE_FONT_SCALE, PLATE_THICKNESS)
    rx1, ry2 = max(0, x1), max(0, y1 - 12)
    ry1, rx2 = max(0, ry2 - (th + PLATE_PAD_Y*2)), min(frame.shape[1]-1, rx1 + tw + PLATE_PAD_X*2)
    over = frame.copy()
    cv2.rectangle(over, (rx1, ry1), (rx2, ry2), (0,0,0), -1)
    cv2.addWeighted(over, PLATE_BG_ALPHA, frame, 1-PLATE_BG_ALPHA, 0, frame)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0,255,255) if selected else color, PLATE_BORDER_THICKNESS)
    cv2.putText(frame, text, (rx1+PLATE_PAD_X, ry2-PLATE_PAD_Y), cv2.FONT_HERSHEY_SIMPLEX, PLATE_FONT_SCALE, PLATE_TEXT_COLOR, PLATE_THICKNESS, cv2.LINE_AA)

def draw_bottom_banner(frame, text):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.05, 2)
    bw, bh = tw + BANNER_PAD_X*2, th + BANNER_PAD_Y*2
    x1, y2 = (w-bw)//2, h - BANNER_Y_MARGIN
    y1, x2 = y2 - bh, x1 + bw
    over = frame.copy()
    cv2.rectangle(over, (x1, y1), (x2, y2), (0,0,0), -1)
    cv2.addWeighted(over, BANNER_ALPHA, frame, 1-BANNER_ALPHA, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,255), BANNER_BORDER)
    cv2.putText(frame, text, (x1+BANNER_PAD_X, y2-BANNER_PAD_Y), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255,255,255), 2, cv2.LINE_AA)

def draw_hud(frame, lines):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, HUD_FONT_SCALE, HUD_THICKNESS
    lh = int(28 * (scale/0.75))
    x0, y0 = 18, 18
    wmax = max([cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines] or [0])
    x1, y1 = x0 + wmax + 30, y0 + lh*len(lines) + 18
    over = frame.copy()
    cv2.rectangle(over, (x0, y0), (x1, y1), HUD_BG_COLOR, -1)
    cv2.addWeighted(over, HUD_BG_ALPHA, frame, 1-HUD_BG_ALPHA, 0, frame)
    for i, s in enumerate(lines):
        cv2.putText(frame, s, (x0+14, y0+30+i*lh), font, scale, HUD_TEXT_COLOR, thick, cv2.LINE_AA)

def print_menu(paused, highlights, selected):
    print("\n--- CONTROLS ---")
    print("SPACE: Pause/Play")
    print("a: Add Highlight")
    print("r: Relock (1-5)")
    print("x: Delete Selected")
    print("t: Edit Label")
    print("s: SAVE Data")
    print("q: Quit (Auto-save)")
    print(f"Status: {'PAUSED' if paused else 'PLAY'}")
    if highlights:
        print(f"Highlights ({len(highlights)}/{MAX_HIGHLIGHTS}):")
        for i, h in enumerate(highlights):
            mk = "<==" if i==selected else ""
            print(f"  [{i+1}] Slot {h['slot']+1}: {h['label']} {mk}")

# ============================================================
# TRACKER LOGIC
# ============================================================
def update_highlight_tracker(hh, boxes, ids, W, H, frame, fidx, ocr_reader):
    tid = hh["id"]
    found, tbox = False, None
    
    # 1. SAME ID
    if len(ids)>0 and tid in ids:
        idx = int(np.where(ids == tid)[0][0])
        tbox = clamp_box(boxes[idx], W, H)
        found = True
        
        # [IMPROVEMENT] Anti-Drift: Only update histogram if similarity is high
        if hh["lost"] == 0 and fidx % 30 == 0:
            cur_hist = get_color_hist(frame, tbox)
            if hh.get("ref_hist") is not None and cur_hist is not None:
                sim = float(cv2.compareHist(hh["ref_hist"], cur_hist, cv2.HISTCMP_CORREL))
                if sim > 0.85: # Only blend if highly similar
                    cv2.accumulateWeighted(cur_hist, hh["ref_hist"], 0.1)
                    cv2.normalize(hh["ref_hist"], hh["ref_hist"], 0, 1, cv2.NORM_MINMAX)
            elif hh.get("ref_hist") is None:
                hh["ref_hist"] = cur_hist
            
    # 2. REACQUIRE
    if not found and hh["last_center"] and len(ids)>0:
        hh["lost"] += 1
        lost = hh["lost"]
        allow_switch = lost >= MAX_LOST_BEFORE_SWITCH
        drag = 0.95 ** lost
        px, py = hh["last_center"][0] + hh["vel"][0]*drag*lost, hh["last_center"][1] + hh["vel"][1]*drag*lost
        rad = SEARCH_RADIUS_BASE + lost*SEARCH_RADIUS_GROWTH
        
        best_s, best_id, best_box = -1e9, None, None
        
        for box, tid_ in zip(boxes, ids):
            box = clamp_box(box, W, H)
            cx, cy = box_center(box)
            dist = math.hypot(cx-px, cy-py)
            if dist > rad: continue
            
            # [ADJUSTMENT] Higher motion weight for back-facing players
            s = 70.0 * (1.0 - dist/rad)
            if hh["last_box"] is not None: s += 130.0 * iou(hh["last_box"], box)
            
            if not allow_switch and iou(hh["last_box"], box) < MIN_IOU_TO_KEEP: continue
            
            bh = box_h(box)
            if hh["last_h"] > 0:
                if abs(bh - hh["last_h"])/hh["last_h"] <= MAX_HEIGHT_RATIO_DELTA: s += 18.0
                else: s -= 25.0
            
            sim = compare_hist(torso_crop(frame, box), hh.get("ref_hist"))
            # [ADJUSTMENT] Slightly lower histogram weight (85 vs 95)
            s += 85.0 * sim
            if not allow_switch and sim < MIN_HIST_SIM_TO_KEEP: continue
            
            # Staggered OCR
            if USE_OCR and ocr_reader and lost >= OCR_MIN_LOST_FRAMES and (fidx + hh["slot"]) % OCR_EVERY_N_FRAMES_WHEN_LOST == 0:
                hint = hh.get("hint", "")
                crop = torso_crop(frame, box)
                if hint and crop is not None:
                    items = safe_ocr(ocr_reader, crop)
                    for t, c in items:
                        if hint in t and c>=0.3: s+=80.0
                        elif c>=0.85 and t!=hint: s-=60.0
            
            if s > best_s: best_s, best_id, best_box = s, int(tid_), box
            
        if best_id is not None and best_s >= REACQUIRE_THRESHOLD:
            if hh["pend_id"] == best_id: hh["pend_cnt"] += 1
            else: hh["pend_id"], hh["pend_cnt"] = best_id, 1
            
            if hh["pend_cnt"] >= SWITCH_CONFIRM_FRAMES:
                hh["id"], tbox, found = best_id, best_box, True
                hh["lost"], hh["pend_id"], hh["pend_cnt"] = 0, None, 0
                h = get_color_hist(frame, tbox)
                if h is not None: hh["ref_hist"] = h

    if found and tbox is not None:
        cx, cy = box_center(tbox)
        if hh["last_center"]:
            vx, vy = cx-hh["last_center"][0], cy-hh["last_center"][1]
            hh["vel"] = (0.65*hh["vel"][0]+0.35*vx, 0.65*hh["vel"][1]+0.35*vy)
        hh["last_center"], hh["last_h"], hh["last_box"] = (cx, cy), box_h(tbox), tbox.copy()
        hh["lost"] = 0
        return True, tbox
    return False, None

# ============================================================
# EDITOR MODE (Interactive)
# ============================================================
def editor_mode():
    print("\n>>> EDITOR MODE")
    print(">>> No video encoding = High Speed.")
    print(f">>> Data will be saved to: {DATA_FILE}\n")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    W, H, FPS = detect_video_props(cap)
    cfg = autotune(W, H, FPS)
    tracker_yaml = get_builtin_bytetrack_yaml()
    model = YOLO(YOLO_WEIGHTS)
    
    ocr_reader = None
    if USE_OCR and easyocr:
        ocr_reader = easyocr.Reader(["en"], gpu=OCR_GPU)
    
    cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Editor", cv2.WND_PROP_TOPMOST, 1) # Try to force foreground
    
    # State
    tracking_db = {} 
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                tracking_db = {int(k): v for k,v in json.load(f).items()}
            print(f">>> Loaded existing tracking data ({len(tracking_db)} frames).")
        except: pass

    highlights = [] 
    selected = 0
    paused = True
    show_hud = True
    fidx = 0
    show_banner = SHOW_BOTTOM_BANNER
    editor_state = {"active": False, "text": ""}
    
    # Print menu immediately
    print_menu(paused, highlights, selected)
    
    # ROI helpers
    roi = {"active": False, "box": (0,0,0,0), "sel": False}
    def mouse_cb(event, x, y, f, p):
        s = min(PREVIEW_MAX_W/W, PREVIEW_MAX_H/H, 1.0)
        mx, my = int(x/s), int(y/s)
        if event==cv2.EVENT_LBUTTONDOWN: roi["box"] = (mx,my,0,0); roi["sel"] = False
        elif event==cv2.EVENT_LBUTTONUP:
            x1,y1,w,h = roi["box"][0], roi["box"][1], mx-roi["box"][0], my-roi["box"][1]
            if w<0: x1+=w; w=abs(w)
            if h<0: y1+=h; h=abs(h)
            roi["box"] = (int(x1), int(y1), int(w), int(h)); roi["sel"] = True
    cv2.setMouseCallback("Editor", mouse_cb)

    # ROI Lock Logic
    def run_roi_mode(start_frame):
        roi["active"] = True; roi["sel"] = False; roi["box"] = (0,0,0,0)
        curr_f = start_frame.copy()
        cand = []
        cidx = 0
        frozen = False
        while True:
            disp = curr_f.copy()
            cv2.rectangle(disp, (0,0), (disp.shape[1], 80), (0,0,0), -1)
            cv2.putText(disp, "ROI MODE: Draw box. SPACE to lock. 'n' next cand. 'q' cancel.", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            if roi["box"][2]>0:
                x,y,w,h = roi["box"]
                cv2.rectangle(disp, (x,y), (x+w, y+h), (255,0,0), 3)
            if roi["sel"] and not frozen:
                b, i = run_track(model, curr_f, tracker_yaml, cfg)
                cand = candidates_near_roi_fullres(b, i, roi["box"])
                frozen = True; cidx = 0
            if cand:
                c = cand[cidx]
                x1,y1,x2,y2 = map(int, c['box'])
                cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(disp, f"ID: {c['id']}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            
            disp_s = resize_for_preview(disp)
            cv2.imshow("Editor", disp_s)
            k = cv2.waitKey(20) & 0xFF
            
            if not roi["sel"]: frozen = False; cand = []
            if k == ord('q'): return None
            if k == ord('n') and cand: cidx = (cidx+1)%len(cand)
            if k in (32, 13) and cand: return cand[cidx]

    while True:
        if not paused:
            ret, current_frame = cap.read()
            if not ret: break
            fidx += 1
        else:
            if 'current_frame' not in locals():
                ret, current_frame = cap.read()
                if not ret: break
                fidx = 1

        boxes, ids = np.empty((0,4)), np.empty((0,))
        if highlights: boxes, ids = run_track(model, current_frame, tracker_yaml, cfg)
        
        frame_out = current_frame.copy()
        frame_db_entry = [] 
        
        for i, hh in enumerate(highlights):
            found, box = update_highlight_tracker(hh, boxes, ids, W, H, current_frame, fidx, ocr_reader)
            if found:
                frame_db_entry.append({"slot": hh["slot"], "box": [float(x) for x in box], "label": hh["label"], "color": hh["color"]})
                draw_plate_name(frame_out, box, hh["label"], (i==selected), hh["color"])
                if DRAW_FEET_MARKER: draw_feet_marker(frame_out, box, hh["color"])
        
        if frame_db_entry: tracking_db[fidx] = frame_db_entry
        elif fidx in tracking_db and not paused and highlights: del tracking_db[fidx]

        if show_banner: draw_bottom_banner(frame_out, BOTTOM_CAPTION_TEXT)

        if show_hud:
            lines = [f"EDITOR | Frame {fidx} | {'PAUSED' if paused else 'PLAY'} | Data: {len(tracking_db)} frames"]
            lines.append("SPACE: Pause | a: Add | r: Relock | t: Label | s: SAVE | q: Quit")
            if highlights:
                h = highlights[selected]
                lines.append(f"Selected: [{h['slot']+1}] {h['label']} (ID {h['id']})")
            if editor_state["active"]: lines.append(f"EDITING: {editor_state['text']}_")
            draw_hud(frame_out, lines)

        disp = resize_for_preview(frame_out)
        cv2.imshow("Editor", disp)
        
        # INCREASED WAITKEY FOR RESPONSIVENESS
        k = cv2.waitKey(20) & 0xFF
        
        if editor_state["active"]:
            if k == 13: 
                if highlights: highlights[selected]["label"] = editor_state["text"]
                editor_state["active"] = False
            elif k == 27: editor_state["active"] = False
            elif k == 127 or k == 8: editor_state["text"] = editor_state["text"][:-1]
            elif 32 <= k <= 126: editor_state["text"] += chr(k)
            continue

        if k == ord('q'): break
        if k == 32: 
            paused = not paused
            print_menu(paused, highlights, selected)
        if k == ord('s'):
            with open(DATA_FILE, 'w') as f: json.dump(tracking_db, f)
            print(f">>> Saved to {DATA_FILE}")
        if k == ord('h'): show_hud = not show_hud
        if k == ord('b'): show_banner = not show_banner
        if k == 9: 
            if highlights: selected = (selected+1)%len(highlights)
        if k == ord('x'): 
            if highlights:
                print(f"Deleted {highlights[selected]['label']}")
                highlights.pop(selected)
                if selected >= len(highlights): selected = max(0, len(highlights)-1)
        if k == ord('t'): 
            if highlights:
                editor_state["active"] = True
                editor_state["text"] = highlights[selected]["label"]

        if k == ord('a'):
            if len(highlights) >= MAX_HIGHLIGHTS: print("Max highlights."); continue
            res = run_roi_mode(current_frame)
            if res and isinstance(res, dict):
                slots = {h['slot'] for h in highlights}
                s = next(i for i in range(MAX_HIGHLIGHTS) if i not in slots)
                hh = {
                    "slot": s, "color": SLOT_COLORS[s], "id": int(res['id']),
                    "label": f"Player {s+1}", "hint": DEFAULT_NUMBER_HINTS[s],
                    "last_box": res['box'], "last_center": box_center(res['box']),
                    "last_h": box_h(res['box']), "vel": (0.0,0.0), "lost": 0,
                    "pend_id": None, "pend_cnt": 0, "ref_hist": get_color_hist(current_frame, res['box'])
                }
                highlights.append(hh)
                selected = len(highlights)-1
                roi["active"] = False; cv2.setMouseCallback("Editor", mouse_cb)

        if k == ord('r'):
            if not highlights: continue
            print("Press 1-5 to relock slot...")
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == 27: break
                if ord('1') <= k2 <= ord('5'):
                    slot = int(chr(k2)) - 1
                    tgt = next((h for h in highlights if h['slot'] == slot), None)
                    if not tgt: print("Slot empty."); break
                    res = run_roi_mode(current_frame)
                    if res and isinstance(res, dict):
                        tgt['id'] = int(res['id'])
                        tgt['last_box'] = res['box']
                        tgt['last_center'] = box_center(res['box'])
                        tgt['last_h'] = box_h(res['box'])
                        tgt['vel'] = (0.0,0.0); tgt['lost'] = 0
                        tgt['ref_hist'] = get_color_hist(current_frame, res['box'])
                        print(f"Relocked Slot {slot+1}")
                    roi["active"] = False; cv2.setMouseCallback("Editor", mouse_cb)
                    break

    with open(DATA_FILE, 'w') as f: json.dump(tracking_db, f)
    print(f">>> Auto-saved data to {DATA_FILE}")
    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# RENDER MODE (Offline)
# ============================================================
def render_mode():
    print("\n>>> RENDER MODE")
    if not os.path.exists(DATA_FILE):
        print(f"Error: No data file found at {DATA_FILE}")
        print("Run Editor Mode (e) first.")
        return

    with open(DATA_FILE, 'r') as f:
        tracking_db = {int(k): v for k,v in json.load(f).items()}
    
    print(f">>> Loaded {len(tracking_db)} frames of data.")
    print(">>> Starting FFmpeg render...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    W, H, FPS = detect_video_props(cap)
    writer = FFmpegVideoWriter(VIDEO_PATH, OUTPUT_PATH, W, H, FPS)
    
    fidx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        fidx += 1
        if fidx % 30 == 0:
            sys.stdout.write(f"\rRendering frame {fidx}/{total} ({(fidx/total)*100:.1f}%)")
            sys.stdout.flush()
        if fidx in tracking_db:
            highlights = tracking_db[fidx]
            for h in highlights: h['selected'] = False 
            draw_scene(frame, highlights, SHOW_BOTTOM_BANNER, BOTTOM_CAPTION_TEXT)
        writer.write(frame)
        
    print("\n>>> Finalizing...")
    cap.release()
    writer.close()
    print(f">>> Done! Output: {OUTPUT_PATH}")

# ============================================================
# HELPERS
# ============================================================
PREVIEW_MAX_W = 1920
PREVIEW_MAX_H = 1080
def resize_for_preview(frame):
    h, w = frame.shape[:2]
    scale = min(PREVIEW_MAX_W/w, PREVIEW_MAX_H/h, 1.0)
    if scale >= 1.0: return frame
    return cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def candidates_near_roi_fullres(boxes, ids, roi_box):
    rx, ry, rw, rh = roi_box
    rcx, rcy = rx+rw/2, ry+rh/2
    out = []
    for b, i in zip(boxes, ids):
        bcx, bcy = box_center(b)
        if rx-50 < bcx < rx+rw+50 and ry-50 < bcy < ry+rh+50:
            dist = math.hypot(bcx-rcx, bcy-rcy)
            out.append({'id': i, 'box': b, 'dist': dist})
    out.sort(key=lambda x: x['dist'])
    return out

if __name__ == "__main__":
    print(f"Target Video: {VIDEO_PATH}")
    m = input("Select Mode -> (e)ditor or (r)ender: ").strip().lower()
    if m.startswith('e'): editor_mode()
    elif m.startswith('r'): render_mode()
    else: print("Invalid mode.")
