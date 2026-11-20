# Hockey player highlights

Put a marker under a player's skates and a name plate above them, then export the clip with the original audio.

## Setup (Mac)

```bash
brew install ffmpeg
pip install -r requirements.txt
```

Model weights (`yolo26s.pt`, `yolo26n-pose.pt`, `sam2.1_l.pt`, PARSeq) download on first run.
`sam3.pt` is gated: request access at huggingface.co/facebook/sam3 and put the file next to the scripts.

## Which script?

| Script | Use it for | How it tracks |
|---|---|---|
| `sam_track.py` | **Most accurate.** One or more players, you don't mind waiting a bit | You draw a box once; SAM 3 (or SAM 2.1) follows that exact player. Press `r` to stop and fix a box, then continue. |
| `Multiplayer_Highlight.py` | Up to 5 players, fast interactive editing | YOLO26 + BoT-SORT with re-ID, jersey-number voting, manual relock. `(e)ditor` saves data, `(r)ender` burns the video. |
| `Highlight_gen.py` | One player, single pass | Same tracker as above, writes the video while you watch. |

`sam_track.py` and `Multiplayer_Highlight.py` share the same tracking-data file, so the usual flow is:

1. Set `VIDEO_PATH` / `OUTPUT_PATH` at the top of `Multiplayer_Highlight.py`.
2. `python sam_track.py` → draw boxes → ENTER → fix with `r` if it ever drifts → `q`.
3. `python Multiplayer_Highlight.py` → `r` (render).

You can also mix them: track most of the clip in the editor and redo a tricky stretch with SAM. SAM only replaces the slots and frames it tracked.

## Jersey numbers

Set the number per player (`TARGET_NUMBER` in `Highlight_gen.py`, `j` key or `DEFAULT_NUMBER_HINTS` in the multiplayer editor).
Numbers are read with PARSeq on a pose-located torso crop and **voted across frames**. A number only counts once several reads agree.
If a tracked player confidently reads as the wrong number, the lock is dropped and reacquired.

For better accuracy you can set `PARSEQ_WEIGHTS` to a hockey fine-tuned checkpoint from
[mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline).

## Getting more accuracy

- `YOLO_WEIGHTS = "yolo26m.pt"` is slower but finds small or far-away players better.
- Fine-tuning YOLO on ~300 frames from your own rink, with a separate `referee` class, stops the tracker from jumping onto refs.
- In `sam_track.py`, `WORK_HEIGHT = 720` roughly doubles speed; 1080 is sharper for small players.
