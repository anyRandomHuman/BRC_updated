from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


VIDEO_DIR = Path('NOHANDS')
OUT = VIDEO_DIR / 'merged.mp4'
TMP_OUT = VIDEO_DIR / 'merged.mp4'
COLS = 3
ROWS = 3


videos = sorted(
    path for path in VIDEO_DIR.glob('*.mp4')
    if not path.stem.startswith('LOCO_6_grid_3x2')
)[:COLS * ROWS]

if len(videos) != COLS * ROWS:
    raise RuntimeError(f'Expected 6 videos in {VIDEO_DIR}, found {len(videos)}')

caps = [cv2.VideoCapture(str(path)) for path in videos]
width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = caps[0].get(cv2.CAP_PROP_FPS) or 30
num_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)

writer = cv2.VideoWriter(
    str(TMP_OUT),
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (COLS * width, ROWS * height),
)

for _ in range(num_frames):
    frames = []
    for cap in caps:
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        frames.append(frame)

    rows = [
        np.hstack(frames[row * COLS:(row + 1) * COLS])
        for row in range(ROWS)
    ]
    writer.write(np.vstack(rows))

for cap in caps:
    cap.release()
writer.release()

if shutil.which('ffmpeg'):
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i', str(TMP_OUT),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            str(OUT),
        ],
        check=True,
    )
    TMP_OUT.unlink()
else:
    TMP_OUT.replace(OUT)

print(f'wrote {OUT}')
