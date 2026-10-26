from pathlib import Path

import cv2


VIDEO_DIR = Path('LOCO_6/cat')


for video_path in sorted(VIDEO_DIR.glob('*.mp4')):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay_ms = max(1, int(1000 / fps))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow(str(video_path), frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit
        if key != 255:
            break

    cap.release()
    cv2.destroyAllWindows()
