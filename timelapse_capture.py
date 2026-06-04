from picamera2 import Picamera2
from libcamera import controls
import time
import os
import subprocess
import sys
import cv2

# -------------------------
# MODE SELECTION
# -------------------------
mode = sys.argv[1] if len(sys.argv) > 1 else "1080p"

if mode == "4k":
    W, H = 3840, 2160
    FPS = 5
    FRAMES = 50
elif mode == "1080p":
    W, H = 1920, 1080
    FPS = 20
    FRAMES = 200
else:
    print("Usage: python3 timelapse_capture.py [1080p|4k]")
    exit()

# -------------------------
# AUTO FOLDER NUMBERING
# -------------------------
base_dir = "/home/royin"
i = 1

while os.path.exists(f"{base_dir}/capture_{i}"):
    i += 1

folder = f"{base_dir}/capture_{i}"
frames_dir = folder + "/frames"

os.makedirs(frames_dir, exist_ok=True)

print(f"Saving to: {folder}")
print(f"Mode: {mode} | {W}x{H} @ {FPS}fps")
print(f"Frames: {FRAMES}")

# -------------------------
# CAMERA SETUP
# -------------------------
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (W, H), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

# -------------------------
# CAMERA SETTINGS
# -------------------------
picam2.set_controls({
    "AfMode": controls.AfModeEnum.Continuous,
    "AfSpeed": controls.AfSpeedEnum.Fast,

    # image clarity
    "Sharpness": 2.2,
    "Contrast": 1.2,
    "Saturation": 1.1,

    # exposure / brightness fix
    "ExposureTime": 35000,
    "AnalogueGain": 3
})

print("Capturing frames...")

# -------------------------
# CAPTURE LOOP
# -------------------------
for i in range(FRAMES):
    frame = picam2.capture_array()

    path = f"{frames_dir}/img_{i:05d}.jpg"

    cv2.imwrite(path, frame)

    print("Saved", path)

#    time.sleep(1 / FPS)
picam2.stop()

# -------------------------
# VIDEO ENCODING (FFMPEG)
# -------------------------
print("Creating video...")

cmd = [
    "ffmpeg", "-y",
    "-r", str(FPS),
    "-i", f"{frames_dir}/img_%05d.jpg",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "ultrafast",
    "-crf", "20",
    f"{folder}/video.mp4"
]

result = subprocess.run(cmd)

if result.returncode == 0:
    print("DONE!")
    print("Folder:", folder)
    print("Video:", f"{folder}/video.mp4")
else:
    print("FFMPEG FAILED")
