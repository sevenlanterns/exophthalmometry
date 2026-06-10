from flask import Flask, render_template, Response, request, jsonify
from picamera2 import Picamera2
from libcamera import controls
import cv2
import threading
import time
import numpy as np
import subprocess
import os

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════
#  STREAM DIMENSIONS
#
#  lores  → preview only (low CPU, low bandwidth)
#  main   → recording only (high quality sensor data)
#
#  The IMX519 can do 2328×1748 @ 30fps.
#  A Pi 3B is limited in CPU, so we keep lores tiny and
#  let ffmpeg do the heavy encoding in a subprocess.
# ═══════════════════════════════════════════════════════════
LORES_W, LORES_H = 1280, 720
MAIN_W,  MAIN_H  = 2328, 1748   # full sensor; change to 1920×1080 if Pi 3B struggles

# Scale factors used to map lores ROI → main (hires) crop
SCALE_X = MAIN_W / LORES_W
SCALE_Y = MAIN_H / LORES_H

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main  ={"size": (MAIN_W,  MAIN_H),  "format": "RGB888"},
    lores ={"size": (LORES_W, LORES_H), "format": "YUV420"},
    # encode="lores" not set — we handle both streams manually
)
picam2.configure(config)
picam2.start()

# ═══════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════

# ROI is always stored in lores (preview) coordinates.
roi = {"x": 0, "y": 0, "w": LORES_W, "h": LORES_H}

recording       = False
record_path     = None          # set fresh each time recording starts
ffmpeg_process  = None

CAPTURE_BASE    = "/home/royin"  # parent directory for all capture_N folders

def next_capture_path():
    """Return /home/royin/capture_N/video.mp4 for the lowest N whose
    folder does not yet exist, creating that folder in the process."""
    n = 1
    while True:
        folder = os.path.join(CAPTURE_BASE, f"capture_{n}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return os.path.join(folder, "video.mp4")
        n += 1

# Recording quality settings (can be updated at runtime)
rec_quality = {"crf": 18, "preset": "fast"}

frame_lock          = threading.Lock()
latest_preview      = None   # lores YUV → converted to BGR for display
latest_hires        = None   # main RGB888 → used for recording crop
_capture_debug_printed = False   # one-time shape/dtype diagnostic

# Rolling FPS measurement — updated by capture_loop every frame
from collections import deque
_frame_times   = deque(maxlen=30)  # timestamps of the last 30 frames
measured_fps   = 30.0              # running estimate, initialised to 30


# ═══════════════════════════════════════════════════════════
#  CAPTURE THREAD
#  capture_arrays() grabs lores + main in a single atomic
#  call so they're always from the same moment in time.
# ═══════════════════════════════════════════════════════════
def capture_loop():
    global latest_preview, latest_hires, measured_fps
    while True:
        # Use separate capture_array calls per stream — most reliable across
        # Picamera2 versions. "main" = hires RGB888, "lores" = preview YUV420.
        hires = picam2.capture_array("main")
        lores = picam2.capture_array("lores")

        # Timestamp immediately after capture — used for FPS measurement
        now = time.monotonic()
        _frame_times.append(now)
        if len(_frame_times) >= 2:
            # FPS = (number of intervals) / (time span across them)
            measured_fps = (len(_frame_times) - 1) / (_frame_times[-1] - _frame_times[0])

        # One-time diagnostic — confirm shapes look correct before proceeding
        global _capture_debug_printed
        if not _capture_debug_printed:
            print(f"[DEBUG] hires: type={type(hires).__name__} shape={hires.shape} dtype={hires.dtype}")
            print(f"[DEBUG] lores: type={type(lores).__name__} shape={lores.shape} dtype={lores.dtype}")
            _capture_debug_printed = True

        # Ensure contiguous numpy arrays (some versions return memoryview-backed objects)
        hires = np.ascontiguousarray(hires)
        lores = np.ascontiguousarray(lores)

        # Picamera2 YUV420 is I420 planar — use COLOR_YUV2BGR_I420
        preview_bgr = cv2.cvtColor(lores, cv2.COLOR_YUV2BGR_I420)

        with frame_lock:
            latest_preview = preview_bgr
            latest_hires   = hires

capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()


# ═══════════════════════════════════════════════════════════
#  CAMERA CONTROLS
# ═══════════════════════════════════════════════════════════
def set_controls(
    exposure=None, gain=None,
    autofocus=None, focus=None,
    brightness=None, contrast=None,
    saturation=None, sharpness=None,
    awb=None, noise_reduction=None,
):
    ctrl = {}

    if exposure is not None:
        ctrl[controls.ExposureTime] = int(exposure)

    if gain is not None:
        ctrl[controls.AnalogueGain] = float(gain)

    if autofocus is not None:
        ctrl[controls.AfMode] = (
            controls.AfModeEnum.Continuous if autofocus else controls.AfModeEnum.Manual
        )

    if focus is not None:
        ctrl[controls.LensPosition] = float(focus)

    if brightness is not None:
        # libcamera: -1.0 to 1.0
        ctrl[controls.Brightness] = float(brightness)

    if contrast is not None:
        # libcamera: 0.0 to 32.0, default 1.0
        ctrl[controls.Contrast] = float(contrast)

    if saturation is not None:
        # libcamera: 0.0 to 32.0, default 1.0
        ctrl[controls.Saturation] = float(saturation)

    if sharpness is not None:
        # libcamera: 0.0 to 16.0, default 1.0
        ctrl[controls.Sharpness] = float(sharpness)

    if awb is not None:
        ctrl[controls.AwbEnable] = bool(awb)

    if noise_reduction is not None:
        # 0=Off, 1=Fast, 2=HighQuality
        mode_map = {0: controls.draft.NoiseReductionModeEnum.Off,
                    1: controls.draft.NoiseReductionModeEnum.Fast,
                    2: controls.draft.NoiseReductionModeEnum.HighQuality}
        ctrl[controls.draft.NoiseReductionMode] = mode_map.get(int(noise_reduction),
                                                                controls.draft.NoiseReductionModeEnum.Fast)

    if ctrl:
        picam2.set_controls(ctrl)


# ═══════════════════════════════════════════════════════════
#  PREVIEW STREAM
#  Reads from lores (already BGR from capture_loop).
#  Draws the ROI overlay, encodes to MJPEG.
# ═══════════════════════════════════════════════════════════
def generate():
    while True:
        with frame_lock:
            if latest_preview is None:
                time.sleep(0.01)
                continue
            frame = latest_preview.copy()
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

        # Semi-transparent darkening outside ROI
        mask = np.zeros_like(frame)
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        display = cv2.addWeighted(frame, 0.3, mask, 0.7, 0)

        # ROI border
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 229, 255), 2)

        _, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 72])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')


# ═══════════════════════════════════════════════════════════
#  RECORDING
#
#  Design:
#  • record_stop_event signals the loop to exit cleanly
#  • record_thread is joined before stdin is closed so there
#    is never a write-to-closed-file race
#  • ffmpeg stderr is captured so startup errors are visible
#  • crop dimensions are verified even before ffmpeg starts;
#    a test crop from the actual hires frame confirms the size
# ═══════════════════════════════════════════════════════════
record_stop_event = threading.Event()
record_thread     = None


def _hires_crop_dims():
    """Return (hw, hh) — the hires pixel dimensions of the current ROI,
    forced to even numbers as required by yuv420p."""
    with frame_lock:
        lw, lh = roi["w"], roi["h"]
    hw = int(round(lw * SCALE_X))
    hh = int(round(lh * SCALE_Y))
    hw = hw - (hw % 2)
    hh = hh - (hh % 2)
    return max(2, hw), max(2, hh)


def record_loop(hw, hh, fps):
    """Grab hires frames, crop to ROI, pipe raw RGB to ffmpeg stdin.
    Paces itself to fps so the recording matches the actual sensor rate."""
    global ffmpeg_process

    interval   = 1.0 / fps
    last_write = time.monotonic() - interval  # write first frame immediately

    while not record_stop_event.is_set():
        with frame_lock:
            if latest_hires is None:
                time.sleep(0.005)
                continue
            frame = latest_hires.copy()
            lx, ly, lw, lh = roi["x"], roi["y"], roi["w"], roi["h"]

        # Map lores ROI → hires coordinates, clamp, force even
        hx = min(int(round(lx * SCALE_X)), MAIN_W - 2)
        hy = min(int(round(ly * SCALE_Y)), MAIN_H - 2)

        actual_hw = min(hw, MAIN_W - hx)
        actual_hh = min(hh, MAIN_H - hy)

        cropped = frame[hy:hy + actual_hh, hx:hx + actual_hw]

        # Skip if crop is wrong size (ROI changed mid-recording)
        if cropped.shape[0] != hh or cropped.shape[1] != hw:
            time.sleep(0.005)
            continue

        if ffmpeg_process is None or ffmpeg_process.poll() is not None:
            print("[REC] ffmpeg exited early — stopping recording")
            break

        # Pace writes to match the measured sensor FPS
        now   = time.monotonic()
        delta = now - last_write
        if delta < interval:
            time.sleep(interval - delta)

        try:
            ffmpeg_process.stdin.write(cropped.tobytes())
            last_write = time.monotonic()
        except (BrokenPipeError, OSError) as e:
            print(f"[REC] pipe error: {e}")
            break


def start_record():
    global ffmpeg_process, recording, record_thread, record_path

    if recording:
        return

    hw, hh = _hires_crop_dims()
    record_path = next_capture_path()

    # Snapshot the measured FPS now; round to 2 decimal places for ffmpeg.
    # Use at least 1 fps and cap at 120 as a sanity guard.
    fps = round(max(1.0, min(120.0, measured_fps)), 2)
    print(f"[REC] starting — crop {hw}×{hh} @ {fps} fps → {record_path}")

    crf    = rec_quality.get("crf",    18)
    preset = rec_quality.get("preset", "fast")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f",       "rawvideo",
        "-pix_fmt", "rgb24",
        "-s",       f"{hw}x{hh}",
        "-r",       str(fps),
        "-i",       "pipe:0",
        "-c:v",     "libx264",
        "-preset",  preset,
        "-crf",     str(crf),
        "-pix_fmt", "yuv420p",
        record_path,
    ]

    # Keep stderr open so we see ffmpeg startup errors in the terminal
    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=None,
    )

    # Give ffmpeg a moment to start and check it didn't immediately die
    time.sleep(0.3)
    if ffmpeg_process.poll() is not None:
        print(f"[REC] ffmpeg failed to start (exit {ffmpeg_process.returncode})")
        ffmpeg_process = None
        return

    record_stop_event.clear()
    recording = True

    record_thread = threading.Thread(target=record_loop, args=(hw, hh, fps), daemon=True)
    record_thread.start()


def stop_record():
    global ffmpeg_process, recording, record_thread

    if not recording:
        return

    print("[REC] stopping…")
    recording = False
    record_stop_event.set()

    # Wait for the write loop to finish before closing the pipe
    if record_thread is not None:
        record_thread.join(timeout=3)
        record_thread = None

    if ffmpeg_process is not None:
        try:
            ffmpeg_process.stdin.flush()
            ffmpeg_process.stdin.close()
        except OSError:
            pass
        ffmpeg_process.wait()
        print(f"[REC] ffmpeg done — file: {record_path} "
              f"({'exists' if os.path.exists(record_path) else 'MISSING'})")
        ffmpeg_process = None


# ═══════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/camera_info')
def camera_info():
    """
    Returns stream dimensions so the frontend can:
      - map its crop box (in lores pixels) for display
      - show the user what hires resolution their crop will be recorded at
    """
    return jsonify({
        "lores": {"w": LORES_W, "h": LORES_H},
        "main":  {"w": MAIN_W,  "h": MAIN_H},
        "scale": {"x": SCALE_X, "y": SCALE_Y},
    })


@app.route('/fps')
def get_fps():
    """Live measured capture FPS — polled by the UI."""
    return jsonify({"fps": round(measured_fps, 1)})


@app.route('/set_roi')
def set_roi():
    """ROI coordinates are always in lores space."""
    with frame_lock:
        roi["x"] = max(0, min(int(request.args.get("x", 0)),      LORES_W - 1))
        roi["y"] = max(0, min(int(request.args.get("y", 0)),      LORES_H - 1))
        roi["w"] = max(1, min(int(request.args.get("w", LORES_W)), LORES_W - roi["x"]))
        roi["h"] = max(1, min(int(request.args.get("h", LORES_H)), LORES_H - roi["y"]))

    # Compute and return what the hires recording crop will be
    hw = int(round(roi["w"] * SCALE_X))
    hh = int(round(roi["h"] * SCALE_Y))
    return jsonify({**roi, "hires_w": hw, "hires_h": hh})


@app.route('/controls', methods=['POST'])
def controls_route():
    data = request.json
    set_controls(
        exposure       = data.get("exposure"),
        gain           = data.get("gain"),
        autofocus      = data.get("autofocus"),
        focus          = data.get("focus"),
        brightness     = data.get("brightness"),
        contrast       = data.get("contrast"),
        saturation     = data.get("saturation"),
        sharpness      = data.get("sharpness"),
        awb            = data.get("awb"),
        noise_reduction= data.get("noise_reduction"),
    )
    return jsonify({"status": "ok"})


@app.route('/rec_quality', methods=['POST'])
def set_rec_quality():
    """Update CRF and preset. Takes effect on the next recording."""
    data = request.json
    if "crf"    in data: rec_quality["crf"]    = int(data["crf"])
    if "preset" in data: rec_quality["preset"] = str(data["preset"])
    return jsonify(rec_quality)


@app.route('/record/start')
def record_start():
    if not recording:
        start_record()
    return jsonify({"recording": True})


@app.route('/record/stop')
def record_stop():
    stop_record()
    return jsonify({"recording": False})


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
