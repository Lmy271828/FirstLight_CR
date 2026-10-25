"""Screen capture for image observations.

Bridges the offline engine's native-render mode to pixel observations.
Two backends feed the same calibrated arena crop:

- `capture_arena` / `capture_png`: one-shot `adb exec-out screencap -p`
  (~0.4-0.8 s per frame on MuMu). Simple, always works, used as fallback.
- `StreamCapture`: a persistent `adb exec-out screenrecord --output-format=h264`
  stream decoded by PyAV in a background thread. The guest encoder emits
  frames continuously (5-40 ms apart at 540x960) independent of the training
  loop, so a step only pays the ~3 ms crop/resize of the newest frame. Same
  wire path works on any adb-speaking guest (MuMu, redroid, AVD).

Geometry source of truth: `native_runner.native_overlay`
(`WORLD_PROJECTION_SOURCE_RECT`, `SOURCE_WIDTH/HEIGHT`). The engine must be
running in native-render mode with rendering ON -- `render off` blocks
`eglSwapBuffers`, so screenshots would freeze at the last presented frame.
"""

from __future__ import annotations

import io
import os
import signal
import subprocess
import threading
import time

import numpy as np
from PIL import Image

# Same rect the V4 overlay uses for the exact 18x32 logical grid projection.
# Calibrated in the 1080x1920 reference space of native_overlay.py.
ARENA_CROP = (111, 430, 967, 1620)
SOURCE_REFERENCE = (1080, 1920)


def scale_crop(crop: tuple[int, int, int, int], actual: tuple[int, int]) -> tuple[int, int, int, int]:
    """Map a calibration-space rect onto the actual screenshot resolution.

    MuMu instances may render at resolutions other than the 1080x1920
    reference (e.g. 1440x2560). Same-aspect surfaces need a uniform scale;
    different aspects are a configuration error worth raising on.
    """

    left, upper, right, lower = crop
    sx = actual[0] / SOURCE_REFERENCE[0]
    sy = actual[1] / SOURCE_REFERENCE[1]
    if abs(sx - sy) > 0.02:
        raise ValueError(f"capture aspect {actual} deviates from reference {SOURCE_REFERENCE}")
    return (
        round(left * sx),
        round(upper * sy),
        round(right * sx),
        round(lower * sy),
    )


def _kill_tree(pid: int) -> None:
    """Kill an adb client and anything it forked (the server can be a child
    holding our pipe handles, which would keep communicate() draining
    forever after a plain kill)."""

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=10,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def capture_png(adb: str, serial: str, timeout: float = 15.0) -> bytes:
    """Grab the current guest screen as PNG bytes over adb.

    adbd serializes screencaps per guest: a second concurrent capturing
    client (diagnostic probes, another env on the same VM) can stall an
    individual call for tens of seconds, so the timeout must actually
    fire. On Windows killing only the adb client is not enough -- the
    client may have forked the adb server as a child that inherited our
    stdout/stderr pipes, and communicate() would drain forever past the
    timeout; taskkill /T takes down the whole tree so EOF is guaranteed.
    """

    proc = subprocess.Popen(
        [adb, "-s", serial, "exec-out", "screencap", "-p"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        start_new_session=(os.name != "nt"),
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        try:
            proc.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
        raise TimeoutError(f"screencap timed out after {timeout:.0f}s on {serial}") from None
    if proc.returncode != 0 or not stdout:
        detail = stderr[:200] if stderr else "empty stdout"
        raise RuntimeError(f"screencap failed on {serial}: {detail}")
    return stdout


def crop_resize(
    png: bytes,
    size: tuple[int, int],
    crop: tuple[int, int, int, int] = ARENA_CROP,
) -> np.ndarray:
    """Decode PNG bytes, crop the arena rect (resolution-adaptive), resize to (H, W)."""

    import io

    probe = Image.open(io.BytesIO(png))
    actual = (probe.width, probe.height)
    probe.close()
    scaled = scale_crop(crop, actual)
    return crop_resize_rect(png, scaled, size)


def crop_resize_rect(
    png: bytes,
    rect: tuple[int, int, int, int],
    size: tuple[int, int],
) -> np.ndarray:
    image = Image.open(io.BytesIO(png)).convert("RGB").crop(rect)
    height, width = int(size[0]), int(size[1])
    image = image.resize((width, height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def capture_arena(
    adb: str,
    serial: str,
    size: tuple[int, int],
    timeout: float = 15.0,
    crop: tuple[int, int, int, int] = ARENA_CROP,
) -> np.ndarray:
    """One-shot: screencap the guest, crop the arena, downsample. uint8 (H, W, 3)."""

    return crop_resize(capture_png(adb, serial, timeout=timeout), size, crop=crop)


# ---------------------------------------------------------------------------
# Window capture path (fast: ~10-40 ms/frame, but the MuMu window must stay
# visible on a monitor -- occlusion/minimizing breaks it).
# ---------------------------------------------------------------------------

_MUMU_WINDOW_PROCESSES = ("MuMuNxMain.exe", "MuMuNxDevice.exe", "MuMuPlayer.exe", "MuMuVMMHeadless.exe")
_MUMU_WINDOW_TITLES = ("MuMu模拟器", "皇室战争模拟器", "MuMu")

SW_RESTORE = 9
SW_SHOW = 5


def _win32_user32():
    import ctypes

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass
    return user32


def _find_mumu_hwnd(process_names: tuple[str, ...] = _MUMU_WINDOW_PROCESSES) -> tuple[int, tuple[int, int, int, int]] | None:
    """Return (hwnd, client_screen_rect) of the largest MuMu window (含最小化/隐藏窗口).

    匹配进程名或窗口标题（MuMu模拟器 / 皇室战争模拟器），按窗口矩形面积取最大。
    """

    import ctypes
    from ctypes import wintypes

    user32 = _win32_user32()
    kernel32 = ctypes.windll.kernel32

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    best: tuple[int, int] | None = None  # (window_area, hwnd)

    def on_window(hwnd: int, _param: int) -> bool:
        nonlocal best
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        process = kernel32.OpenProcess(0x1000, False, process_id.value)
        if not process:
            return True
        try:
            capacity = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(capacity.value)
            if not kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(capacity)):
                return True
            basename = os.path.basename(buffer.value).lower()
        finally:
            kernel32.CloseHandle(process)
        if not any(basename == candidate.lower() for candidate in process_names):
            return True
        if not user32.IsWindowVisible(hwnd):
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        area = (rect.right - rect.left) * (rect.bottom - rect.top)
        if best is None or area > best[0]:
            best = (area, hwnd)
        return True

    callback = EnumWindowsProc(on_window)
    user32.EnumWindows(callback, 0)
    if best is None:
        return None
    return best[1]


def ensure_mumu_window_onscreen(
    process_names: tuple[str, ...] = _MUMU_WINDOW_PROCESSES,
    window_titles: tuple[str, ...] = _MUMU_WINDOW_TITLES,
    move_to: tuple[int, int] = (40, 40),
    target_size: tuple[int, int] = (720, 1280),
) -> tuple[int, tuple[int, int, int, int]]:
    """把 MuMu 主窗口恢复到屏幕上的 9:16 标准位置（最小化/被挪走都能拉回）。

    返回 (hwnd, client_screen_rect)。训练采集的窗口快路径依赖窗口在屏可见。
    """

    import ctypes
    from ctypes import wintypes

    user32 = _win32_user32()
    kernel32 = ctypes.windll.kernel32

    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    best: tuple[int, int] | None = None  # (window_area, hwnd)

    def on_window(hwnd: int, _param: int) -> bool:
        nonlocal best
        if not user32.IsWindow(hwnd):
            return True
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        process = kernel32.OpenProcess(0x1000, False, process_id.value)
        owner_name = ""
        if process:
            try:
                capacity = wintypes.DWORD(32768)
                buffer = ctypes.create_unicode_buffer(capacity.value)
                if kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(capacity)):
                    owner_name = os.path.basename(buffer.value).lower()
            finally:
                kernel32.CloseHandle(process)
        title = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, title, 512)
        title_value = title.value
        process_match = any(owner_name == candidate.lower() for candidate in process_names)
        title_match = any(t in title_value for t in window_titles)
        if not (process_match or title_match):
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        area = (rect.right - rect.left) * (rect.bottom - rect.top)
        if best is None or area > best[0]:
            best = (area, hwnd)
        return True

    callback = EnumWindowsProc(on_window)
    user32.EnumWindows(callback, 0)
    if best is None:
        raise RuntimeError("no MuMu window found to bring on-screen")
    hwnd = best[1]
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    user32.MoveWindow(hwnd, move_to[0], move_to[1], target_size[0], target_size[1], True)
    user32.SetForegroundWindow(hwnd)
    client = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(client))
    point = wintypes.POINT(client.left, client.top)
    user32.ClientToScreen(hwnd, ctypes.byref(point))
    return hwnd, (point.x, point.y, client.right - client.left, client.bottom - client.top)


def capture_window_arena(
    size: tuple[int, int],
    process_names: tuple[str, ...] = _MUMU_WINDOW_PROCESSES,
    window_titles: tuple[str, ...] = _MUMU_WINDOW_TITLES,
    crop: tuple[int, int, int, int] = ARENA_CROP,
    move_to: tuple[int, int] = (40, 40),
    target_size: tuple[int, int] = (720, 1280),
) -> tuple[np.ndarray, dict[str, int]]:
    """Grab the arena region straight from the MuMu window on screen (fast path).

   先把 MuMu 主窗口拉回屏幕（必要时还原最小化）并摆到标准 9:16 位置，
    再按标定比例从屏幕裁出 arena。返回 (uint8 (H, W, 3), metadata dict)。
    找不到窗口时抛 RuntimeError，调用方可回退 adb 路径。
    """

    import mss

    hwnd, client_rect = ensure_mumu_window_onscreen(
        process_names=process_names,
        window_titles=window_titles,
        move_to=move_to,
        target_size=target_size,
    )
    origin_x, origin_y, client_w, client_h = client_rect

    # Bottom-anchored 9:16 viewport inside the client rect (native_overlay 语义).
    target_aspect = SOURCE_REFERENCE[0] / SOURCE_REFERENCE[1]
    if client_w / client_h > target_aspect:
        view_h = client_h
        view_w = int(round(view_h * target_aspect))
        view_x = origin_x + (client_w - view_w) // 2
        view_y = origin_y
    else:
        view_w = client_w
        view_h = int(round(view_w / target_aspect))
        view_x = origin_x
        view_y = origin_y + (client_h - view_h)

    # ARENA_CROP 在 1080x1920 标定空间；视口是该空间的屏幕投影。
    sx = view_w / SOURCE_REFERENCE[0]
    sy = view_h / SOURCE_REFERENCE[1]
    left, upper, right, lower = crop
    grab_rect = {
        "left": int(view_x + left * sx),
        "top": int(view_y + upper * sy),
        "width": max(1, int((right - left) * sx)),
        "height": max(1, int((lower - upper) * sy)),
    }

    with mss.mss() as scanner:
        frame = scanner.grab(grab_rect)
    rgb = np.asarray(frame)[..., [2, 1, 0]]  # BGRA -> RGB
    image = Image.fromarray(rgb).resize((size[1], size[0]), Image.BILINEAR)
    metadata = {
        "viewport_x": view_x,
        "viewport_y": view_y,
        "viewport_w": view_w,
        "viewport_h": view_h,
        "grab_left": grab_rect["left"],
        "grab_top": grab_rect["top"],
    }
    return np.asarray(image, dtype=np.uint8), metadata


class StreamCapture:
    """Background screenrecord h264 stream serving the newest arena frame.

    The guest-side encoder runs independently of the training loop, so env
    steps never block on capture: `latest()` converts the most recent decoded
    frame (~3 ms) instead of paying a screencap round trip. screenrecord has
    a hard 180 s recording cap, so the stream is restarted in a loop; during
    the ~1-2 s respawn the last frame is served with its age so the caller
    can decide to fall back to the adb path.
    """

    def __init__(
        self,
        adb: str,
        serial: str,
        size: tuple[int, int],
        crop: tuple[int, int, int, int] = ARENA_CROP,
        stream_size: tuple[int, int] = (540, 960),
        bit_rate: int = 8_000_000,
        time_limit_s: int = 180,
    ):
        import av  # noqa: F401 -- fail fast here, not inside the thread

        self._adb = adb
        self._serial = serial
        self._size = (int(size[0]), int(size[1]))
        self._crop = crop
        self._stream_size = tuple(stream_size)
        self._bit_rate = bit_rate
        self._time_limit_s = time_limit_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = None  # latest decoded av.VideoFrame, converted lazily
        self._t_frame = 0.0
        self._stats = {"frames": 0, "restarts": 0, "decode_errors": 0}

    def _loop(self):
        while not self._stop.is_set():
            proc = None
            try:
                proc = subprocess.Popen(
                    [
                        self._adb, "-s", self._serial, "exec-out", "screenrecord",
                        "--output-format=h264",
                        "--size", f"{self._stream_size[0]}x{self._stream_size[1]}",
                        "--bit-rate", str(self._bit_rate),
                        "--time-limit", str(self._time_limit_s),
                        "-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                import av

                with av.open(proc.stdout, format="h264") as container:
                    for frame in container.decode(video=0):
                        if self._stop.is_set():
                            break
                        with self._lock:
                            self._frame = frame
                            self._t_frame = time.time()
                            self._stats["frames"] += 1
            except Exception:
                with self._lock:
                    self._stats["decode_errors"] += 1
            finally:
                if proc is not None:
                    _kill_tree(proc.pid)
            if self._stop.is_set():
                break
            with self._lock:
                self._stats["restarts"] += 1
            self._stop.wait(0.5)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def latest(self) -> tuple[np.ndarray | None, dict]:
        """Newest arena frame as uint8 (H, W, 3) plus metadata (age, stats)."""

        with self._lock:
            frame, t_frame = self._frame, self._t_frame
            stats = dict(self._stats)
        if frame is None:
            return None, {"age_s": None, **stats}
        rgb = frame.to_ndarray(format="rgb24")
        rect = scale_crop(self._crop, (frame.width, frame.height))
        left, upper, width, height = rect
        cropped = rgb[upper:upper + height, left:left + width]
        image = np.asarray(
            Image.fromarray(cropped).resize((self._size[1], self._size[0]), Image.BILINEAR),
            dtype=np.uint8,
        )
        return image, {"age_s": time.time() - t_frame, **stats}

    def stop(self):
        self._stop.set()

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)
