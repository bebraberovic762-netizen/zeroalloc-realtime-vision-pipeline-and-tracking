# APEX LEGENDS OPTIMIZED DETECTOR TRT - PRODUCTION GDI+ OPTIMIZED (CORRECTED)
import os
import sys
import ctypes
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02

def is_mouse_button_down(vk):
    return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0

import time
import psutil
from typing import List, Optional
import numpy as np
import cv2
import ctypes
from ctypes import windll, byref, c_int, c_void_p, Structure, POINTER
from ctypes.wintypes import LONG, DWORD, WORD, HDC, HWND, BOOL, UINT

# ============================================================================
# SYSTEM OPTIMIZATIONS
# ============================================================================
# Set high process priority
try:
    import psutil

    p = psutil.Process(os.getpid())
    if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
        p.nice(psutil.HIGH_PRIORITY_CLASS)
except Exception:
    pass

# Make process DPI aware for better capture performance
try:
    windll.user32.SetProcessDPIAware()
except Exception:
    pass

# ============================================================================
# CUDA SETUP (CLEANED)
# ============================================================================
import sysconfig
from pathlib import Path

sp = Path(sysconfig.get_paths()["purelib"])
venv_cuda_bins = [
    sp / "nvidia" / "cudnn" / "bin",
    sp / "nvidia" / "cublas" / "bin",
    sp / "nvidia" / "cuda_runtime" / "bin",
]

prepend = ";".join(str(d) for d in venv_cuda_bins if d.exists())

# Clean PATH setup - single line as suggested
os.environ["PATH"] = (
        r"C:\TensorRT-10.13.3.9\lib;"
        + prepend + ";"
        + os.environ.get("PATH", "")
)

for d in venv_cuda_bins:
    if d.exists():
        try:
            os.add_dll_directory(str(d))
        except Exception:
            pass

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import onnxruntime as ort


# ============================================================================
# MOUSE BACKEND
# ============================================================================
import interception


class InterceptionMouse:
    def __init__(self):
        self.enabled = True
        self.device = None
        self.context = None
        self._initialize()

    def _initialize(self):
        """Initialize interception device"""
        try:
            # Create interception context
            self.context = interception.lib.interception_create_context()

            if not self.context:
                self.enabled = False
                return

            # Find mouse device
            for device_id in range(1, 21):
                if interception.lib.interception_is_mouse(device_id):
                    self.device = device_id
                    break

            if not self.device:
                self.enabled = False
                return

            # Set minimal filter
            interception.lib.interception_set_filter(
                self.context,
                interception.lib.interception_is_mouse,
                0
            )

        except Exception:
            self.enabled = False

    def move_relative(self, dx: int, dy: int):
        """Send relative mouse movement"""
        if not self.enabled or not self.device or not self.context:
            return

        try:
            # Create mouse stroke
            stroke = interception.ffi.new("InterceptionMouseStroke *")
            stroke[0].state = interception.lib.INTERCEPTION_MOUSE_MOVE_RELATIVE
            stroke[0].flags = 0
            stroke[0].x = dx
            stroke[0].y = dy  # Positive = down, Negative = up
            stroke[0].information = 0

            # Send the stroke
            interception.lib.interception_send(self.context, self.device, stroke, 1)

        except Exception:
            pass

    def cleanup(self):
        """Cleanup resources"""
        self.enabled = False
        if self.context:
            try:
                interception.lib.interception_destroy_context(self.context)
            except:
                pass

                def next(self):
                    dx, dy = self.pattern[self.index]
                    self.index = (self.index + 1) % len(self.pattern)
                    return dx, dy
class RecoilMacro:
    def __init__(self, mouse):
        self.mouse = mouse

        # Softer recoil pattern (still effective)
        self.pattern = [
            (0, 0),
            (-0, -0),
            (0, 0),
            (0, 0),
        ]

        self.index = 0

    def update(self):
        # Only active when BOTH buttons held
        if not (is_mouse_button_down(VK_LBUTTON) and is_mouse_button_down(VK_RBUTTON)):
            self.index = 0
            return

        dx, dy = self.pattern[self.index]
        self.mouse.move_relative(dx, dy)

        self.index = (self.index + 1) % len(self.pattern)


import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import time
from dataclasses import dataclass
from typing import Optional, List, Dict
import math


@dataclass
class TargetScore:
    """Represents a target with its calculated score"""
    detection: Dict
    score: float
    screen_x: float
    screen_y: float
    distance_sq: float


class AdvancedAimController:
    def __init__(self, config, mouse_backend):
        self.mouse = mouse_backend

        # ---- USER TUNING ----
        self.sensitivity = 1.25
        self.smoothness = 0.1
        self.max_step = 18

        # ---- HEAD AIMING ----
        self.head_offset_ratio = 0.1  # % of bbox height (0.2–0.35)
        self.head_offset_pixels = 0  # fine pixel adjust (screen space)

        # ---- ADVANCED TARGET SELECTION ----
        self.fov = 500
        self.fov2 = self.fov * self.fov
        self.selection_threshold = 250  # Distance threshold for scoring

        # Target locking (like in the C# code)
        self.current_target = None
        self.current_target_id = None
        self.lock_score = 0.0
        self.max_lock_score = 100.0
        self.lock_decay_rate = 0.95
        self.lock_build_rate = 5.0

        # Size bonus parameters (from C#: area / 50000f)
        self.max_size_for_bonus = 50000.0

        # Confidence bonus (from C#: confidence * 0.3f)
        self.confidence_weight = 0.3

        # Size bonus range (from C#: 0-0.2 range)
        self.size_weight = 0.2

        # Lock bonus range (from C#: 0-0.5 range)
        self.lock_weight = 0.5

        # ---- INTERNAL STATE ----
        self.acc_x = 0.0
        self.acc_y = 0.0
        self.last_update_time = time.time()

        # Screen / model scaling
        self.screen_center_x = config.SCREEN_WIDTH // 2
        self.screen_center_y = config.SCREEN_HEIGHT // 2

        self.scale_x = config.SCREEN_WIDTH / config.MODEL_WIDTH
        self.scale_y = config.SCREEN_HEIGHT / config.MODEL_HEIGHT

        # For tracking target continuity
        self.last_target_pos = None
        self.target_velocity = (0.0, 0.0)

    def _calculate_target_score(self, detection, predicted_x, predicted_y) -> TargetScore:
        """Calculate target score using the same logic as the C# CalculateTargetScore"""
        # Convert to screen space
        screen_x = detection['x'] * self.scale_x
        screen_y = detection['y'] * self.scale_y

        # Distance to predicted position (where we expect current target to be)
        dx = screen_x - predicted_x
        dy = screen_y - predicted_y
        dist_sq = dx * dx + dy * dy

        # Normalize distance score (0 = far, 1 = close)
        # From C#: float distanceScore = Math.Max(0f, 1f - (distSq / thresholdSq))
        threshold_sq = self.selection_threshold * self.selection_threshold
        distance_score = max(0.0, 1.0 - (dist_sq / threshold_sq))

        # Confidence bonus (0-0.3 range)
        confidence = detection.get('confidence', 1.0)
        confidence_bonus = confidence * self.confidence_weight

        # Size bonus - larger targets are more stable (0-0.2 range)
        # From C#: float area = candidate.Rectangle.Width * candidate.Rectangle.Height;
        # float sizeBonus = Math.Min(0.2f, area / 50000f);
        area = detection['width'] * detection['height'] * (self.scale_x * self.scale_y)
        size_bonus = min(self.size_weight, area / self.max_size_for_bonus)

        # Lock bonus for current target (0-0.5 range based on accumulated score)
        # From C#: float lockBonus = (currentTarget != null && distanceScore > 0.3f) ? (currentLockScore / maxLockScore) * 0.5f : 0f
        lock_bonus = 0.0
        if self.current_target is not None and distance_score > 0.3:
            lock_bonus = (self.lock_score / self.max_lock_score) * self.lock_weight

        total_score = distance_score + confidence_bonus + size_bonus + lock_bonus

        return TargetScore(
            detection=detection,
            score=total_score,
            screen_x=screen_x,
            screen_y=screen_y,
            distance_sq=dist_sq
        )

    def _get_predicted_position(self) -> tuple:
        """Get predicted position for where we expect the current target to be"""
        if self.current_target is None or self.last_target_pos is None:
            return (self.screen_center_x, self.screen_center_y)

        # Simple linear prediction based on velocity
        current_time = time.time()
        dt = current_time - self.last_update_time
        dt = max(0.001, min(0.1, dt))  # Clamp like in C# code

        pred_x = self.last_target_pos[0] + self.target_velocity[0] * dt
        pred_y = self.last_target_pos[1] + self.target_velocity[1] * dt

        return (pred_x, pred_y)

    def _select_best_target(self, detections, predicted_pos) -> Optional[TargetScore]:
        """Select best target using advanced scoring instead of just closest"""
        if not detections:
            return None

        best_score = None

        for detection in detections:
            # Basic FOV check first
            dx_model = detection['x'] - (self.screen_center_x / self.scale_x)
            dy_model = detection['y'] - (self.screen_center_y / self.scale_y)
            d2 = dx_model * dx_model + dy_model * dy_model

            if d2 >= self.fov2:
                continue

            # Calculate advanced score
            score = self._calculate_target_score(detection, predicted_pos[0], predicted_pos[1])

            if best_score is None or score.score > best_score.score:
                best_score = score

        return best_score

    def _update_velocity(self, new_screen_x, new_screen_y):
        """Update target velocity for prediction"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        dt = max(0.001, min(0.1, dt))  # Clamp like in C# code

        if self.last_target_pos is not None:
            vx = (new_screen_x - self.last_target_pos[0]) / dt
            vy = (new_screen_y - self.last_target_pos[1]) / dt

            # Clamp velocity to reasonable values (like in C# MaxVelocity = 5000)
            max_vel = 5000.0
            vx = max(-max_vel, min(max_vel, vx))
            vy = max(-max_vel, min(max_vel, vy))

            self.target_velocity = (vx, vy)

        self.last_target_pos = (new_screen_x, new_screen_y)
        self.last_update_time = current_time

    def _update_lock_score(self, is_same_target: bool):
        """Update target lock score (like in the C# lock bonus system)"""
        if is_same_target:
            # Build up lock score
            self.lock_score = min(self.max_lock_score, self.lock_score + self.lock_build_rate)
        else:
            # Decay lock score
            self.lock_score *= self.lock_decay_rate
            if self.lock_score < 1.0:
                self.lock_score = 0.0

    def _calculate_head_aim_point(self, target: Dict, screen_x: float, screen_y: float) -> tuple:
        """Calculate head aim point in screen space"""
        # Model space calculation (same as before)
        head_y_model = (
                target['y']
                - target['height'] * 0.5
                + target['height'] * self.head_offset_ratio
        )

        # Safety clamp
        min_y = target['y'] - target['height'] * 0.6
        max_y = target['y'] + target['height'] * 0.2
        head_y_model = max(min_y, min(max_y, head_y_model))

        # Convert to screen space
        aim_x = screen_x
        aim_y = head_y_model * self.scale_y + self.head_offset_pixels

        return aim_x, aim_y

    def update(self, detections):
        if not detections or not self.mouse.enabled:
            self.acc_x = 0.0
            self.acc_y = 0.0
            self.current_target = None
            self.lock_score *= self.lock_decay_rate
            return

        # Get predicted position for where we expect current target to be
        predicted_pos = self._get_predicted_position()

        # Select best target using advanced scoring
        best_target = self._select_best_target(detections, predicted_pos)

        if best_target is None:
            self.current_target = None
            self.lock_score *= self.lock_decay_rate
            return

        # Check if this is the same target as before
        target_id = id(best_target.detection)  # Simple ID for tracking
        is_same_target = (self.current_target_id == target_id)

        # Update lock score
        self._update_lock_score(is_same_target)

        # Update current target tracking
        self.current_target = best_target.detection
        self.current_target_id = target_id

        # Calculate head aim point
        aim_x, aim_y = self._calculate_head_aim_point(
            best_target.detection,
            best_target.screen_x,
            best_target.screen_y
        )

        # Update velocity for next frame prediction
        self._update_velocity(aim_x, aim_y)

        # Calculate movement (same proportional control as before, but smoother)
        dx = aim_x - self.screen_center_x
        dy = aim_y - self.screen_center_y

        # Apply sensitivity and smoothness
        # Add slight reduction based on lock score for smoother tracking
        lock_smooth_factor = 1.0 - (self.lock_score / self.max_lock_score) * 0.3
        effective_smoothness = self.smoothness * lock_smooth_factor

        self.acc_x += dx * self.sensitivity * effective_smoothness
        self.acc_y += dy * self.sensitivity * effective_smoothness * 0.6

        mx = int(self.acc_x)
        my = int(self.acc_y)

        self.acc_x -= mx
        self.acc_y -= my

        mx = max(-self.max_step, min(self.max_step, mx))
        my = max(-self.max_step, min(self.max_step, my))

        if mx or my:
            self.mouse.move_relative(mx, my)

    def cleanup(self):
        self.mouse.cleanup()
# ============================================================================
# GDI+ STRUCTURES & PROPER API SIGNATURES
# ============================================================================
class BITMAPINFOHEADER(Structure):
    _fields_ = [
        ('biSize', DWORD),
        ('biWidth', LONG),
        ('biHeight', LONG),
        ('biPlanes', WORD),
        ('biBitCount', WORD),
        ('biCompression', DWORD),
        ('biSizeImage', DWORD),
        ('biXPelsPerMeter', LONG),
        ('biYPelsPerMeter', LONG),
        ('biClrUsed', DWORD),
        ('biClrImportant', DWORD)
    ]


class BITMAPINFO(Structure):
    _fields_ = [
        ('bmiHeader', BITMAPINFOHEADER),
        ('bmiColors', DWORD * 3)
    ]


# Constants
DIB_RGB_COLORS = 0
SRCCOPY = 0x00CC0020

# Load DLLs
gdi32 = windll.gdi32
user32 = windll.user32

# Set proper function signatures
user32.GetDesktopWindow.argtypes = []
user32.GetDesktopWindow.restype = HWND

user32.GetDC.argtypes = [HWND]
user32.GetDC.restype = HDC

user32.ReleaseDC.argtypes = [HWND, HDC]
user32.ReleaseDC.restype = c_int

gdi32.CreateCompatibleDC.argtypes = [HDC]
gdi32.CreateCompatibleDC.restype = HDC

gdi32.CreateDIBSection.argtypes = [HDC, POINTER(BITMAPINFO), UINT, POINTER(c_void_p), c_void_p, DWORD]
gdi32.CreateDIBSection.restype = c_void_p  # HBITMAP

gdi32.SelectObject.argtypes = [HDC, c_void_p]
gdi32.SelectObject.restype = c_void_p

gdi32.BitBlt.argtypes = [HDC, c_int, c_int, c_int, c_int, HDC, c_int, c_int, DWORD]
gdi32.BitBlt.restype = BOOL

gdi32.DeleteDC.argtypes = [HDC]
gdi32.DeleteDC.restype = BOOL

gdi32.DeleteObject.argtypes = [c_void_p]
gdi32.DeleteObject.restype = BOOL

# ============================================================================
# ROBUST GDI+ CAPTURE WITH DIB SECTION (SIMPLIFIED)
# ============================================================================
class GDIPlusCapture:
    """
    Optimized GDI+ using CreateDIBSection
    - No retry logic (BitBlt rarely fails)
    - Stable buffer creation
    """

    def __init__(self):
        self.hdesktop = None
        self.desktop_dc = None
        self.mem_dc = None
        self.bitmap = None
        self.buffer = None

        self.left = 0
        self.top = 0
        self.width = 0
        self.height = 0

        self.initialized = False
        self.bitblt_failures = 0

    def initialize(self, left: int, top: int, width: int, height: int) -> bool:
        """Initialize with CreateDIBSection"""
        try:
            self.left = left
            self.top = top
            self.width = width
            self.height = height

            # Get desktop DC
            self.hdesktop = user32.GetDesktopWindow()
            self.desktop_dc = user32.GetDC(self.hdesktop)
            if not self.desktop_dc:
                print("✗ GetDC failed")
                return False

            self.mem_dc = gdi32.CreateCompatibleDC(self.desktop_dc)
            if not self.mem_dc:
                print("✗ CreateCompatibleDC failed")
                return False

            # Setup BITMAPINFO
            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = width
            bmi.bmiHeader.biHeight = -height  # Top-down (faster)
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32  # BGRA
            bmi.bmiHeader.biCompression = 0  # BI_RGB

            # CreateDIBSection
            bits = c_void_p()
            self.bitmap = gdi32.CreateDIBSection(
                self.mem_dc,
                byref(bmi),
                DIB_RGB_COLORS,
                byref(bits),
                None,
                0
            )

            if not self.bitmap or not bits:
                print("✗ CreateDIBSection failed")
                return False

            gdi32.SelectObject(self.mem_dc, self.bitmap)

            # Create stable numpy buffer
            buffer_size = width * height * 4
            ArrayType = ctypes.c_uint8 * buffer_size
            array_ptr = ctypes.cast(bits, ctypes.POINTER(ArrayType)).contents

            # Create numpy array from ctypes array
            self.buffer = np.ctypeslib.as_array(array_ptr).reshape((height, width, 4))

            self.initialized = True
            print(f"✓ GDI+ DIBSection initialized ({width}x{height})")
            return True

        except Exception as e:
            print(f"✗ GDI+ init failed: {e}")
            self.cleanup()
            return False

    def capture(self) -> Optional[np.ndarray]:
        """Single attempt capture - simple and fast"""
        if not self.initialized:
            return None

        result = gdi32.BitBlt(
            self.mem_dc,
            0, 0, self.width, self.height,
            self.desktop_dc,
            self.left, self.top,
            SRCCOPY
        )

        if not result:
            self.bitblt_failures += 1
            if self.bitblt_failures % 100 == 0:
                print(f"⚠ BitBlt failed {self.bitblt_failures} times")
            return None

        return self.buffer

    def cleanup(self):
        """Release GDI resources"""
        try:
            if self.mem_dc:
                gdi32.DeleteDC(self.mem_dc)
                self.mem_dc = None
            if self.bitmap:
                gdi32.DeleteObject(self.bitmap)
                self.bitmap = None
            if self.desktop_dc and self.hdesktop:
                user32.ReleaseDC(self.hdesktop, self.desktop_dc)
                self.desktop_dc = None
            if self.initialized:
                print("✓ GDI+ resources released")
            self.initialized = False
        except Exception as e:
            print(f"⚠ Cleanup warning: {e}")


# ============================================================================
# CONFIGURATION - OPTIMIZED
# ============================================================================
class DetectorConfig:
    MODEL_PATH = r"C:\Users\rosen\a new start\modelr6.onnx"

    # Screen dimensions
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 960

    # OPTIMIZATION: Capture smaller region and upscale to 640x640
    CAPTURE_WIDTH = 416  # Smaller capture for faster BitBlt
    CAPTURE_HEIGHT = 416
    MODEL_WIDTH = 640  # Model expects 640x640
    MODEL_HEIGHT = 640

    # Detection parameters
    CONF_THRESH = 0.4
    IOU_THRESH = 0.45
    MAX_DETECTIONS = 5
    TOP_K_BEFORE_NMS = 200

    # Size constraints (in model coordinates 640x640)
    MIN_WIDTH, MAX_WIDTH = 20, 300
    MIN_HEIGHT, MAX_HEIGHT = 30, 400

    MODEL_EXPECTS_BGR = True
    TARGET_FPS = 240

    # Upscale interpolation method
    UPSAMPLE_METHOD = cv2.INTER_LINEAR  # Good balance of speed/quality


# ============================================================================
# OPTIMIZED DETECTOR WITH PROPER ZERO-ALLOC
# ============================================================================
class ZeroAllocDetector:
    __slots__ = ('config', 'session', 'input_name', 'output_name', 'gdi_capture',
                 'input_buffer', 'capture_buffer', 'model_buffer', 'region',
                 'is_single_class', 'valid_mask', 'nms_areas', 'nms_x1', 'nms_x2',
                 'nms_y1', 'nms_y2', 'feed_dict', 'model_n_preds', 'upscale_factor')

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.session = None
        self.input_name = None
        self.output_name = None
        self.feed_dict = None
        self.model_n_preds = None

        # Calculate upscale factor
        self.upscale_factor = config.MODEL_WIDTH / config.CAPTURE_WIDTH

        # GDI+ capture (uses smaller resolution)
        self.gdi_capture = GDIPlusCapture()

        # Pre-allocated buffers - ZERO ALLOCATION
        # 1) Model input buffer (640x640 float32 NCHW)
        self.input_buffer = np.empty((1, 3, config.MODEL_HEIGHT, config.MODEL_WIDTH), dtype=np.float32)

        # 2) Capture buffer (416x416 uint8 BGR) - for fast BitBlt output
        self.capture_buffer = np.empty((config.CAPTURE_HEIGHT, config.CAPTURE_WIDTH, 3), dtype=np.uint8)

        # 3) Model-sized buffer (640x640 uint8 BGR) - after upscaling
        self.model_buffer = np.empty((config.MODEL_HEIGHT, config.MODEL_WIDTH, 3), dtype=np.uint8)

        # NMS buffers (sized for top-K)
        max_nms = config.TOP_K_BEFORE_NMS
        self.nms_x1 = np.empty(max_nms, dtype=np.float32)
        self.nms_x2 = np.empty(max_nms, dtype=np.float32)
        self.nms_y1 = np.empty(max_nms, dtype=np.float32)
        self.nms_y2 = np.empty(max_nms, dtype=np.float32)
        self.nms_areas = np.empty(max_nms, dtype=np.float32)

        # Valid mask will be sized after model load
        self.valid_mask = None

        # Capture region (centered, using capture dimensions)
        self.region = {
            'left': (config.SCREEN_WIDTH - config.CAPTURE_WIDTH) // 2,
            'top': (config.SCREEN_HEIGHT - config.CAPTURE_HEIGHT) // 2,
            'width': config.CAPTURE_WIDTH,
            'height': config.CAPTURE_HEIGHT
        }

        self.is_single_class = None

        # Initialize GDI+ with capture dimensions
        if not self.gdi_capture.initialize(
                self.region['left'],
                self.region['top'],
                self.region['width'],
                self.region['height']
        ):
            raise RuntimeError("GDI+ initialization failed")

    def load_model(self) -> bool:
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1

            avail = ort.get_available_providers()

            # TRT engine cache directory
            trt_cache_dir = r"C:\Users\rosen\AppData\Local\ort_trt_cache"
            os.makedirs(trt_cache_dir, exist_ok=True)

            providers = []
            provider_label = []

            # 1) TensorRT first
            if "TensorrtExecutionProvider" in avail:
                providers.append(("TensorrtExecutionProvider", {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache_dir,
                }))
                provider_label.append("TRT")

            # 2) CUDA fallback
            if "CUDAExecutionProvider" in avail:
                providers.append(("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "do_copy_in_default_stream": True,
                }))
                provider_label.append("CUDA")

            # 3) CPU final fallback
            providers.append("CPUExecutionProvider")
            provider_label.append("CPU")

            provider = " -> ".join(provider_label)

            self.session = ort.InferenceSession(
                self.config.MODEL_PATH,
                sess_options=sess_options,
                providers=providers
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            output_shape = self.session.get_outputs()[0].shape

            self.is_single_class = output_shape[1] == 5

            # Allocate valid_mask based on actual model output size
            self.model_n_preds = output_shape[2]
            self.valid_mask = np.zeros(self.model_n_preds, dtype=bool)

            # Pre-build feed dict
            self.feed_dict = {self.input_name: self.input_buffer}

            print(f"✓ Model loaded ({provider}) - {'Single' if self.is_single_class else 'Multi'}-class")
            print(f"  Capture: {self.config.CAPTURE_WIDTH}x{self.config.CAPTURE_HEIGHT}")
            print(f"  Model: {self.config.MODEL_WIDTH}x{self.config.MODEL_HEIGHT}")
            print(f"  Upscale factor: {self.upscale_factor:.3f}x")
            return True

        except Exception as e:
            print(f"✗ Load failed: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """GDI+ capture with error handling"""
        return self.gdi_capture.capture()

    def preprocess_frame(self, frame_bgra: np.ndarray) -> np.ndarray:
        """
        ZERO-ALLOC preprocessing with upscaling
        1. BGRA -> BGR into capture buffer (uint8)
        2. Upscale to model size (uint8 -> uint8)
        3. Optional color conversion (uint8 -> uint8)
        4. Normalize into NCHW float32 (no intermediate float image)
        """
        # 1) BGRA -> BGR into capture buffer (uint8)
        # Using np.copyto is allocation-free
        np.copyto(self.capture_buffer, frame_bgra[:, :, :3])

        # 2) Upscale (uint8 -> uint8), fast C++ path
        cv2.resize(
            self.capture_buffer,
            (self.config.MODEL_WIDTH, self.config.MODEL_HEIGHT),
            dst=self.model_buffer,
            interpolation=self.config.UPSAMPLE_METHOD
        )

        # 3) Optional color convert (still uint8 -> uint8)
        if not self.config.MODEL_EXPECTS_BGR:
            cv2.cvtColor(self.model_buffer, cv2.COLOR_BGR2RGB, dst=self.model_buffer)

        # 4) Normalize directly into NCHW float32 (no temp float image)
        # Scale factor computed once
        scale = 1.0 / 255.0
        for c in range(3):
            # Multiply uint8 by scale directly into float32 output buffer
            # casting='unsafe' allows uint8→float32 conversion
            np.multiply(
                self.model_buffer[:, :, c],
                scale,
                out=self.input_buffer[0, c],
                casting="unsafe"
            )

        return self.input_buffer

    def parse_detections(self, output: np.ndarray) -> List[dict]:
        """Optimized parsing with upscaled coordinate adjustment"""
        out = output[0] if output.ndim == 3 else output
        preds = out.T
        n = min(len(preds), self.model_n_preds)

        x_col = preds[:n, 0]
        y_col = preds[:n, 1]
        w_col = preds[:n, 2]
        h_col = preds[:n, 3]
        conf_col = preds[:n, 4]

        # Combined filtering
        m = self.valid_mask[:n]
        np.greater(conf_col, self.config.CONF_THRESH, out=m)
        m &= (w_col >= self.config.MIN_WIDTH) & (w_col <= self.config.MAX_WIDTH) & \
             (h_col >= self.config.MIN_HEIGHT) & (h_col <= self.config.MAX_HEIGHT)

        vi = np.flatnonzero(m)
        if len(vi) == 0:
            return []

        # Top-K optimization
        vc_full = len(vi)
        if vc_full > self.config.TOP_K_BEFORE_NMS:
            conf_valid = conf_col[vi]
            topk_idx = np.argpartition(-conf_valid, self.config.TOP_K_BEFORE_NMS)[:self.config.TOP_K_BEFORE_NMS]
            vi = vi[topk_idx]
            vc = self.config.TOP_K_BEFORE_NMS
        else:
            vc = vc_full

        # Write to NMS buffers
        np.subtract(x_col[vi], w_col[vi] * 0.5, out=self.nms_x1[:vc])
        np.add(x_col[vi], w_col[vi] * 0.5, out=self.nms_x2[:vc])
        np.subtract(y_col[vi], h_col[vi] * 0.5, out=self.nms_y1[:vc])
        np.add(y_col[vi], h_col[vi] * 0.5, out=self.nms_y2[:vc])

        # Clamp to model dimensions
        np.maximum(self.nms_x1[:vc], 0, out=self.nms_x1[:vc])
        np.maximum(self.nms_y1[:vc], 0, out=self.nms_y1[:vc])
        np.minimum(self.nms_x2[:vc], self.config.MODEL_WIDTH, out=self.nms_x2[:vc])
        np.minimum(self.nms_y2[:vc], self.config.MODEL_HEIGHT, out=self.nms_y2[:vc])

        # Areas
        np.multiply(w_col[vi], h_col[vi], out=self.nms_areas[:vc])

        # Sort by confidence
        order = np.argsort(-conf_col[vi])

        # NMS
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1 or len(keep) >= self.config.MAX_DETECTIONS:
                break

            rest = order[1:]

            xx1 = np.maximum(self.nms_x1[i], self.nms_x1[rest])
            yy1 = np.maximum(self.nms_y1[i], self.nms_y1[rest])
            xx2 = np.minimum(self.nms_x2[i], self.nms_x2[rest])
            yy2 = np.minimum(self.nms_y2[i], self.nms_y2[rest])

            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            inter = w_inter * h_inter

            iou = inter / (self.nms_areas[i] + self.nms_areas[rest] - inter + 1e-7)

            order = rest[iou < self.config.IOU_THRESH]

        # Build detections
        detections = []
        for idx in keep[:self.config.MAX_DETECTIONS]:
            orig_idx = vi[idx]
            detections.append({
                'x': float(x_col[orig_idx]),
                'y': float(y_col[orig_idx]),
                'width': float(w_col[orig_idx]),
                'height': float(h_col[orig_idx]),
                'confidence': float(conf_col[orig_idx]),
                'class_id': 0,
                'index': int(orig_idx)
            })

        return detections

    def detect(self) -> dict:
        """Full optimized pipeline"""
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        frame = self.capture_frame()
        if frame is None:
            return {'detections': [], 'count': 0, 'error': 'capture_failed'}
        t_capture = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.preprocess_frame(frame)
        t_preprocess = time.perf_counter() - t0

        t0 = time.perf_counter()
        output = self.session.run([self.output_name], self.feed_dict)[0]
        t_inference = time.perf_counter() - t0

        t0 = time.perf_counter()
        detections = self.parse_detections(output)
        t_parse = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        return {
            'detections': detections,
            'count': len(detections),
            'timing': {
                'total_ms': t_total * 1000,
                'capture_ms': t_capture * 1000,
                'preprocess_ms': t_preprocess * 1000,
                'inference_ms': t_inference * 1000,
                'parse_ms': t_parse * 1000
            },
            'fps': 1.0 / t_total if t_total > 0 else 0
        }

    def cleanup(self):
        """Release resources"""
        self.gdi_capture.cleanup()


# ============================================================================
# BENCHMARK & MAIN
# ============================================================================
def run_benchmark(detector: ZeroAllocDetector, warmup: int = 10, frames: int = 100):
    print(f"\n{'=' * 60}\nBENCHMARK\n{'=' * 60}")

    print(f"Warmup: {warmup} frames...")
    for _ in range(warmup):
        detector.detect()

    print(f"Benchmark: {frames} frames...")
    timings = {'capture': [], 'preprocess': [], 'inference': [], 'parse': [], 'total': []}

    for _ in range(frames):
        result = detector.detect()
        if 'timing' in result:
            t = result['timing']
            timings['capture'].append(t['capture_ms'])
            timings['preprocess'].append(t['preprocess_ms'])
            timings['inference'].append(t['inference_ms'])
            timings['parse'].append(t['parse_ms'])
            timings['total'].append(t['total_ms'])

    print(f"\n{'=' * 60}\nRESULTS\n{'=' * 60}")

    total_mean = np.mean(timings['total'])
    fps = 1000.0 / total_mean

    print(f"\nPerformance:")
    print(f"  FPS: {fps:.1f} ({total_mean:.2f}ms/frame)")
    print(f"  Stability: ±{np.std(timings['total']):.2f}ms")

    print(f"\nBreakdown:")
    for stage in ['capture', 'preprocess', 'inference', 'parse']:
        mean = np.mean(timings[stage])
        pct = (mean / total_mean * 100)
        print(f"  {stage:12s}: {mean:5.2f}ms ({pct:4.1f}%)")

    print(f"\n{'=' * 60}\n")
    return fps


def main():
    print("APEX DETECTOR - OPTIMIZED GDI+ WITH ZERO-ALLOC UPSAMPLING")
    print("=" * 60)



    config = DetectorConfig()
    detector = ZeroAllocDetector(config)

    if not detector.load_model():
        return

    # Initialize aiming system
    mouse_backend = InterceptionMouse()
    aim_controller = AdvancedAimController(config, mouse_backend)
    recoil_macro = RecoilMacro(mouse_backend)

    # Run benchmark first
    run_benchmark(detector, warmup=10, frames=100)

    print("LIVE DETECTION (Ctrl+C to stop)\n")
    print("Metrics: Proc FPS | Wall FPS | Detections | Inference ms | Capture ms")
    print("-" * 80)

    try:
        frames = 0
        last_print = time.time()
        loop_start = time.time()

        while True:
            result = detector.detect()
            frames += 1

            recoil_macro.update()  # <-- ALWAYS RUNS

            if result['detections']:
                aim_controller.update(result['detections'])

            now = time.time()
            if now - last_print >= 1.0:
                wall_fps = frames / (now - loop_start)
                proc_fps = result['fps']

                print(f"[{frames:05d}] Proc: {proc_fps:5.1f} | Wall: {wall_fps:5.1f} | "
                      f"Det: {result['count']} | "
                      f"Inf: {result['timing']['inference_ms']:4.1f}ms | "
                      f"Cap: {result['timing']['capture_ms']:4.1f}ms")
                last_print = now

            # Simple sleep to maintain target FPS
            target_time = 1.0 / config.TARGET_FPS
            actual_time = result['timing']['total_ms'] / 1000
            sleep_time = target_time - actual_time
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 60}")
        print(f"Stopped. Total frames: {frames}")
        detector.cleanup()
        aim_controller.cleanup()


if __name__ == "__main__":
    main()
