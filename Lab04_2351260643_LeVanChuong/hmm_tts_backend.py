"""
hmm_tts_backend.py
HMM-based Speech Synthesis Backend (Statistical Parametric TTS)
CSE457 - Xử lý âm thanh và tiếng nói - Lab 04
Sinh viên: Lê Văn Chương - MSSV: 2351260643
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, Tuple, Any

# Từ điển tham số âm học chuẩn cho các từ trong miền hẹp (đặc trưng Formants F1, F2, F3 (Hz), Bandwidth B1, B2, B3, F0 gốc, Voicing)
# Mỗi từ mô hình hóa bằng 3 state (Onset, Nucleus, Coda)
ACOUSTIC_PHONE_MODELS: Dict[str, Dict[str, Any]] = {
    "mời": {
        "f0_base": 190.0,
        "f0_slope": -20.0, # thanh huyền đi xuống
        "states": [
            {"dur": 7, "f1": 350, "f2": 950, "f3": 2400, "b1": 90, "b2": 120, "b3": 180, "gain": 0.04, "voiced": True},
            {"dur": 16, "f1": 520, "f2": 1450, "f3": 2550, "b1": 80, "b2": 110, "b3": 160, "gain": 0.12, "voiced": True},
            {"dur": 9, "f1": 420, "f2": 1200, "f3": 2450, "b1": 100, "b2": 140, "b3": 200, "gain": 0.05, "voiced": True},
        ]
    },
    "khách": {
        "f0_base": 220.0,
        "f0_slope": +35.0, # thanh sắc đi lên
        "states": [
            {"dur": 5, "f1": 400, "f2": 1500, "f3": 2600, "b1": 120, "b2": 150, "b3": 220, "gain": 0.03, "voiced": False}, # âm vô thanh 'kh'
            {"dur": 12, "f1": 750, "f2": 1650, "f3": 2700, "b1": 90, "b2": 120, "b3": 170, "gain": 0.11, "voiced": True},  # 'a'
            {"dur": 5, "f1": 350, "f2": 1900, "f3": 2800, "b1": 140, "b2": 180, "b3": 250, "gain": 0.02, "voiced": False}, # tắc vô thanh 'ch'
        ]
    },
    "hàng": {
        "f0_base": 185.0,
        "f0_slope": -25.0, # thanh huyền
        "states": [
            {"dur": 5, "f1": 450, "f2": 1350, "f3": 2500, "b1": 130, "b2": 160, "b3": 230, "gain": 0.03, "voiced": False}, # âm xát 'h'
            {"dur": 16, "f1": 780, "f2": 1300, "f3": 2600, "b1": 85, "b2": 115, "b3": 165, "gain": 0.11, "voiced": True},  # 'a'
            {"dur": 8, "f1": 380, "f2": 1100, "f3": 2400, "b1": 90, "b2": 130, "b3": 190, "gain": 0.04, "voiced": True},   # âm mũi 'ng'
        ]
    },
    "số": {
        "f0_base": 215.0,
        "f0_slope": +30.0, # thanh sắc
        "states": [
            {"dur": 6, "f1": 380, "f2": 1600, "f3": 2800, "b1": 140, "b2": 180, "b3": 250, "gain": 0.03, "voiced": False}, # âm xát 's'
            {"dur": 15, "f1": 530, "f2": 1050, "f3": 2500, "b1": 80, "b2": 110, "b3": 160, "gain": 0.12, "voiced": True},  # 'ô'
            {"dur": 6, "f1": 460, "f2": 1000, "f3": 2450, "b1": 100, "b2": 130, "b3": 180, "gain": 0.04, "voiced": True},
        ]
    },
    "không": {
        "f0_base": 195.0,
        "f0_slope": -5.0, # thanh ngang
        "states": [
            {"dur": 6, "f1": 400, "f2": 1400, "f3": 2600, "b1": 130, "b2": 160, "b3": 220, "gain": 0.03, "voiced": False},
            {"dur": 14, "f1": 510, "f2": 1000, "f3": 2450, "b1": 85, "b2": 115, "b3": 165, "gain": 0.11, "voiced": True},
            {"dur": 8, "f1": 360, "f2": 950, "f3": 2400, "b1": 95, "b2": 135, "b3": 190, "gain": 0.05, "voiced": True},
        ]
    },
    "một": {
        "f0_base": 175.0,
        "f0_slope": -30.0, # thanh nặng
        "states": [
            {"dur": 6, "f1": 350, "f2": 950, "f3": 2400, "b1": 90, "b2": 120, "b3": 180, "gain": 0.04, "voiced": True},
            {"dur": 13, "f1": 520, "f2": 1150, "f3": 2500, "b1": 85, "b2": 115, "b3": 165, "gain": 0.10, "voiced": True},
            {"dur": 5, "f1": 380, "f2": 1600, "f3": 2600, "b1": 150, "b2": 200, "b3": 260, "gain": 0.02, "voiced": False},
        ]
    },
    "hai": {
        "f0_base": 200.0,
        "f0_slope": -5.0, # thanh ngang
        "states": [
            {"dur": 5, "f1": 450, "f2": 1350, "f3": 2500, "b1": 130, "b2": 160, "b3": 230, "gain": 0.03, "voiced": False},
            {"dur": 15, "f1": 780, "f2": 1400, "f3": 2600, "b1": 85, "b2": 115, "b3": 165, "gain": 0.12, "voiced": True},
            {"dur": 10, "f1": 350, "f2": 2100, "f3": 2900, "b1": 80, "b2": 110, "b3": 160, "gain": 0.07, "voiced": True},
        ]
    },
    "ba": {
        "f0_base": 202.0,
        "f0_slope": -5.0, # thanh ngang
        "states": [
            {"dur": 5, "f1": 320, "f2": 900, "f3": 2350, "b1": 90, "b2": 120, "b3": 180, "gain": 0.05, "voiced": True},
            {"dur": 18, "f1": 790, "f2": 1350, "f3": 2600, "b1": 85, "b2": 115, "b3": 165, "gain": 0.12, "voiced": True},
            {"dur": 7, "f1": 720, "f2": 1300, "f3": 2550, "b1": 95, "b2": 125, "b3": 175, "gain": 0.05, "voiced": True},
        ]
    },
    "bốn": {
        "f0_base": 218.0,
        "f0_slope": +28.0, # thanh sắc
        "states": [
            {"dur": 5, "f1": 320, "f2": 900, "f3": 2350, "b1": 90, "b2": 120, "b3": 180, "gain": 0.05, "voiced": True},
            {"dur": 15, "f1": 530, "f2": 1050, "f3": 2500, "b1": 80, "b2": 110, "b3": 160, "gain": 0.11, "voiced": True},
            {"dur": 8, "f1": 360, "f2": 1600, "f3": 2650, "b1": 95, "b2": 130, "b3": 190, "gain": 0.05, "voiced": True},
        ]
    },
    "năm": {
        "f0_base": 198.0,
        "f0_slope": -5.0, # thanh ngang
        "states": [
            {"dur": 6, "f1": 350, "f2": 1600, "f3": 2650, "b1": 95, "b2": 130, "b3": 190, "gain": 0.05, "voiced": True},
            {"dur": 16, "f1": 760, "f2": 1400, "f3": 2600, "b1": 85, "b2": 115, "b3": 165, "gain": 0.12, "voiced": True},
            {"dur": 8, "f1": 340, "f2": 950, "f3": 2400, "b1": 90, "b2": 125, "b3": 185, "gain": 0.05, "voiced": True},
        ]
    },
    "sáu": {
        "f0_base": 218.0,
        "f0_slope": +30.0, # thanh sắc
        "states": [
            {"dur": 6, "f1": 380, "f2": 1600, "f3": 2800, "b1": 140, "b2": 180, "b3": 250, "gain": 0.03, "voiced": False},
            {"dur": 15, "f1": 780, "f2": 1350, "f3": 2600, "b1": 85, "b2": 115, "b3": 165, "gain": 0.11, "voiced": True},
            {"dur": 7, "f1": 420, "f2": 1000, "f3": 2400, "b1": 90, "b2": 125, "b3": 185, "gain": 0.05, "voiced": True},
        ]
    },
    "bảy": {
        "f0_base": 190.0,
        "f0_slope": -20.0, # thanh hỏi
        "states": [
            {"dur": 5, "f1": 320, "f2": 900, "f3": 2350, "b1": 90, "b2": 120, "b3": 180, "gain": 0.05, "voiced": True},
            {"dur": 15, "f1": 650, "f2": 1500, "f3": 2650, "b1": 85, "b2": 115, "b3": 165, "gain": 0.11, "voiced": True},
            {"dur": 8, "f1": 380, "f2": 1900, "f3": 2800, "b1": 85, "b2": 120, "b3": 175, "gain": 0.06, "voiced": True},
        ]
    },
    "tám": {
        "f0_base": 215.0,
        "f0_slope": +28.0, # thanh sắc
        "states": [
            {"dur": 5, "f1": 380, "f2": 1650, "f3": 2700, "b1": 140, "b2": 180, "b3": 250, "gain": 0.02, "voiced": False},
            {"dur": 15, "f1": 780, "f2": 1350, "f3": 2600, "b1": 85, "b2": 115, "b3": 165, "gain": 0.11, "voiced": True},
            {"dur": 7, "f1": 340, "f2": 950, "f3": 2400, "b1": 90, "b2": 125, "b3": 185, "gain": 0.05, "voiced": True},
        ]
    },
    "chín": {
        "f0_base": 222.0,
        "f0_slope": +32.0, # thanh sắc
        "states": [
            {"dur": 5, "f1": 380, "f2": 1800, "f3": 2800, "b1": 130, "b2": 170, "b3": 240, "gain": 0.03, "voiced": False},
            {"dur": 15, "f1": 350, "f2": 2150, "f3": 2950, "b1": 80, "b2": 110, "b3": 160, "gain": 0.11, "voiced": True},
            {"dur": 7, "f1": 350, "f2": 1650, "f3": 2650, "b1": 90, "b2": 130, "b3": 190, "gain": 0.05, "voiced": True},
        ]
    },
    "đến": {
        "f0_base": 220.0,
        "f0_slope": +30.0, # thanh sắc
        "states": [
            {"dur": 5, "f1": 360, "f2": 1700, "f3": 2700, "b1": 100, "b2": 140, "b3": 200, "gain": 0.04, "voiced": True},
            {"dur": 15, "f1": 500, "f2": 1850, "f3": 2750, "b1": 80, "b2": 115, "b3": 165, "gain": 0.11, "voiced": True},
            {"dur": 8, "f1": 360, "f2": 1600, "f3": 2650, "b1": 90, "b2": 130, "b3": 190, "gain": 0.05, "voiced": True},
        ]
    },
    "quầy": {
        "f0_base": 188.0,
        "f0_slope": -22.0, # thanh huyền
        "states": [
            {"dur": 6, "f1": 380, "f2": 1100, "f3": 2500, "b1": 120, "b2": 150, "b3": 220, "gain": 0.04, "voiced": False},
            {"dur": 16, "f1": 620, "f2": 1350, "f3": 2600, "b1": 85, "b2": 115, "b3": 165, "gain": 0.11, "voiced": True},
            {"dur": 9, "f1": 380, "f2": 1950, "f3": 2850, "b1": 85, "b2": 120, "b3": 175, "gain": 0.06, "voiced": True},
        ]
    }
}

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

def normalize_text_for_hmm(text: str) -> list:
    text = text.lower().replace(".", "").replace(",", "")
    out = []
    for tok in text.split():
        if tok.isdigit():
            out.extend(DIGIT_MAP[c] for c in tok if c in DIGIT_MAP)
        else:
            out.append(tok)
    return out

def mlpg_smooth_trajectory(raw_traj: np.ndarray, window_len: int = 7) -> np.ndarray:
    """
    Maximum Likelihood Parameter Generation (MLPG) approximation.
    Làm trơn quỹ đạo tham số từ các state bằng lọc Gaussian / moving window,
    mô phỏng hiệu ứng của dynamic (delta) constraint.
    """
    if len(raw_traj) < window_len:
        return raw_traj
    # Gaussian smoothing kernel
    t = np.linspace(-2.5, 2.5, window_len)
    kernel = np.exp(-0.5 * t**2)
    kernel /= kernel.sum()
    smoothed = np.convolve(raw_traj, kernel, mode='same')
    # giữ nguyên giá trị biên
    smoothed[:window_len//2] = raw_traj[:window_len//2]
    smoothed[-window_len//2:] = raw_traj[-window_len//2:]
    return smoothed

def resonator_filter(x: np.ndarray, f_res: float, bw: float, sr: int = 16000) -> np.ndarray:
    """
    Digital 2nd-order IIR Resonator cho một formant.
    H(z) = (1 - r) / (1 - 2*r*cos(theta)*z^-1 + r^2*z^-2)
    """
    if f_res <= 0 or bw <= 0:
        return x
    r = np.exp(-np.pi * bw / sr)
    theta = 2.0 * np.pi * f_res / sr
    a1 = -2.0 * r * np.cos(theta)
    a2 = r * r
    b0 = 1.0 - r # gain chuẩn hóa
    b = [b0]
    a = [1.0, a1, a2]
    return signal.lfilter(b, a, x)

def synthesize_hmm(
    text: str,
    dur_scale: float = 1.0,
    f0_scale: float = 1.0,
    sr: int = 16000,
    frame_shift_ms: float = 10.0
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Tổng hợp tiếng nói tham số HMM (HMM-based Speech Synthesis)
    - text: câu đầu vào
    - dur_scale: hệ số tỉ lệ duration (0.8x, 1.0x, 1.2x)
    - f0_scale: hệ số tỉ lệ pitch F0 (0.9x, 1.0x, 1.1x)
    - sr: sample rate (16000)
    
    Returns:
    - y: waveform âm thanh (numpy float32)
    - sr: 16000
    - tracks: dictionary chứa f0, formants (F1, F2, F3), energy, duration, states
    """
    tokens = normalize_text_for_hmm(text)
    
    # 1. State sequence & Duration generation
    state_records = []
    raw_f0 = []
    raw_f1 = []
    raw_f2 = []
    raw_f3 = []
    raw_b1 = []
    raw_b2 = []
    raw_b3 = []
    raw_gain = []
    raw_voiced = []
    
    # Pause silence ban đầu (20ms = 2 frames)
    for _ in range(2):
        raw_f0.append(0.0)
        raw_f1.append(400.0); raw_f2.append(1200.0); raw_f3.append(2500.0)
        raw_b1.append(100.0); raw_b2.append(150.0); raw_b3.append(200.0)
        raw_gain.append(0.0)
        raw_voiced.append(False)
        state_records.append("sil")

    for w_idx, tok in enumerate(tokens):
        if tok not in ACOUSTIC_PHONE_MODELS:
            tok = "hai" # fallback
        model = ACOUSTIC_PHONE_MODELS[tok]
        base_f0 = model["f0_base"] * f0_scale
        slope_f0 = model["f0_slope"]
        
        # Đếm tổng số frame dự kiến của từ
        total_word_dur = sum(int(round(s["dur"] * dur_scale)) for s in model["states"])
        word_frame_idx = 0
        
        for s_idx, s in enumerate(model["states"]):
            n_frames = max(2, int(round(s["dur"] * dur_scale)))
            state_name = f"{tok}_s{s_idx+1}"
            
            for _ in range(n_frames):
                # Tính đường F0 contour theo thời gian của từ
                t_rel = word_frame_idx / max(1, total_word_dur)
                curr_f0 = (base_f0 + slope_f0 * t_rel) if s["voiced"] else 0.0
                
                raw_f0.append(curr_f0)
                raw_f1.append(float(s["f1"]))
                raw_f2.append(float(s["f2"]))
                raw_f3.append(float(s["f3"]))
                raw_b1.append(float(s["b1"]))
                raw_b2.append(float(s["b2"]))
                raw_b3.append(float(s["b3"]))
                raw_gain.append(float(s["gain"]))
                raw_voiced.append(bool(s["voiced"]))
                state_records.append(state_name)
                word_frame_idx += 1
                
        # Giữa các từ thêm 1-2 frames short silence/transition
        for _ in range(1):
            raw_f0.append(0.0)
            raw_f1.append(400.0); raw_f2.append(1200.0); raw_f3.append(2500.0)
            raw_b1.append(100.0); raw_b2.append(150.0); raw_b3.append(200.0)
            raw_gain.append(0.0)
            raw_voiced.append(False)
            state_records.append("pau")

    # 2. Dynamic Trajectory Smoothing (MLPG approximation)
    raw_f0 = np.array(raw_f0, dtype=np.float32)
    # Lọc F0 chỉ trên vùng voiced
    voiced_mask = np.array(raw_voiced)
    smooth_f0 = raw_f0.copy()
    if np.any(voiced_mask):
        smooth_f0[voiced_mask] = mlpg_smooth_trajectory(raw_f0[voiced_mask], window_len=5)
        
    smooth_f1 = mlpg_smooth_trajectory(np.array(raw_f1, dtype=np.float32), window_len=7)
    smooth_f2 = mlpg_smooth_trajectory(np.array(raw_f2, dtype=np.float32), window_len=7)
    smooth_f3 = mlpg_smooth_trajectory(np.array(raw_f3, dtype=np.float32), window_len=7)
    smooth_b1 = mlpg_smooth_trajectory(np.array(raw_b1, dtype=np.float32), window_len=5)
    smooth_b2 = mlpg_smooth_trajectory(np.array(raw_b2, dtype=np.float32), window_len=5)
    smooth_b3 = mlpg_smooth_trajectory(np.array(raw_b3, dtype=np.float32), window_len=5)
    smooth_gain = mlpg_smooth_trajectory(np.array(raw_gain, dtype=np.float32), window_len=5)

    # 3. Source-Filter Vocoder Waveform Synthesis
    frame_len_samples = int(sr * (frame_shift_ms / 1000.0))
    total_frames = len(smooth_f0)
    total_samples = total_frames * frame_len_samples
    
    # Sinh tín hiệu nguồn kích thích (excitation e[n])
    excitation = np.zeros(total_samples, dtype=np.float32)
    phase = 0.0
    
    for f in range(total_frames):
        st = f * frame_len_samples
        en = st + frame_len_samples
        f0_val = smooth_f0[f]
        is_v = voiced_mask[f] and f0_val > 40.0
        
        for n in range(st, en):
            if is_v:
                phase += f0_val / sr
                if phase >= 1.0:
                    phase -= 1.0
                    excitation[n] = 1.0 # xung pitch glottal impulse
                else:
                    # Rosenberg glottal flow slope nhẹ
                    excitation[n] = 0.0
            else:
                # White noise cho unvoiced
                excitation[n] = np.random.uniform(-0.5, 0.5)

    # Lọc Formant Filter Cascade (H1 -> H2 -> H3)
    # Lọc theo từng đoạn nhỏ hoặc biến thiên theo frame
    synth_wave = np.zeros(total_samples, dtype=np.float32)
    
    for f in range(total_frames):
        st = f * frame_len_samples
        en = st + frame_len_samples
        frame_exc = excitation[st:en]
        
        f1_val = max(100.0, min(smooth_f1[f], sr*0.48))
        f2_val = max(300.0, min(smooth_f2[f], sr*0.48))
        f3_val = max(800.0, min(smooth_f3[f], sr*0.48))
        b1_val = max(40.0, smooth_b1[f])
        b2_val = max(50.0, smooth_b2[f])
        b3_val = max(70.0, smooth_b3[f])
        gain_val = smooth_gain[f]
        
        y1 = resonator_filter(frame_exc, f1_val, b1_val, sr=sr)
        y2 = resonator_filter(frame_exc, f2_val, b2_val, sr=sr)
        y3 = resonator_filter(frame_exc, f3_val, b3_val, sr=sr)
        
        # Parallel formant combination with weighting
        combined = 0.6 * y1 + 0.3 * y2 + 0.1 * y3
        synth_wave[st:en] = combined * gain_val

    # Normalize RMS
    rms_val = np.sqrt(np.mean(synth_wave**2))
    if rms_val > 1e-6:
        synth_wave = synth_wave * (0.08 / rms_val)
    synth_wave = np.clip(synth_wave, -0.99, 0.99).astype(np.float32)

    time_frames = np.arange(total_frames) * (frame_shift_ms / 1000.0)
    tracks = {
        "time": time_frames,
        "f0": smooth_f0,
        "raw_f0": raw_f0,
        "f1": smooth_f1,
        "f2": smooth_f2,
        "f3": smooth_f3,
        "energy": smooth_gain,
        "states": state_records,
        "tokens": tokens,
        "duration_sec": total_samples / sr
    }
    
    return synth_wave, sr, tracks
