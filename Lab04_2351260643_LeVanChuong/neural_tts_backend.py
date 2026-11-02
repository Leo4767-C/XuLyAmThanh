"""
neural_tts_backend.py
Deep Learning Neural TTS Backend (Text-to-Speech)
CSE457 - Xử lý âm thanh và tiếng nói - Lab 04
Sinh viên: Lê Văn Chương - MSSV: 2351260643
"""

import time
import io
import asyncio
import concurrent.futures
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, Dict, Any

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

try:
    import gtts
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

def _normalize_text_for_neural(text: str) -> str:
    cleaned = text.lower().replace(".", "").replace(",", "").replace(";", "").replace(":", "")
    words = []
    for tok in cleaned.split():
        if tok.isdigit():
            words.extend(DIGIT_MAP[d] for d in tok if d in DIGIT_MAP)
        else:
            words.append(tok)
    return " ".join(words)

async def _edge_tts_coro(clean_text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(clean_text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

def synthesize(
    text: str,
    voice: str = "vi-VN-HoaiMyNeural",
    sr: int = 16000
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Inference mô hình Neural TTS (học sâu)
    
    Parameters:
    - text: chuỗi văn bản đầu vào (ví dụ: 'Mời khách hàng số 23 đến quầy 5')
    - voice: tên giọng đọc pretrained
    - sr: sample rate đầu ra (mặc định 16000 Hz)
    
    Returns:
    - y_nn: waveform dạng float32 numpy array
    - sr: tần số lấy mẫu (16000)
    - extra: dictionary chứa mel-spectrogram, latency (ms), RTF, duration
    """
    t_start = time.perf_counter()
    clean_text = _normalize_text_for_neural(text)
    y = None
    
    # 1. Thử nghiệm Edge-TTS (Microsoft Neural Voice) chạy qua ThreadPool an toàn tuyệt đối
    if HAS_EDGE_TTS:
        try:
            def _run():
                return asyncio.run(_edge_tts_coro(clean_text, voice))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                audio_bytes = ex.submit(_run).result(timeout=8.0)
            
            if audio_bytes and len(audio_bytes) > 0:
                y, orig_sr = sf.read(io.BytesIO(audio_bytes))
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if orig_sr != sr:
                    y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        except Exception:
            audio_bytes = None

    # 2. Fallback Google Neural TTS (gTTS) - Chất lượng cao, cực kỳ ổn định
    if y is None and HAS_GTTS:
        try:
            tts = gtts.gTTS(text=clean_text, lang='vi', slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            y, _ = librosa.load(buf, sr=sr)
        except Exception:
            y = None

    # 3. Fallback Offline eSpeak-NG
    if y is None:
        import subprocess
        proc = subprocess.run(
            ["espeak-ng", "-v", "vi", "-w", "/tmp/temp_espeak.wav", clean_text],
            capture_output=True
        )
        if proc.returncode == 0:
            y, orig_sr = sf.read("/tmp/temp_espeak.wav")
            if orig_sr != sr:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        else:
            raise RuntimeError("Không thể tổng hợp tiếng nói từ bất kỳ backend Neural nào.")

    # Trim silence nhẹ ở 2 đầu
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y.astype(np.float32)
    
    t_end = time.perf_counter()
    latency_sec = t_end - t_start
    duration_sec = len(y) / float(sr)
    rtf = latency_sec / max(duration_sec, 1e-4) # Real-Time Factor
    
    # 2. Trích xuất Mel-spectrogram 80 mel bands
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        fmin=50,
        fmax=sr // 2
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    extra = {
        "text": text,
        "clean_text": clean_text,
        "voice": voice,
        "latency_sec": round(latency_sec, 4),
        "latency_ms": round(latency_sec * 1000.0, 2),
        "duration_sec": round(duration_sec, 4),
        "rtf": round(rtf, 4),
        "mel_spectrogram": log_mel,
        "sample_rate": sr
    }
    
    return y, sr, extra
