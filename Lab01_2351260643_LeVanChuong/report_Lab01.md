# CSE457 — Báo cáo Lab 1: Phân tích và xử lý tín hiệu âm thanh số

**Tệp dữ liệu:** `voice_10s.wav` — đoạn tiếng nói, mono, 22 050 Hz, PCM 16‑bit, thời lượng 10.094 s, kích thước 445 172 byte.

Notebook đi kèm: `Lab01_MSSV.ipynb` (chạy được từ đầu đến cuối). Toàn bộ hình trong `figures/`, audio xử lý trong `audio/`.

---

## 1. Metadata (Khối A)

| Đại lượng | Giá trị |
|---|---|
| Fₛ | 22 050 Hz |
| Số kênh | 1 (mono) |
| Độ rộng mẫu | 16 bit/mẫu |
| Thời lượng | 10.094 s |
| Kích thước tệp | 445 172 byte ≈ 434.7 KB |
| Peak | 0.994 |
| RMS | 0.1106 (≈ −19.13 dBFS) |
| Số mẫu clip (\|x\|≥0.999) | 0 |

Fₛ = 22 050 Hz thoả điều kiện Nyquist cho giọng nói (phổ hữu ích chủ yếu < 8 kHz). Không phát hiện clipping.

## 2. Miền thời gian (Khối B) — xem `figures/waveform.png`

| Đoạn | Peak | RMS | Năng lượng |
|---|---|---|---|
| 1.0–2.0 s (giữa câu, nhiều âm tiết) | 0.954 | 0.115 | 292.6 |
| 0.0–0.3 s (đầu câu) | 0.627 | 0.102 | 68.5 |

Waveform có dạng "burst" điển hình của tiếng nói: các cụm biên độ lớn (âm tiết) xen kẽ khoảng gần‑lặng.

## 3. FFT (Khối C) — xem `figures/fft.png`

Đoạn phân tích: 1.0–1.5 s, cửa sổ Hamming.

- NFFT = 2048 → Δf = Fₛ/NFFT ≈ **10.77 Hz**
- NFFT = 16384 → Δf ≈ **1.35 Hz**
- Một số đỉnh phổ nổi bật (< 4 kHz): **≈ 113, 229, 350, 417, 446 Hz** — nằm trong dải F0/hoạ âm thấp của giọng nói.

NFFT lớn hơn chỉ làm bin dày hơn (nội suy mịn hơn), **không** tăng độ phân giải tần số thật sự — độ phân giải thật do độ dài cửa sổ (0.5 s) quyết định.

## 4. STFT / Spectrogram (Khối D) — xem `figures/spectrogram.png`

| Frame | Hop | Overlap |
|---|---|---|
| 10 ms (220 mẫu) | 4.0 ms (88 mẫu) | 60% |
| 25 ms (551 mẫu) | 10.0 ms (220 mẫu) | 60% |
| 50 ms (1102 mẫu) | 20.0 ms (441 mẫu) | 60% |

Frame ngắn (10 ms) → phân giải thời gian tốt, dải tần nhoè; frame dài (50 ms) → dải tần sắc nét hơn nhưng biến đổi nhanh theo thời gian (biên âm tiết) bị mờ. Đúng trade-off lý thuyết.

## 5. Cửa sổ Rectangular vs Hamming (Khối E) — xem `figures/window_comparison.png`

Cùng frame, cùng NFFT = 8192. Hamming cho nền phổ (side‑lobe) thấp và mượt hơn rõ rệt (giảm spectral leakage), đổi lại main‑lobe rộng hơn nên các đỉnh gần nhau dễ bị "dính" vào nhau hơn so với Rectangular.

## 6. Lọc số (Khối F) — xem `figures/filter_response.png`

- FIR 201 taps, cửa sổ Hamming.
- Low‑pass, fc = 3000 Hz; High‑pass, fc = 300 Hz.
- Group delay = (201−1)/2 = 100 mẫu ≈ **4.54 ms** (Fₛ = 22 050 Hz).
- Sau LPF 3 kHz: năng lượng > 3 kHz suy giảm > 40 dB — nghe "tối/muffled" hơn, mất phụ âm xát.
- Sau HPF 300 Hz: phổ gần như không đổi trong dải nghe được vì năng lượng giọng nói < 300 Hz vốn đã nhỏ.
- File âm thanh: `audio/filtered_lpf_3000.wav`, `audio/filtered_hpf_300.wav`.

## 7. Lượng tử hoá, Resampling, Mã hoá (Khối G)

### 7.1 SNR lượng tử — xem `figures/quantization_snr.png`

| B (bit) | SNR đo được (dB) |
|---|---|
| 4 | 10.69 |
| 6 | 22.61 |
| 8 | 34.71 |
| 12 | 58.85 |
| 16 | 90.99 |

SNR tăng gần tuyến tính theo B, khớp tốt với công thức SNR_Q = 6B + 4.77 − 20log₁₀(Xₘₐₓ/σₓ) khi thay σₓ = RMS thực tế (≈0.1106). Sai lệch so với "6 dB/bit" tuyệt đối do: tín hiệu không phân bố đều, RMS thấp hơn nhiều so với Xₘₐₓ = 1 (nhiều headroom), và mô hình nhiễu lượng tử đều chỉ là xấp xỉ.

### 7.2 Resampling — xem `figures/resampling_spectrum.png`

Resample về 16 kHz và 8 kHz bằng `scipy.signal.resample` (miền tần số, tự lọc chống alias). Không quan sát alias rõ trong phổ; bản 8 kHz mất hoàn toàn nội dung phổ > 4 kHz, giọng nói nghe "đục" nhưng lời vẫn nhận diện được. File: `audio/resampled_16000.wav`, `audio/resampled_8000.wav`.

### 7.3 Bit rate & kích thước — xem `figures/bitrate_comparison.png`

- PCM gốc: 22 050 × 16 × 1 = **352 800 bit/s ≈ 352.8 kbps**; kích thước tệp 10.09 s ≈ 445.1 KB.
- PCM 8 kHz/8‑bit: 64.0 kbps; kích thước ≈ 80.75 KB.
- Compression ratio (gốc / 8kHz‑8bit) ≈ **5.51 : 1**.

---

## 8. Trả lời câu hỏi báo cáo (Mục 6 đề bài)

1. **Vì sao Fₛ = 44.1 kHz chỉ biểu diễn độc lập đến 22.05 kHz?**
   Theo định lý lấy mẫu Nyquist–Shannon, để khôi phục chính xác một tín hiệu từ các mẫu rời rạc mà không bị chồng phổ (aliasing), tần số lấy mẫu phải thoả Fₛ ≥ 2Fₘₐₓ. Do đó tần số cao nhất có thể biểu diễn độc lập (không lẫn với tần số khác qua alias) là Fₛ/2 — tần số Nyquist. Với Fₛ = 44.1 kHz, Fₙyquist = 22.05 kHz; mọi thành phần trên ngưỡng này khi lấy mẫu sẽ bị "gập" (fold) trở lại phía dưới 22.05 kHz và trộn lẫn với tín hiệu gốc.

2. **NFFT tăng từ 2048 lên 8192, frame vẫn dài 25 ms: điều gì thay đổi và không đổi?**
   Thay đổi: khoảng cách giữa các bin tần số Δf = Fₛ/NFFT giảm 4 lần (phổ hiển thị mịn hơn, nội suy dày hơn — đây là *zero‑padding*). Không đổi: *độ phân giải tần số thật sự* (khả năng phân biệt hai tần số gần nhau) vẫn do độ dài cửa sổ thời gian (25 ms) quyết định, vì zero‑padding không bổ sung thêm thông tin thực sự về tín hiệu.

3. **Vì sao Hamming giảm leakage nhưng làm các đỉnh gần nhau khó phân tách hơn?**
   Cửa sổ Hamming có biên độ giảm dần về hai đầu (thay vì cắt đột ngột như Rectangular), khiến năng lượng side‑lobe trong miền tần số nhỏ hơn nhiều → giảm rò rỉ phổ (spectral leakage). Tuy nhiên, để đạt được side‑lobe thấp, main‑lobe của Hamming lại rộng hơn Rectangular; hai đỉnh tần số có khoảng cách nhỏ hơn độ rộng main‑lobe này sẽ chồng lấn và khó tách biệt trên phổ.

4. **Độ trễ FIR 201 taps tại 44.1 kHz? Có quan trọng trong xử lý thời gian thực không?**
   Group delay ≈ (L−1)/2 = 100 mẫu; ở Fₛ = 44.1 kHz, độ trễ ≈ 100/44100 ≈ **2.27 ms** (ở Fₛ = 22 050 Hz như trong bài này là ≈ 4.54 ms). Với ứng dụng nghe thông thường, độ trễ vài mili‑giây gần như không nhận biết được. Nhưng trong xử lý thời gian thực có yêu cầu độ trễ thấp (hội thoại trực tuyến, điều khiển vòng kín, đồng bộ audio‑video chặt), vài ms cộng dồn qua nhiều tầng xử lý có thể trở nên đáng kể và cần được cân nhắc hoặc bù trừ.

5. **Ảnh hưởng của B và σₓ trong SNR_Q; vì sao giảm mức tín hiệu làm SNR giảm?**
   SNR_Q(dB) = 6B + 4.77 − 20log₁₀(Xₘₐₓ/σₓ): tăng B (số bit/mẫu) làm tăng số mức lượng tử L = 2^B, giảm bước lượng tử Δ, do đó tăng SNR theo tốc độ ≈ 6 dB/bit. σₓ (RMS tín hiệu) càng lớn (gần Xₘₐₓ) thì tỉ số Xₘₐₓ/σₓ càng nhỏ, số hạng trừ đi càng nhỏ → SNR càng cao. Nếu giảm mức tín hiệu đầu vào (σₓ giảm) trong khi Xₘₐₓ (dải lượng tử) giữ nguyên, tỉ số Xₘₐₓ/σₓ tăng, số hạng −20log₁₀(...) trở nên âm hơn (giảm SNR) — vì lúc này tín hiệu chỉ dùng một phần nhỏ trong dải động của bộ lượng tử, "lãng phí" các mức lượng tử ở biên trong khi nhiễu lượng tử (phụ thuộc Δ, cố định theo Xₘₐₓ và B) không đổi.

6. **File WAV 16‑bit stereo 44.1 kHz dài 60 s: kích thước PCM lý thuyết? So với MP3 128 kbps?**
   R_PCM = 44 100 × 16 × 2 = 1 411 200 bit/s ≈ 1.4112 Mbps.
   Size ≈ R_PCM × 60 / 8 = 1 411 200 × 60 / 8 byte = 10 584 000 byte ≈ **10.09 MB**.
   MP3 128 kbps trong 60 s: 128 000 × 60/8 = 960 000 byte ≈ **0.92 MB**.
   Compression ratio ≈ 1411.2/128 ≈ **11.03 : 1**.

7. **Hai trường hợp "nghe tốt hơn" không đồng nghĩa "SNR lớn hơn":**
   - *Nén cảm thụ (perceptual coding, MP3/AAC):* các bộ mã hoá này cố tình phân bổ ít bit hơn cho những thành phần bị hiệu ứng che khuất thính giác (masking) làm cho tai gần như không nghe thấy, dẫn tới sai số toàn phần (và do đó SNR đo trên toàn phổ) có thể thấp hơn PCM tuyến tính đơn giản ở cùng bit rate, nhưng chất lượng nghe cảm nhận lại tốt hơn nhiều vì lỗi được "giấu" đúng chỗ tai không nhạy.
   - *Lọc/khử nhiễu có chủ đích:* một bộ lọc low‑pass hoặc noise‑gate có thể loại bỏ ồn nền tần số cao khiến tín hiệu output khác nhiều so với tín hiệu gốc (SNR so với bản gốc thấp) nhưng người nghe lại đánh giá bản đã lọc "sạch" và dễ chịu hơn — SNR toán học so với gốc không phản ánh chất lượng cảm thụ chủ quan.

---

## 9. Các lưu ý về phương pháp đã tuân thủ

- FFT chỉ áp dụng trên từng đoạn ngắn (0.5 s), không dùng cho kết luận theo thời điểm trên toàn bộ 10 s (dùng STFT/spectrogram cho việc đó).
- Spectrogram so sánh giữa 3 độ dài cửa sổ dùng chung thang màu (vmin=−90, vmax=−10 dB).
- So sánh cửa sổ Rectangular/Hamming giữ nguyên dữ liệu và NFFT.
- Downsampling dùng `scipy.signal.resample` (có anti‑aliasing trong miền tần số), không lấy mẫu thủ công.
- Thiết kế FIR filter truyền rõ `fs=Fs` cho `firwin`/`freqz`, tránh nhầm normalized frequency.
- Mọi giá trị dB được ghi rõ là dBFS (mức tín hiệu) hoặc SNR dB (tỉ số tín hiệu/nhiễu).
