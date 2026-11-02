# Lab 4 - Tổng hợp tiếng nói (Text-to-Speech Synthesis)
**Học phần:** CSE457 - Xử lý Âm thanh và Tiếng nói  
**Trường:** Đại học Thủy lợi - Khoa Công nghệ Thông tin - Bộ môn Trí tuệ Nhân tạo  
**Sinh viên thực hiện:** Lê Văn Chương | **MSSV:** 2351260643  

---

## 1. Cấu trúc Thư mục

```text
Lab04_2351260643_LeVanChuong/
├── Lab 4.pdf                    # Đề bài thực hành chính thức của môn học
├── Lab04.py                     # Pipeline hoàn chỉnh từ A -> G, sinh toàn bộ audio và biểu đồ
├── Lab04_Notebook.ipynb         # Jupyter Notebook tương tác trực quan với audio player inline
├── hmm_tts_backend.py           # Backend tổng hợp tham số thống kê HMM (MLPG + Formant Vocoder)
├── neural_tts_backend.py        # Backend tổng hợp học sâu Neural TTS (Mel-spectrogram + Vocoder)
├── ratings.csv                  # Bảng điểm đánh giá chủ quan thính giác MOS (Listening Test)
├── report_Lab04.md              # Báo cáo thực hành chi tiết (Markdown)
├── Lab04_Report.docx            # Báo cáo thực hành chuẩn Bộ môn (Word định dạng chuyên nghiệp)
├── README.md                    # Hướng dẫn môi trường, cấu trúc và cách chạy
├── lab4_data/
│   ├── units/                   # Kho 32 đơn vị âm thanh chuẩn hóa (16 từ x 2 candidate, 16kHz mono)
│   └── metadata.csv             # Bảng thông tin metadata (label, duration, rms, speaker, filepath)
├── audio_output/                # 26 file âm thanh phát sinh từ các phương pháp và thí nghiệm
└── figures/                     # 7 hình ảnh biểu đồ phân tích kỹ thuật độ phân giải cao
```

---

## 2. Hướng dẫn Môi trường và Cách Chạy

### 2.1. Cài đặt các thư viện cần thiết:
```bash
pip install numpy scipy matplotlib soundfile librosa pandas edge-tts python-docx
```

### 2.2. Chạy toàn bộ pipeline từ đầu:
```bash
python3 Lab04.py
```
Lệnh trên sẽ tự động:
1. Đọc và tiền xử lý chuẩn hóa câu thông báo xếp hàng: `"Mời khách hàng số 23 đến quầy 5."`
2. Kiểm tra kho âm thanh 32 đơn vị và vẽ biểu đồ phân bố `fig1_inventory_stats.png`.
3. Ghép nối cơ sở (Baseline raw concatenation, xfade=0ms) sinh ra `raw_concat.wav`.
4. Chạy thuật toán quy hoạch động **Viterbi Unit Selection** tìm chuỗi candidate tối ưu; thử nghiệm các mức làm trơn biên cross-fade 0ms, 5ms, 15ms, 30ms; lưu `best_concat.wav` và vẽ `fig2_raw_vs_best_boundaries.png`, `fig3_unit_selection_grid.png`.
5. Thực nghiệm phân tích Prosody và Source-Filter: Đo chu kỳ cơ bản $F_0 = F_s / N_p$, dãn thời gian (0.8x, 1.2x) và dịch cao độ (-2, +2 semitones); vẽ `fig4_prosody_experiments.png`.
6. Thực nghiệm HMM-based synthesis với ràng buộc động MLPG và Formant Vocoder; lưu `hmm_tts.wav` và các biến thể điều khiển; vẽ `fig5_hmm_trajectories.png`.
7. Thực nghiệm Neural TTS suy luận 3 câu kiểm thử; trích xuất Mel-spectrogram, tính Latency và RTF.
8. Tổng hợp bảng điểm đánh giá chủ quan thính giác `ratings.csv`; vẽ `fig6_comparison_three_methods.png` và `fig7_listening_test_mos.png`.

### 2.3. Khởi chạy Jupyter Notebook:
Mở và chạy file [Lab04_Notebook.ipynb](file:///home/chuonglv/Documents/XLAT/TH_XLAT/Lab04_2351260643_LeVanChuong/Lab04_Notebook.ipynb) trên VS Code hoặc trình duyệt để trải nghiệm giao diện tương tác và nghe âm thanh inline.

---

## 3. Tóm tắt Kết quả Thực nghiệm

| Tiêu chí Đánh giá | Concatenative TTS | HMM Parametric TTS | Neural TTS |
| :--- | :---: | :---: | :---: |
| **Độ rõ nội dung (Intelligibility %)** | 97.83% | 90.83% | **99.50%** |
| **Độ tự nhiên (Naturalness MOS 1-5)** | 3.70 / 5.0 | 2.90 / 5.0 | **4.80 / 5.0** |
| **Tính liền mạch (Continuity MOS 1-5)**| 3.60 / 5.0 | 4.10 / 5.0 | **4.87 / 5.0** |
| **Độ trễ xử lý (Latency)** | **13.2 ms** | 39.9 ms | 325.0 ms |
| **Hệ số Real-Time Factor (RTF)** | **0.005** | 0.015 | 0.150 |
| **Dung lượng lưu trữ (Footprint)** | Trung bình (~4.8 MB) | **Siêu nhỏ (~1.2 MB)**| Lớn (>45 MB) |

### Nhận xét cốt lõi:
1. **Ghép nối (Concatenative):** Thuật toán Viterbi kết hợp Cross-fade 15ms triệt tiêu hoàn toàn tiếng click tại biên nối và giảm 24.4% tổng chi phí so với baseline cố định. Tốc độ xử lý cực nhanh (RTF < 0.01), phù hợp hoàn hảo cho các hệ thống thông báo nhúng miền đóng.
2. **Tham số HMM:** Cho phép điều khiển cao độ $F_0$ và trường độ duration liên tục, mượt mà nhờ thuật toán MLPG, chiếm bộ nhớ cực nhỏ, tuy nhiên âm sắc còn mang tính cơ giới (robotic) do bộ vocoder nguồn-lọc.
3. **Mạng nơ-ron học sâu (Neural TTS):** Cho chất lượng âm thanh vượt trội, độ tự nhiên và cảm xúc gần như không thể phân biệt với giọng người thật (MOS 4.8/5.0), là tiêu chuẩn công nghiệp hiện đại, đánh đổi lại chi phí tính toán cao hơn.
