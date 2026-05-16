# Chi tiết luồng xử lý (Phiên bản đồ án)

Tài liệu này mô tả chi tiết các bước xử lý trong dự án "XÂY DỰNG HỆ THỐNG CHUYỂN ĐỔI ẢNH BẢN NHẠC THÀNH DỮ LIỆU MUSICXML SỬ DỤNG MÔ HÌNH U-NET". Mục tiêu là cung cấp hướng dẫn rõ ràng cho việc triển khai, kiểm thử và diễn giải kết quả.

1) Tổng quan luồng xử lý

- Ảnh đầu vào → tiền xử lý → suy luận mô hình (U-Net) → chuẩn hóa hình học → trích xuất khuông và ký hiệu → suy luận tiết tấu → tạo MusicXML.

2) Điểm vào (Entry point)

- Script chính: `oemer/ete.py`
- Hàm chạy thử:
  - `main()` nhận tham số: đường dẫn ảnh, thư mục đầu ra.
  - `extract(img_path, output_dir)` điều phối toàn bộ 8 bước của pipeline.

3) Bước 1 — Tiền xử lý ảnh

- Chuyển ảnh sang thang xám, chuẩn hóa kích thước, loại nhiễu nhẹ.
- Nếu ảnh lớn, chia theo từng mảnh (patch) để suy luận hiệu quả.

4) Bước 2 — Suy luận mô hình (Inference)

- Sử dụng mô hình U-Net đã huấn luyện để phân đoạn pixel theo lớp: nền, khuông, ký hiệu, thân nốt, v.v.
- Vì ảnh lớn, áp dụng kỹ thuật cửa sổ trượt (sliding window) với ghép vùng chồng chéo và trung bình kết quả.
- Kết quả lưu dưới dạng ma trận xác suất (numpy array) vào cơ chế lưu trữ trung gian (layer registry): `staff_pred`, `symbols_pred`, `stems_pred`, …

5) Bước 3 — Chuẩn hóa hình học (Dewarp)

- Dựa trên thông tin `staff_pred` để xác định đường cong khuông.
- Tạo lưới hiệu chỉnh (remap grid) và áp dụng biến đổi để làm thẳng khuông.
- Áp dụng cùng biến đổi lên các bản đồ phân đoạn để giữ đồng bộ.

6) Bước 4 — Trích xuất dòng khuông (Staff extraction)

- Tính projection theo hàng (sum theo trục x) để tìm các đỉnh biểu thị dây khuông.
- Gom nhóm 5 đỉnh liên tiếp thành một khuông (staff), tính `unit_size` (khoảng cách giữa dây), xác định hệ số nghiêng (slope) nếu có.

7) Bước 5 — Trích xuất nốt (Notehead extraction)

- Tiền xử lý vùng ký hiệu: làm sạch nhiễu bằng thao tác hình thái học (erode/dilate).
- Tìm contour, lọc theo kích thước dựa trên `unit_size`.
- Với mỗi nốt: xác định hộp bao (bbox), staff liên quan, vị trí tương đối trên khuông để suy ra cao độ (pitch).

8) Bước 6 — Nhóm nốt và xử lý thân/beam

- Phân tích liên kết giữa nốt và thân (stem/beam) để nhóm các nốt nối chân chung.
- Số lượng chân (beam) quyết định độ dài nốt (duration) cơ bản: không chân → quarter, 1 chân → eighth, v.v.

9) Bước 7 — Trích xuất ký hiệu phụ (Clef, Accidental, Barline, Rest)

- Phát hiện vạch chia khuôn (barline) để chia measure.
- Nhận dạng khóa nhạc (clef) để xác định tham chiếu cao độ.
- Nhận dạng dấu hóa (thăng/giáng/tự nhiên) và ký hiệu nghỉ (rest) bằng bộ phân loại đơn giản.

10) Bước 8 — Suy luận tiết tấu và sinh MusicXML

- Dựa trên hình dạng nốt, beam, dot (dấu chấm) và nhóm nốt để suy ra `duration` cho từng nốt.
- Dùng `build_system.MusicXMLBuilder` để ghép các thông tin thành cấu trúc MusicXML chuẩn.

11) Lưu trữ dữ liệu trung gian (Layer Registry)

- Cơ chế: `oemer/layers.py` cung cấp `register_layer(name, data)` và `get_layer(name)` để các module trao đổi dữ liệu mà không cần truyền tham số dài.

12) Ghi chú triển khai

- Các module quan trọng: `inference.py`, `dewarp.py`, `staffline_extraction.py`, `notehead_extraction.py`, `note_group_extraction.py`, `symbol_extraction.py`, `rhythm_extraction.py`, `build_system.py`.
- Để chạy nhanh thử nghiệm, dùng ảnh mẫu trong `docs/images` và gọi:

```powershell
python -m oemer.ete docs/images/test0.png output
```

13) Kiểm thử và đánh giá

- Đo lường: IoU (Intersection over Union) cho từng lớp phân đoạn, tỉ lệ ký hiệu ánh xạ đúng sang MusicXML, tỉ lệ lỗi cao độ (pitch errors).
- Ghi lại kết quả thử nghiệm trong thư mục `output/` để làm báo cáo.

14) Gợi ý cải tiến

- Cải thiện bộ phân loại cho ký hiệu hiếm.
- Áp dụng lọc hậu để sửa lỗi ngữ cảnh (ví dụ: nếu measure thiếu duration, sửa lại bằng luật nhạc học).
- Mở rộng nhận dạng cho bản nhạc viết tay (yêu cầu dữ liệu khác).

    #       ```
    #       beam_width = khoảng rộng của chân nốt
    #       num_beams = beam_width / unit_size
    #       duration = map(num_beams) → QUARTER/8TH/16TH/...
    #       ```
    #
    #   B) Dot detection (Dấu chấm)
    #       1. Tìm chấm bên phải nốt (x > notehead_right)
    #       2. Kích thước: nhỏ (radius ~ 2-3 pixels)
    #       3. Nếu có chấm:
    #          duration_new = duration_old × 1.5
    #          VD: Dotted quarter = 1.5 beat
    #
    #   C) Beam analysis (Phân tích kết nốt)
    #       1. Đếm số chân trong note_group
    #       2. Chân càng nhiều → duration càng ngắn
    #       3. Các nốt cùng chân → cùng duration
    #       
    #       VD: 4 nốt liên tiếp với 1 chân nối
    #           ♪ ♪ ♪ ♪ (4 eighth notes)
    #           Mỗi nốt: duration = 8TH
    #
    #   D) Cập nhật notes
    #       Dùng lại `notes` từ lần lần, thêm trường duration:
    #       ```
    #       note.duration = Duration.QUARTER  # VD
    #       note.is_dotted = True/False
    #       ```
    #
    # Output:
    #   - Cập nhật mỗi NoteHead trong notes:
    #     * duration: Duration enum (WHOLE, HALF, QUARTER, 8TH, 16TH, 32ND, 64TH)
    #     * is_dotted: Boolean (có dấu chấm?)
    #   
    #   VD:
    #     Note(pitch='C4', duration=QUARTER, is_dotted=False)
    #     Note(pitch='D4', duration=EIGHTH, is_dotted=False)
    #     Note(pitch='E4', duration=QUARTER, is_dotted=True)  # 1.5 beat
    #
    # Cập nhật vào layers: 'notes' (overwrite từ bước 4)
    
    # BƯỚC 8: SINH MusicXML (BUILD SYSTEM)
    # ==================================
    # Mục đích: Ghép tất cả dữ liệu đã trích xuất thành file MusicXML
    # Tại sao cần:
    #   - MusicXML là định dạng chuẩn cho nhạc (dùng ở Finale, Sibelius...)
    #   - Từ MusicXML → có thể hiển thị, in nhạc, hay playback
    #   - Nếu không có MusicXML → kết quả không dùng được
    #
    # VD: Từ các dữ liệu:
    #     - 5 nốt (C4, D4, E4, F4, G4) với duration
    #     - 1 khóa G clef
    #     - 1 ký hiệu tạm dừng
    #     - 1 vạch bài
    #     → Sinh file .musicxml có thể mở trong Finale
    
    build_system.MusicXMLBuilder.build(
        staffs, notes, barlines, clefs, sfns, rests
    )
    # Tệp: build_system.py
    # Input: 
    #   - staffs (từ bước 3): Staff objects
    #   - notes (từ bước 7): NoteHead objects với pitch + duration
    #   - note_groups (từ bước 5): nhóm nốt
    #   - barlines (từ bước 6): vị trí vạch bài
    #   - clefs (từ bước 6): loại khóa
    #   - sfns (từ bước 6): accidentals (#, b, ♮)
    #   - rests (từ bước 6): ký hiệu tạm dừng
    #
    # Quy trình chi tiết:
    #
    #   1. Tạo Measure objects (tác nhạc)
    #       - Dùng barlines để chia bài thành các tác
    #       - Mỗi Measure có:
    #         * measure_number: 1, 2, 3...
    #         * time_signature: 4/4 (mặc định)
    #         * voices: List<Voice>
    #       
    #       VD:
    #       Measure(number=1, time=(4, 4), voices=[Voice1, Voice2])
    #
    #   2. Tạo Voice objects (nhóm nhịp)
    #       - Voice = nhóm nốt/rest trong 1 Measure với nhịp độ nhất quán
    #       - Mỗi Voice chứa một loạt note/rest
    #       - Piano có thể 2 voice (treble + bass)
    #       
    #       VD:
    #       Voice(notes=[Note(C4, QUARTER), Note(D4, QUARTER), ...])
    #       Tổng duration = 4 beat = 1 tác
    #
    #   3. Tạo Action objects (hành động)
    #       Action là class abstract, có subclass:
    #       
    #       - AddNote(pitch, duration, dots, stem_direction)
    #         VD: AddNote('C', 4, octave=4, duration=QUARTER, stem_up=True)
    #       
    #       - AddRest(duration, dots)
    #         VD: AddRest(duration=QUARTER)
    #       
    #       - AddClef(clef_type)
    #         VD: AddClef('G')  # G clef (treble)
    #       
    #       - AddAccidental(type)
    #         VD: AddAccidental('sharp')  # # ký hiệu
    #       
    #       - AddTimeSignature(numerator, denominator)
    #         VD: AddTimeSignature(4, 4)  # 4/4 time
    #       
    #       - AddKeySignature(sharps_or_flats)
    #         VD: AddKeySignature(1)  # 1 sharp = G major
    #
    #   4. Decode note (chuyển đổi note)
    #       ```python
    #       def decode_note(note: NoteHead) -> Action:
    #           # Chuyển từ note object thành Action
    #           pitch_step = note.pitch[0]      # 'C'
    #           pitch_octave = note.pitch[1]    # '4'
    #           duration_quarters = note.duration  # 1 (QUARTER)
    #           dots = 1 if note.is_dotted else 0
    #           
    #           return AddNote(
    #               step=pitch_step,
    #               octave=int(pitch_octave),
    #               duration=duration_quarters,
    #               dots=dots,
    #               stem_direction='up' if note.stem_up else 'down',
    #               accidental=note.sfn if note.sfn else None  # #, b, ♮
    #           )
    #       ```
    #
    #   5. Build MusicXML
    #       - Dùng tất cả Action → tạo XML elements
    #       - Struct:
    #         ```xml
    #         <score-partwise>
    #           <part-list>...</part-list>
    #           <part id="P1">
    #             <measure number="1">
    #               <attributes>
    #                 <clef>...</clef>
    #                 <time>...</time>
    #               </attributes>
    #               <note>
    #                 <pitch><step>C</step><octave>4</octave></pitch>
    #                 <duration>4</duration>
    #                 <type>quarter</type>
    #                 <stem>up</stem>
    #               </note>
    #               ...
    #             </measure>
    #           </part>
    #         </score-partwise>
    #         ```
    #   
    #   6. Serialize (chuyển thành bytes)
    #       - Encode XML string thành UTF-8 bytes
    #       - Thêm DOCTYPE header
    #       - Output: bytes (có thể lưu thành .musicxml file)
    #
    # Output:
    #   - musicxml_bytes: XML bytes
    #   - Kích thước: ~10-50 KB (tùy độ phức tạp bài nhạc)
    #   
    # Sau đó:
    #   - Lưu vào file: output/music.musicxml
    #   - Có thể mở trong Finale, Sibelius, MuseScore
    
    return musicxml_bytes
```

---

## 💾 **PHẦN 2: DATA FLOW (LUỒNG DỮ LIỆU)**

### **2.1 Layer Registry Pattern - Cái "Kho Chứa Trung Tâm"**

**Tệp:** `oemer/layers.py`

**Mục đích:** Các module không gọi hàm với tham số trực tiếp. Thay vào đó, chúng ghi dữ liệu vào một "kho chứa" chung, và các module khác lấy dữ liệu từ kho này.

**Tại sao cần pattern này:**
- ✅ **Loose coupling**: Module A không cần biết Module B tồn tại
- ✅ **Flexible order**: Có thể gọi module theo thứ tự bất kỳ (miễn có dữ liệu)
- ✅ **Easy debug**: In ra dữ liệu trung gian dễ dàng
- ✅ **No parameter pollution**: Tránh hàm có 10+ tham số
- ❌ **Hidden dependencies**: Khó biết module nào phụ thuộc vào cái gì
- ❌ **Global state**: Dữ liệu toàn cục, khó test và debug

**Cơ chế chi tiết:**

```python
# layers.py - Nơi lưu trữ
import numpy as np

_layers = {}  # Dictionary toàn cục

def register_layer(name: str, data):
    """Ghi dữ liệu vào kho"""
    _layers[name] = data
    print(f"✓ Registered: {name}")
    # VD: register_layer('staff_pred', array_2000x2500)
    # → _layers['staff_pred'] = array_2000x2500

def get_layer(name: str):
    """Lấy dữ liệu từ kho"""
    if name not in _layers:
        raise KeyError(f"Layer '{name}' not found!")
    return _layers[name]
    # VD: staff_pred = get_layer('staff_pred')
    # → Trả về array_2000x2500

def delete_layer(name: str):
    """Xóa dữ liệu khi không cần"""
    if name in _layers:
        del _layers[name]

def list_layers() -> list[str]:
    """Liệt kê tất cả dữ liệu"""
    return list(_layers.keys())
    # Output: ['staff_pred', 'symbols_pred', 'staffs', 'notes', ...]
```

**Cách dùng trong các module:**

```python
# ========== Module A: inference.py ==========
from oemer.layers import register_layer

def inference(model_path, img_path):
    model = load_model(model_path)
    img = cv2.imread(img_path)
    
    # Suy luận model
    pred = model.predict(img)
    
    # Lưu kết quả vào kho
    register_layer('staff_pred', pred[:,:,0])
    register_layer('symbols_pred', pred[:,:,1])
    # Xong! Không cần return gì cả

# ========== Module B: staffline_extraction.py ==========
from oemer.layers import get_layer, register_layer

def extract():
    # Lấy dữ liệu từ kho (được inference.py lưu)
    staff_pred = get_layer('staff_pred')  # ← Lấy từ kho
    
    # Xử lý
    staffs = process_staffline(staff_pred)
    
    # Lưu kết quả vào kho
    register_layer('staffs', staffs)  # ← Lưu vào kho
    return staffs

# ========== Module C: notehead_extraction.py ==========
from oemer.layers import get_layer, register_layer

def gen_notes():
    # Lấy dữ liệu từ kho
    staffs = get_layer('staffs')                    # ← Từ staffline_extraction
    symbols_pred = get_layer('symbols_pred')        # ← Từ inference
    
    # Xử lý
    notes = process_notes(staffs, symbols_pred)
    
    # Lưu vào kho
    register_layer('notes', notes)  # ← Lưu vào kho

# ========== Main execution ==========
# ete.py
inference.inference(model_path, img_path)         # Lưu staff_pred, symbols_pred
staffline_extraction.extract()                    # Lấy staff_pred → lưu staffs
notehead_extraction.gen_notes()                   # Lấy staffs, symbols_pred → lưu notes
# Không cần truyền dữ liệu trực tiếp!
```

**Ưu điểm:**
- ✅ Các module độc lập, không phụ thuộc vào nhau
- ✅ Thứ tự gọi hàm linh hoạt
- ✅ Dễ debug (có thể in ra dữ liệu ở bất kỳ điểm nào)
- ✅ Tránh tham số dài (parameter pollution)

**Nhược điểm:**
- ❌ Khó theo dõi dữ liệu từ đâu đến
- ❌ Dễ xảy ra lỗi nếu quên `register_layer`

---

### **2.2 Dữ Liệu Qua Các Bước (Input/Output)**

#### **Bước 1: Inference (Suy Luận Mô Hình)**

**Tệp:** `oemer/inference.py`

**Mục đích:** Chạy 2 mô hình U-Net để dự đoán pixel-wise

**Input:**
- Ảnh gốc (PNG): `test0.png` (2000×2500 pixels)
- Model path: `'oemer/models/seg_unet.keras'` (hoặc `.onnx`)
- Input shape của model: thường 256×256 hoặc 288×288

**Chi tiết quá trình:**

```python
def inference(model_path: str, img_path: str, step_size: int = 128, batch_size: int = 16):
    """
    Chạy model trên ảnh lớn bằng sliding window
    """
    # 1. Load model
    if model_path.endswith('.onnx'):
        model = load_onnx_model(model_path)  # ONNX runtime
    else:
        model = load_tf_model(model_path)    # TensorFlow
    
    # 2. Load ảnh
    img = cv2.imread(img_path)              # Shape: (2000, 2500, 3)
    h, w = img.shape[:2]
    
    # 3. Sliding window (vì ảnh quá lớn)
    #    Model chỉ accept 256×256, nhưng ảnh 2000×2500
    #    Giải pháp: chia ảnh thành patches, dự đoán từng patch
    
    patches = []
    for y in range(0, h, step_size):      # step_size = 128
        for x in range(0, w, step_size):
            # Cắt patch
            y2 = min(y + 256, h)
            x2 = min(x + 256, w)
            patch = img[y:y2, x:x2]       # Shape: (256, 256, 3)
            patches.append((patch, y, x))  # Lưu patch + vị trí
    
    # 4. Batch prediction
    #    Dự đoán toàn bộ patches cùng lúc (dùng batch)
    
    predictions = np.zeros((h, w, 3))     # Output array
    
    for i in range(0, len(patches), batch_size):
        batch_patches = [p[0] for p in patches[i:i+batch_size]]
        batch_preds = model.predict(batch_patches)  # Shape: (batch, 256, 256, 3)
        
        # Đặt prediction vào output array (merge patches)
        for j, (patch, y, x) in enumerate(patches[i:i+batch_size]):
            y2 = min(y + 256, h)
            x2 = min(x + 256, w)
            predictions[y:y2, x:x2] += batch_preds[j]  # Cộng lại (averaging)
    
    # 5. Normalize (vì overlapping patches cộng nhiều lần)
    predictions /= count_overlaps  # Chia lại để normalize
    
    # 6. Extract channels
    staff_pred = predictions[:, :, 1]      # Channel 1: staffline
    symbols_pred = predictions[:, :, 2]    # Channel 2: symbols
    
    return staff_pred, symbols_pred
```

**Output (lưu vào layer registry):**
```python
register_layer('staff_pred', prediction[:,:,1])      # Shape: (2000, 2500)
register_layer('symbols_pred', prediction[:,:,2])    # Shape: (2000, 2500)
register_layer('stems_rests_pred', prediction2[:,:,1])  # Chi tiết
register_layer('clefs_keys_pred', prediction2[:,:,2])   # Chi tiết
```

**Format dữ liệu:**
- Numpy array (float32)
- Shape: `(height, width)` VD: (2000, 2500)
- Value: 0.0-1.0 (xác suất) hoặc 0-255 (sau scale)
- Ý nghĩa:
  - staff_pred[200, 300] = 0.9 → 90% chắc có staffline ở vị trí (200, 300)
  - staff_pred[400, 500] = 0.1 → 10% chắc có staffline ở vị trí (400, 500)

---

#### **Bước 2: Dewarp (Chuẩn Hóa Hình Học)**

**Tệp:** `oemer/dewarp.py`

**Input:**
```python
staff_pred = get_layer('staff_pred')  # Từ bước 1
img = cv2.imread(img_path)            # Ảnh gốc
```

**Quá trình:**
```
1. Quét theo chiều dọc để tìm vị trí khuông nhạc
2. Xây dựng grid các điểm chuẩn chỉnh
3. Tính toán biến dạng từ đường cong thành thẳng
4. Áp dụng cv2.remap() để uốn thẳng ảnh
5. Cũng uốn thẳng tất cả prediction mask
```

**Output:**
```python
register_layer('dewarped_img', warped_image)
register_layer('dewarped_staff_pred', warped_staff_pred)
register_layer('dewarped_symbols_pred', warped_symbols_pred)
```

---

#### **Bước 3: Staffline Extraction (Trích Khuông)**

**Tệp:** `oemer/staffline_extraction.py`

**Input:**
```python
staff_pred = get_layer('staff_pred')  # Dự đoán khuông
# Shape: (height, width)
# Value: 0-255 (xác suất)
```

**Quá trình:**
```
1. Tính tổng pixel theo hàng (projection)
   hist = sum(staff_pred per row)
   
2. Tìm peak (đỉnh) → vị trí các đường khuông
   peaks = find_peaks(hist)
   
3. Nhóm peaks thành 5 nhóm (5 dây khuông)
   
4. Tính unit_size = khoảng cách / 4 (khoảng 1 dây)
   ⭐ unit_size là CHỈ SỐ QUAN TRỌNG!
   
5. Tạo Staff object cho mỗi nhóm
   Staff:
     - line1, line2, line3, line4, line5 (y-coordinate)
     - unit_size (khoảng cách giữa dây)
     - slope (độ nghiêng)
     - track (số khuông từ trên xuống)
```

**Output:**
```python
staffs: List[Staff]
# Mỗi Staff có:
# - center: y-coordinate giữa khuông
# - unit_size: 14 pixels (VD)
# - slope: 0.05 độ (độ nghiêng)
# - track: 0, 1, 2... (số khuông)

register_layer('staffs', staffs)
```

**unit_size là gì?**
- Trong bản nhạc thực tế, khoảng cách giữa 2 dây khuông luôn bằng nhau
- unit_size = khoảng cách đó / 2 (vì có 4 khoảng giữa 5 dây)
- VD: Nếu 5 dây cách đều 56 pixels, unit_size = 14 pixels
- Dùng để:
  - Xác định kích thước nốt dự kiến
  - Tính pitch (cao độ) của nốt
  - Filter các bounding box không hợp lệ

---

#### **Bước 4: Notehead Extraction (Trích Nốt)**

**Tệp:** `oemer/notehead_extraction.py`

**Input:**
```python
staffs = get_layer('staffs')                      # Từ bước 3
notehead_pred = get_layer('symbols_pred')        # Từ bước 1
stem_pred = get_layer('stems_rests_pred')        # Chi tiết thân nốt
unit_size = staffs[0].unit_size
```

**Quá trình:**
```
1. Morphology (xử lý hình ảnh)
   - Erosion, dilation để làm rõ các nốt
   
2. Find contours
   - Tìm tất cả contour từ notehead_pred
   
3. Filter theo kích thước
   - Lọc contour có kích thước
     min_size = unit_size × 0.5
     max_size = unit_size × 2.5
   
4. Gán nốt vào khuông
   - Với mỗi contour:
     - Tính bbox (bounding box)
     - Xác định nốt thuộc khuông nào (find_closest_staffs)
     - Tính y-position trên khuông
     
5. Gán pitch (cao độ)
   - Dùng y-position + khuông để tính pitch
   - Pitch được lưu dưới dạng note name: C4, D4, E4...
   - VD: Nốt nằm trên dây 3 của khuông G → E4
   
6. Parse stem direction
   - Dùng stem_pred để xác định:
     - Có thân nốt không
     - Thân nốt hướng lên/xuống
```

**Output:**
```python
class NoteHead:
    bbox: (x, y, w, h)           # Bounding box
    pitch: str                   # C4, D4, E4, F4...
    staff: int                   # Khuông mấy
    track: int                   # Track (cho piano)
    group_id: int                # ID nhóm (cùng thân)
    has_dot: bool                # Có dấu chấm?
    stem_up: bool                # Thân lên hay xuống?
    staff_line_pos: int          # Vị trí trên dây khuông

notes: List[NoteHead]
register_layer('notes', notes)
```

**Pitch assignment logic:**
```
Khuông G (G clef):
  Dây 1 (top)    → F5
  Dây 2         → D5
  Dây 3 (giữa)  → B4
  Dây 4         → G4
  Dây 5 (bottom)→ E4
  
Khoảng giữa các dây:
  Giữa dây 1-2 → E5
  Giữa dây 2-3 → C5
  ...

Khuông F (F clef): tương tự nhưng dịch xuống
```

---

#### **Bước 5: Note Grouping (Nhóm Nốt)**

**Tệp:** `oemer/note_group_extraction.py`

**Input:**
```python
notes = get_layer('notes')           # Từ bước 4
stem_pred = get_layer('stems_rests_pred')
```

**Quá trình:**
```
1. Connected component analysis
   - Nối các nốt có thân chung
   
2. Xác định hướng thân
   - Dựa vào vị trí nốt → stem up/down
   
3. Tạo NoteGroup
   class NoteGroup:
       notes: List[NoteHead]        # Các nốt trong nhóm
       stem_up: bool                # Thân lên
       has_stem: bool               # Có thân
       all_same_type: bool          # Cùng loại nốt?
```

**Output:**
```python
note_groups: List[NoteGroup]
register_layer('note_groups', note_groups)
```

---

#### **Bước 6: Symbol Extraction (Trích Ký Hiệu)**

**Tệp:** `oemer/symbol_extraction.py`

**Input:**
```python
symbols_pred = get_layer('symbols_pred')
stems_rests_pred = get_layer('stems_rests_pred')
clefs_keys_pred = get_layer('clefs_keys_pred')
staffs = get_layer('staffs')
notes = get_layer('notes')
```

**Quá trình:**

**A) Barlines (Vạch bài)**
```
1. Line detection từ stems_rests_pred
2. Filter:
   - Chiều dài > staffline height
   - Góc gần thẳng đứng
3. Lưu bounding box của mỗi barline
```

**B) Clefs (Khóa nhạc)**
```
1. Segment symbols_pred thành các box
2. Dùng clef.model (sklearn) để phân loại
   - G clef (Treble)
   - F clef (Bass)
   - C clef (Alto)
```

**C) Accidentals (Dấu thăng/bé/tự nhiên)**
```
1. Tìm các ký hiệu nhỏ trước nốt
2. Dùng sfn.model (sklearn) để phân loại
   - Sharp (#)
   - Flat (b)
   - Natural (♮)
```

**D) Rests (Ký hiệu tạm dừng)**
```
1. Tìm các ký hiệu không phải nốt
2. Dùng rests.model để phân loại
   - Whole rest
   - Half rest
   - Quarter rest
   - 8th rest
   - ...
```

**Output:**
```python
barlines: List[Barline]              # Vị trí vạch bài
clefs: List[Clef]                    # Khóa nhạc
sfns: List[Accidental]               # Dấu thăng/bé
rests: List[Rest]                    # Ký hiệu tạm dừng

register_layer('barlines', barlines)
register_layer('clefs', clefs)
register_layer('sfns', sfns)
register_layer('rests', rests)
```

---

#### **Bước 7: Rhythm Extraction (Trích Tiết Tấu)**

**Tệp:** `oemer/rhythm_extraction.py`

**Input:**
```python
notes = get_layer('notes')
note_groups = get_layer('note_groups')
staffs = get_layer('staffs')
stems_rests_pred = get_layer('stems_rests_pred')
```

**Quá trình:**

**A) Duration (Độ dài nốt)**
```
Dựa vào hình dạng nốt:
- Nốt nhỏ, tròn → whole note (4 beat)
- Nốt trắng, tròn → half note (2 beat)
- Nốt đen, tròn → quarter note (1 beat)
- Nốt đen + 1 chân → 8th note (0.5 beat)
- Nốt đen + 2 chân → 16th note (0.25 beat)
- ...

Dùng beam width từ stem_pred để xác định loại nốt
```

**B) Dot (Dấu chấm)**
```
Tìm chấm bên phải nốt
Nếu có: duration × 1.5
VD: dotted quarter = 1.5 beat
```

**C) Beam analysis**
```
Nối các nốt có cùng "chân nối" (beam)
- 1 chân → 8th notes
- 2 chân → 16th notes
- 3 chân → 32nd notes
- ...

Xác định nhóm nốt được nối chung
```

**Output:**
```python
# Cập nhật mỗi NoteHead:
note.duration = Duration.QUARTER  # VD
note.is_dotted = True/False

register_layer('notes', notes)  # Cập nhật
```

---

#### **Bước 8: MusicXML Generation (Sinh MusicXML)**

**Tệp:** `oemer/build_system.py`

**Input:**
```python
staffs = get_layer('staffs')
notes = get_layer('notes')
note_groups = get_layer('note_groups')
barlines = get_layer('barlines')
clefs = get_layer('clefs')
sfns = get_layer('sfns')
rests = get_layer('rests')
```

**Quá trình:**

```
1. Tạo Measure objects
   Measure = đơn vị tác nhạc (thường 4/4)
   
2. Tạo Voice objects
   Voice = nhóm nốt trong một Measure
   
3. Tạo Action objects
   - AddNote(pitch, duration, ...)
   - AddRest(duration)
   - AddClef(type)
   - AddKey(sharps/flats)
   - AddTimeSignature(4, 4)
   - AddBarline()
   
4. Serialize thành XML
   Định dạng MusicXML 3.1
```

**Output:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise>
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <clef>
          <sign>G</sign>
          <line>2</line>
        </clef>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
      </attributes>
      <note>
        <pitch>
          <step>C</step>
          <octave>4</octave>
        </pitch>
        <duration>4</duration>
        <type>quarter</type>
      </note>
      ...
    </measure>
  </part>
</score-partwise>
```

---

## 🏗️ **PHẦN 3: ARCHITECTURAL PATTERN**

### **3.1 Layer Registry Pattern Chi Tiết**

**Diagram:**

```
┌─────────────────────────────────────────┐
│       LAYER REGISTRY (layers.py)        │
├─────────────────────────────────────────┤
│  staff_pred          [numpy array]      │
│  symbols_pred        [numpy array]      │
│  stems_rests_pred    [numpy array]      │
│  staffs              [List<Staff>]      │
│  notes               [List<NoteHead>]   │
│  note_groups         [List<NoteGroup>]  │
│  barlines            [List<Barline>]    │
│  clefs               [List<Clef>]       │
│  sfns                [List<Accidental>] │
│  rests               [List<Rest>]       │
└─────────────────────────────────────────┘
         ↑              ↑              ↑
         │              │              │
    [write]         [read]         [delete]
         │              │              │
    ┌────┴─────────────┬┴──────────────┴─────┐
    │                  │                      │
    │                  │                      │
inference.py    staffline_extraction.py   rhythm_extraction.py
  [register]         [get/register]           [get/register]
   
dewarp.py       notehead_extraction.py   build_system.py
 [get/register]   [get/register]          [get]
```

**Cách dùng:**

```python
# ✅ Module A lưu dữ liệu
from oemer.layers import register_layer
def process_a():
    result = compute_something()
    register_layer('my_data', result)

# ✅ Module B lấy dữ liệu
from oemer.layers import get_layer
def process_b():
    data = get_layer('my_data')
    return process(data)

# ✅ Không cần truyền tham số trực tiếp!
# process_a()
# process_b()  # tự động lấy từ layer registry
```

---

### **3.2 Các Điểm Kết Nối (Touchpoints) Giữa Modules**

#### **Inference ↔ Dewarp**
```
Inference output:
  - staff_pred (mask khuông)
  - symbols_pred (mask ký hiệu)
  - stems_rests_pred (mask chi tiết)
  
Dewarp input:
  - staff_pred (để tìm vị trí cắt)
  - ảnh gốc (để uốn thẳng)
  
Dewarp output:
  - dewarped_img (ảnh thẳng)
  - dewarped_staff_pred (mask khuông thẳng)
```

#### **Dewarp ↔ Staffline Extraction**
```
Dewarp output:
  - dewarped_staff_pred
  
Staffline input:
  - staff_pred (từ inference hoặc dewarp)
  
Staffline output:
  - staffs (List<Staff> với unit_size)
```

#### **Staffline ↔ Notehead Extraction**
```
Staffline output:
  - staffs (cung cấp unit_size, track info)
  
Notehead input:
  - staffs (để gán nốt vào khuông)
  - symbols_pred (để tìm nốt)
  - unit_size (để filter kích thước)
  
Notehead output:
  - notes (List<NoteHead> với pitch)
```

#### **Notehead ↔ Symbol Extraction**
```
Notehead output:
  - notes (danh sách vị trí nốt)
  
Symbol input:
  - symbols_pred (để tìm các ký hiệu khác)
  - stems_rests_pred (để tìm vạch bài, ký hiệu)
  - notes (để phân biệt nốt vs ký hiệu)
  
Symbol output:
  - barlines, clefs, sfns, rests
```

#### **Notes + Note Groups ↔ Rhythm Extraction**
```
Input:
  - notes (danh sách với pitch, position)
  - note_groups (nhóm có cùng thân)
  - stems_rests_pred (để phân tích beam)
  
Output:
  - notes (cập nhật thêm duration info)
```

#### **All Data ↔ Build System**
```
Input:
  - staffs (track info)
  - notes (pitch + duration)
  - note_groups (grouping info)
  - barlines (vị trí vạch)
  - clefs (loại khóa)
  - sfns (accidentals)
  - rests (tạm dừng)
  
Output:
  - MusicXML bytes
```

---

### **3.3 File Thứ Tự Tải (Dependency Graph)**

```
layers.py
  ↑
  │ (import)
  │
  ├── inference.py
  │   └── models/unet.py
  │
  ├── dewarp.py
  │   └── layers.py
  │
  ├── staffline_extraction.py
  │   └── layers.py
  │
  ├── notehead_extraction.py
  │   ├── layers.py
  │   ├── staffline_extraction.py
  │   └── constant.py (unit_size config)
  │
  ├── note_group_extraction.py
  │   └── layers.py
  │
  ├── symbol_extraction.py
  │   ├── layers.py
  │   ├── classifier.py (clef.model, sfn.model...)
  │   └── sklearn_models/
  │
  ├── rhythm_extraction.py
  │   └── layers.py
  │
  ├── build_system.py
  │   └── layers.py
  │
  └── ete.py (main orchestrator)
      └── tất cả các module trên
```

---

## 📐 **PHẦN 4: DATA STRUCTURE CHI TIẾT**

### **4.1 Nhóm 1: Prediction Masks (Từ Inference)**

```python
staff_pred: np.ndarray
  Shape: (height, width)
  Value: 0-255 (xác suất có staffline)
  VD: height=2000, width=2500
  
symbols_pred: np.ndarray
  Shape: (height, width)
  Value: 0-255 (xác suất có ký hiệu)

stems_rests_pred: np.ndarray
  Shape: (height, width)
  Value: 0-255 (chi tiết thân nốt, vạch bài)
  
clefs_keys_pred: np.ndarray
  Shape: (height, width)
  Value: 0-255 (chi tiết khóa nhạc)
```

### **4.2 Nhóm 2: Geometric Structures (Từ Extraction)**

```python
class Staff:
    center: float              # y-coordinate giữa
    unit_size: float           # khoảng cách/2
    slope: float               # độ nghiêng (độ)
    track: int                 # 0, 1, 2... (từ trên)
    group: int                 # nhóm khuông (piano)
    
    # Thêm:
    line1, line2, line3, line4, line5: float  # y-coords của 5 dây

class NoteHead:
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    pitch: str                        # "C4", "D4", "E4"...
    staff: int                        # staff index
    track: int                        # 0, 1, 2...
    group_id: int                     # nhóm nốt
    has_dot: bool                     # có chấm
    stem_up: bool                     # thân lên/xuống
    stem_right: bool                  # thân phải/trái
    staff_line_pos: int               # 0-8 (vị trí trên dây)
    sfn: str                          # None, "#", "b" (accidental)
    duration: Duration                # WHOLE, HALF, QUARTER...

class NoteGroup:
    notes: List[NoteHead]     # các nốt cùng thân
    stem_up: bool
    has_stem: bool
    all_same_type: bool       # cùng duration?

class Barline:
    x: int                    # x-coordinate
    staff1, staff2: int       # staffs bao quanh
    
class Clef:
    type: str                 # "G", "F", "C"
    staff: int                # staff nào

class Accidental:
    type: str                 # "#" hoặc "b"
    x, y: int                 # vị trí
    
class Rest:
    type: str                 # "whole", "half", "quarter"...
    staff: int                # staff nào
```

---

## 🔄 **PHẦN 5: SEQUENCE DIAGRAM (Thứ Tự Gọi Hàm)**

```
main()
  │
  ├─→ extract()
  │    │
  │    ├─→ generate_pred()
  │    │    ├─→ inference.py: inference('seg_unet.keras')
  │    │    │    ├─→ load_model()
  │    │    │    ├─→ sliding_window_inference()
  │    │    │    └─→ register_layer('staff_pred', ...)
  │    │    │        register_layer('symbols_pred', ...)
  │    │    │        register_layer('stems_rests_pred', ...)
  │    │    │
  │    │    └─→ inference.py: inference('seg_unet.keras' lần 2)
  │    │         └─→ register_layer('clefs_keys_pred', ...)
  │    │
  │    ├─→ dewarp.py: dewarp()
  │    │    ├─→ get_layer('staff_pred')
  │    │    ├─→ build_grid()
  │    │    ├─→ estimate_coords()
  │    │    └─→ register_layer('dewarped_img', ...)
  │    │
  │    ├─→ staffline_extraction.py: extract()
  │    │    ├─→ get_layer('staff_pred')
  │    │    ├─→ find_peaks()
  │    │    ├─→ create Staff objects
  │    │    └─→ register_layer('staffs', staffs)
  │    │
  │    ├─→ notehead_extraction.py: gen_notes()
  │    │    ├─→ get_layer('staffs')
  │    │    ├─→ get_layer('symbols_pred')
  │    │    ├─→ morph & find_contours()
  │    │    ├─→ assign_to_staff()
  │    │    ├─→ calculate_pitch()
  │    │    └─→ register_layer('notes', notes)
  │    │
  │    ├─→ note_group_extraction.py: group_noteheads()
  │    │    ├─→ get_layer('notes')
  │    │    ├─→ connected_component()
  │    │    ├─→ create NoteGroup objects
  │    │    └─→ register_layer('note_groups', note_groups)
  │    │
  │    ├─→ symbol_extraction.py: parse_all()
  │    │    ├─→ get_layer('stems_rests_pred')
  │    │    ├─→ get_layer('clefs_keys_pred')
  │    │    ├─→ parse_barlines()
  │    │    ├─→ parse_clefs()
  │    │    │    └─→ classifier.py: predict(clef.model)
  │    │    ├─→ parse_rests()
  │    │    └─→ register_layer('barlines', ...)
  │    │        register_layer('clefs', ...)
  │    │        register_layer('sfns', ...)
  │    │        register_layer('rests', ...)
  │    │
  │    ├─→ rhythm_extraction.py: parse_rhythm()
  │    │    ├─→ get_layer('notes')
  │    │    ├─→ get_layer('note_groups')
  │    │    ├─→ parse_dot()
  │    │    ├─→ parse_beams()
  │    │    ├─→ calculate_duration()
  │    │    └─→ register_layer('notes', notes) [updated]
  │    │
  │    └─→ build_system.py: MusicXMLBuilder.build()
  │         ├─→ get_layer('staffs')
  │         ├─→ get_layer('notes')
  │         ├─→ get_layer('barlines')
  │         ├─→ decode_note()
  │         ├─→ decode_rest()
  │         ├─→ to_musicxml()
  │         └─→ return musicxml_bytes
  │
  └─→ save to file

```

---

## ⚠️ **PHẦN 6: Những Điểm Quan Trọng**

### **6.1 unit_size - Chỉ Số Vàng**

**Tại sao quan trọng:**
- Tất cả kích thước được tính tương đối (relative) chứ không tuyệt đối
- Các bạn lớp khác nhau → unit_size khác nhau
- Nếu sai unit_size → tất cả bước sau đều sai

**Tính toán:**
```python
unit_size = (khoảng cách giữa dây 1 & 5) / 4
# VD: 5 dây cách đều 56 pixels → unit_size = 14 pixels
```

**Dùng để:**
```python
# Filter contour hợp lệ
min_w = unit_size * 0.5
max_w = unit_size * 2.5

# Tính pitch
pitch_index = round((notehead_y - staff_center) / unit_size)

# Tính duration (so với stem width)
stem_width_ratio = stem_width / unit_size
```

### **6.2 Pitch Assignment - Logic Phức Tạp**

```python
# Pitch phụ thuộc vào:
# 1. Clef type (G, F, C)
# 2. Staff nào (treble, bass, alto)
# 3. Vị trí y so với dây khuông
# 4. Accidental (sharp, flat)

# G clef (Treble, dây 2 = G4):
# Dây 1 → F5, Dây 2 → D5, Dây 3 (giữa) → B4, Dây 4 → G4, Dây 5 → E4
# Khoảng giữa: E5, C5, A4, F4, (dưới thêm D4, C4...)

# F clef (Bass, dây 4 = F3):
# Khoảng giữa các dây và dây được tính khác nhau
```

### **6.3 Beam Analysis - Suy Luận Logic**

```python
# Nốt đen + 1 chân = 8th (♪)
# Nốt đen + 2 chân = 16th (♬)
# Nốt đen + 3 chân = 32nd
# ...

# Khoảng cách giữa chân:
# - Chân dàn = các nốt khác nhau duration
# - Chân nối = các nốt cùng duration

# VD: ♪ ♪ ♬ ♬ (2 eighth + 2 sixteenth)
#     Nốt 1-2: nối chân 1
#     Nốt 3-4: nối chân 2+3
```

### **6.4 Layer Registry - Mẫu Thiết Kế (Design Pattern)**

**Tên gọi:** Registry Pattern (hay Service Locator)

**Ưu điểm:**
- ✅ Loose coupling (các module độc lập)
- ✅ Easy debugging (có thể in ra trung gian)
- ✅ Flexible order (có thể sắp xếp lại thứ tự)

**Nhược điểm:**
- ❌ Hidden dependencies (không rõ module phụ thuộc vào nhau)
- ❌ Global state (khó test)
- ❌ Mutable state (dữ liệu có thể bị thay đổi)

---

## 📋 **PHẦN 7: Checklist Debug**

Khi xảy ra lỗi, kiểm tra:

```
Bước 1: Kiểm tra inference
□ Model có load đúng không? (ONNX vs TensorFlow)
□ Input shape có đúng không?
□ Output prediction có reasonable không? (0-255)

Bước 2: Kiểm tra staffline extraction
□ staff_pred có đúng shape không?
□ unit_size có hợp lý không? (thường 10-20 pixels)
□ Số khuông có đúng không?

Bước 3: Kiểm tra notehead extraction
□ notes có được tạo không?
□ Pitch có đúng không? (test với ảnh đơn giản)
□ Bbox có hợp lý không?

Bước 4: Kiểm tra symbol extraction
□ barlines có được tìm không?
□ clefs có được classify đúng không?
□ accidentals có match với notes không?

Bước 5: Kiểm tra rhythm extraction
□ Duration có được tính không?
□ Dotted notes có được detect không?
□ Beam analysis có đúng không?

Bước 6: Kiểm tra MusicXML
□ XML schema có valid không?
□ Pitch format có đúng không? (<step>C</step><octave>4</octave>)
□ Duration (quarters) có hợp lý không?
```

---

Hy vọng tài liệu này giúp anh/chị hiểu rõ luồng xử lý chi tiết! 🎵
