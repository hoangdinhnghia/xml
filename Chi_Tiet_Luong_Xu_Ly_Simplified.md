# XÂY DỰNG HỆ THỐNG CHUYỂN ẢNH BẢN NHẠC THÀNH MUSICXML — TÓM TẮT LUỒNG XỬ LÝ

Tài liệu tóm tắt này giải thích ngắn gọn và rõ ràng các bước chính trong hệ thống, dùng ngôn ngữ tiếng Việt để phù hợp với báo cáo đồ án.

---

## 🎵 **PHẦN 1: TỔNG QUAN (MỞ ĐẦU)**

### **Bài Toán Là Gì?**

**Vấn đề:**
- Khi bạn chụp ảnh một bản nhạc bằng điện thoại → lưu thành file PNG
- Máy tính cần **tự động chuyển ảnh thành file nhạc** (có thể chạy được)
- File nhạc chuẩn gọi là **MusicXML** (dùng trong MuseScore, Finale...)

**Giải Pháp:**
- Hệ thống dùng **AI + Logic Toán** để:
  1. Nhìn vào ảnh bản nhạc
  2. Hiểu ảnh (khuông nhạc ở đâu, nốt nào, cao độ nào...)
  3. Xuất thành file MusicXML → có thể mở trong MuseScore để nghe nhạc

### **Kiến Trúc Tổng Quan (8 Bước)**

```
┌─────────────────────────────────────────┐
│  BƯỚC 1: AI Dự Đoán (Model U-Net)      │ ← Model học từ 30,000 ảnh
│         (Biết pixel nào là gì)          │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  BƯỚC 2: Sửa Cong (Dewarp)              │ ← Sửa ảnh bị xoay, cong
│         (Làm thẳng khuông nhạc)         │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  BƯỚC 3: Tìm Khuông Nhạc (Staffline)    │ ← Tìm vị trí 5 dây
│         (Xác định tỷ lệ cơ bản)         │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  BƯỚC 4: Tìm Các Nốt (Notehead)         │ ← Tìm từng nốt, xác định cao độ
│         (Xác định pitch C4, D4...)      │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  BƯỚC 5-7: Nhóm Nốt & Tìm Ký Hiệu       │ ← Nhóm nốt, tìm vạch, khóa...
│            (Phân tích chi tiết)         │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│  BƯỚC 8: Xuất MusicXML (Build System)    │ ← Ghép tất cả thành file nhạc
│         (File có thể mở trong MuseScore) │
└─────────────────────────────────────────┘
```

---

## 🔍 **PHẦN 2: CHI TIẾT TỪNG BƯỚC (DỄ HIỂU)**

### **BƯỚC 1: Dự Đoán (Inference) - AI Nhìn Ảnh**

**Điều gì xảy ra?**
- Máy tính nhìn ảnh bản nhạc → dự đoán **mỗi điểm nhỏ (pixel) trong ảnh là gì**
- VD: Điểm ảnh (100, 200) → "khuông nhạc", Điểm ảnh (150, 250) → "nốt"

**Cách làm:**
1. **Chia ảnh ra**: Vì ảnh to (2000×2500 pixel), máy tính chia thành những mảnh nhỏ (256×256)
2. **Dùng Model AI**: Mỗi mảnh → đưa vào model U-Net (AI đã học từ 30,000 ảnh)
3. **Dự đoán**: Model nói "mảnh này có khuông 95%, có nốt 20%..."
4. **Ghép lại**: Ghép tất cả mảnh nhỏ thành bảng dự đoán toàn ảnh

**Output:**
- Bảng dự đoán khuông nhạc (2000×2500): mỗi pixel có xác suất là khuông
- Bảng dự đoán ký hiệu (2000×2500): mỗi pixel có xác suất là nốt/ký hiệu khác

**Ví Dụ Thực Tế:**
```
Input:  Ảnh PNG bản nhạc chụp bằng điện thoại
Output: Bảng số (bản đồ):
        - Giá trị 255 = chắc chắn là khuông
        - Giá trị 128 = có thể là khuông
        - Giá trị 0 = chắc chắn không phải khuông
```

---

### **BƯỚC 2: Sửa Cong (Dewarp) - Làm Thẳng Ảnh**

**Tại sao cần?**
- Ảnh chụp bằng điện thoại thường bị **xoay, cong**
- Nếu khuông không thẳng → tính cao độ nốt sai

**Cách làm:**
1. Dùng bản đồ khuông từ bước 1 để tìm **vị trí khuông trong ảnh**
2. Xác định: khuông bị cong như thế nào?
3. "Kéo" ảnh lại cho thẳng (như kéo vải bị nhăn)

**Output:**
- Ảnh thẳng (khuông song song với mép ảnh)
- Bản đồ thẳng (tương ứng với ảnh)

**Ví Dụ:**
```
Trước: Ảnh xoay 5 độ, khuông bị cong
Sau:  Ảnh thẳng, khuông song song ngang
```

---

### **BƯỚC 3: Tìm Khuông Nhạc (Staffline) - Tim Vị Trí 5 Dây**

**Điều gì xảy ra?**
- Mục tiêu: **tìm chính xác vị trí 5 dây của khuông nhạc**
- Từ đó suy ra "khuông đó bao lớn"

**Cách làm (không cần hiểu chi tiết):**
1. Lấy bản đồ khuông từ bước 1
2. Tính tổng: từ trên xuống, hàng nào đậm nhất? → là dây khuông
3. Tìm được 5 hàng đặc biệt → đó là 5 dây
4. Tính khoảng cách giữa dây → được "tỷ lệ cơ bản" gọi là **unit_size**

**unit_size là gì? ⭐ QUAN TRỌNG**

```
Ảnh 300dpi:        Ảnh 600dpi:
┌─ Dây 1           ┌─ Dây 1
├─                 ├─
├─ Dây 2           ├─
├─                 ├─
└─ Dây 3           └─ Dây 3

Khoảng dây = 20px      Khoảng dây = 40px
unit_size = 20         unit_size = 40

Nốt thường: 0.5-2.5 × unit_size
```

**Tại sao unit_size quan trọng?**
- Dù ảnh DPI khác nhau, tỷ lệ **luôn cố định**
- unit_size → dùng để lọc nốt thật, bỏ nhiễu
- unit_size → dùng để tính cao độ nốt

---

### **BƯỚC 4: Tìm Nốt (Notehead) - Xác Định Cao Độ**

**Điều gì xảy ra?**
- Tìm mỗi nốt nhạc trong ảnh
- Xác định cao độ (C4, D4, E4, F4...)
- Xác định vị trí (x, y) trên ảnh

**Cách làm (3 bước):**

**Bước 4.1: Tìm Vị Trí Nốt**
```
1. Lấy bản đồ ký hiệu từ bước 1
2. Tìm các vùng liền nhau → contour (hình dạng nốt)
3. Lọc theo kích thước:
   - Nốt thường to = 10-50 pixel (dùng unit_size)
   - Loại bỏ bụi nhỏ, bỏ vạch lớn
4. Kết quả: danh sách vị trí nốt (x, y, kích thước)
```

**Bước 4.2: Tính Cao Độ (Pitch)**
```
Ví dụ: Khuông G (Treble)
┌─────── Dây 1 (y=200)
│
├─────── Dây 2 (y=220) 
│
├─────── Dây 3 (y=240)   ← GiữA khuông
│
├─────── Dây 4 (y=260)
│
└─────── Dây 5 (y=280)

Nốt ở y=240 (dây 3 giữa) → Pitch = B4
Nốt ở y=230 (giữa dây 2-3) → Pitch = C5
Nốt ở y=220 (dây 2) → Pitch = D5
...
```

**Bước 4.3: Xác Định Hướng Thân**
- Nếu thân nốt ở bên phải → thân "lên"
- Nếu thân nốt ở bên trái → thân "xuống"

**Output:**
- Danh sách nốt: [(x, y), pitch="D5", stem_up=True, ...]

---

### **BƯỚC 5: Nhóm Nốt (Note Grouping) - Nối Nốt Có Cùng Thân**

**Tại sao cần?**
- Những nốt được nối chân nằm chung thân
- Nhóm này quan trọng để tính thời gian (rhythm)

**Ví Dụ:**
```
♪ ♪ ♪ ♪    ← 4 nốt nối chân
└─┴─┴─┘    ← Cùng 1 thân

Nhóm = [Nốt1, Nốt2, Nốt3, Nốt4] + stem_direction="up"
```

---

### **BƯỚC 6: Tìm Ký Hiệu Khác (Symbol Extraction)**

**Tìm những gì?**
1. **Vạch bài** (barline): đường dọc chia các tác
2. **Khóa nhạc** (clef): G clef, F clef → xác định pitch baseline
3. **Dấu thăng/bé** (#, b): thay đổi cao độ nốt
4. **Ký hiệu tạm dừng** (rest): nốt im lặng

**Cách tìm:**
- Dùng bản đồ ký hiệu từ bước 1
- Dùng classifier (AI nhỏ được train riêng) để phân loại từng ký hiệu

---

### **BƯỚC 7: Tính Thời Gian (Rhythm Extraction)**

**Tìm độ dài nốt:**
```
Hình dạng nốt → Độ dài:
┌─┐ Nốt trắng, to     → WHOLE (4 beat)
│ │
├─┤ Nốt trắng, nhỏ    → HALF (2 beat)
│ │
●   Nốt đen           → QUARTER (1 beat)
●   + 1 chân          → 8TH (0.5 beat)
●   + 2 chân          → 16TH (0.25 beat)
```

**Dấu chấm:**
```
Nốt thường: 1 beat
Nốt có chấm: 1.5 beat (thêm nửa)
```

---

### **BƯỚC 8: Xuất Nhạc (MusicXML Build)**

**Ghép lại thành file:**
1. Lấy tất cả dữ liệu từ các bước trước
2. Sắp xếp theo thứ tự từ trái sang phải (từ đầu đến cuối bài)
3. Xuất thành file **MusicXML** (định dạng chuẩn)

**File MusicXML có gì?**
```
- Nốt: pitch (C4, D4...) + duration (quarter, half...)
- Vạch bài: chia các tác
- Khóa: G clef hay F clef
- Dấu: # hay b
- Tạm dừng: rest
- Tất cả định dạng theo chuẩn MusicXML 3.1
```

**Output:**
- File `.musicxml` → mở được trong MuseScore, Finale, v.v.

---

## 💾 **PHẦN 3: CÁCH DỮ LIỆU CHẢY QUA CÁC BƯỚC**

### **Layer Registry - "Kho Chứa Dữ Liệu"**

**Ý tưởng:**
- Các module không gọi hàm trực tiếp (tránh tham số dài)
- Thay vào đó, dùng "kho chứa" chung (layer registry)

**Ví Dụ:**
```
Module 1 (Inference): 
  → Tính xong bản đồ khuông
  → Lưu vào kho: put("staff_pred", bản_đồ_khuông)

Module 2 (Staffline Extraction):
  → Cần bản đồ khuông
  → Lấy từ kho: get("staff_pred")
  → Xử lý xong → Lưu vào kho: put("staffs", vị_trí_5_dây)

Module 3 (Notehead Extraction):
  → Cần vị_trí_5_dây
  → Lấy từ kho: get("staffs")
  → Xử lý xong → Lưu vào kho: put("notes", danh_sách_nốt)
```

**Lợi ích:**
- Module độc lập, không phụ thuộc lẫn nhau
- Dễ debug: in ra dữ liệu ở bất kỳ điểm nào
- Dễ thêm/xóa bước mà không sửa code cũ

---

## 📌 **PHẦN 4: NHỮNG ĐIỂM QUAN TRỌNG CẦN NHỚ**

### **1️⃣ unit_size - "Tỷ Lệ Ma Thuật"**

**Là gì?**
- Khoảng cách giữa 2 dây khuông nhạc
- = (khoảng từ dây 1 đến dây 5) / 4

**Tại sao quan trọng?**
- Dù ảnh to hay nhỏ, tỷ lệ **luôn cố định**
- unit_size → tất cả tính toán sau dùng nó

**Ví Dụ:**
```
Ảnh 300dpi: unit_size = 14 pixel
Ảnh 600dpi: unit_size = 28 pixel (gấp đôi)

Nhưng nốt thường 0.5-2.5 × unit_size
→ Nốt thường 7-35 pixel (300dpi) hay 14-70 pixel (600dpi)
→ Tỷ lệ không đổi!
```

### **2️⃣ Pitch Assignment - Cách Tính Cao Độ**

**Logic:**
```
Khuông G (Treble):
- Dây 3 giữa = B4
- Dây trên = cao hơn
- Dây dưới = thấp hơn
- Giữa dây = nốt khác

Ví dụ:
- Dây 1 = F5
- Giữa 1-2 = E5  
- Dây 2 = D5
- Giữa 2-3 = C5
- Dây 3 = B4 (giữa)
```

### **3️⃣ Beam Analysis - Chân Nốt**

**Nốt đen có bao nhiêu chân = độ dài:**
```
Không chân   → QUARTER (1 beat)
1 chân ♪     → 8TH (0.5 beat)
2 chân ♬     → 16TH (0.25 beat)
3 chân       → 32ND (0.125 beat)
```

---

## 🎯 **PHẦN 5: TÓMNÚT DỄ NHỚ**

| Bước | Tên | Input | Output | Mục Tiêu |
|------|-----|-------|--------|----------|
| 1 | AI Dự Đoán | Ảnh PNG | Bản đồ khuông, ký hiệu | AI nhìn hiểu ảnh |
| 2 | Sửa Cong | Bản đồ | Ảnh thẳng | Làm thẳng ảnh bị cong |
| 3 | Tìm Khuông | Bản đồ | Vị trí 5 dây, unit_size | Xác định tỷ lệ |
| 4 | Tìm Nốt | Bản đồ | Danh sách nốt + pitch | Xác định cao độ |
| 5 | Nhóm Nốt | Nốt | Nhóm nốt cùng thân | Phân tích kết nốt |
| 6 | Tìm Ký Hiệu | Bản đồ | Vạch, khóa, dấu, rest | Tìm chi tiết |
| 7 | Tính Thời Gian | Nốt | Nốt + duration | Xác định độ dài |
| 8 | Xuất Nhạc | Tất cả | File MusicXML | Output file dùng được |

---

## ✅ **PHẦN 6: CHECKLIST ĐỂ KIỂM TRA HIỂU**

- [ ] Bước 1 làm gì? (AI dự đoán)
- [ ] unit_size là gì? (khoảng dây)
- [ ] Pitch được tính như thế nào? (từ vị trí y)
- [ ] Tại sao có 2 model AI? (divide and conquer)
- [ ] File MusicXML dùng để gì? (file nhạc dùng được)
- [ ] Layer registry là gì? (kho chứa dữ liệu)
- [ ] Bước nào quan trọng nhất? (bước 3 - tìm khuông)

---

## 🎓 **KẾT LUẬN**

Hệ thống hoạt động theo **pipeline tuần tự**:
1. **AI thấy** ảnh là gì
2. **Sửa** ảnh bị cong
3. **Đo** khuông → unit_size
4. **Tìm** nốt → pitch
5. **Nhóm** nốt → thân
6. **Tìm** ký hiệu khác
7. **Tính** thời gian
8. **Xuất** file nhạc dùng được

**Điểm độc đáo:**
- Kết hợp **AI + Logic Toán**
- unit_size giải quyết vấn đề **tỷ lệ**
- Modular (8 module độc lập)
- Output là **file chuẩn MusicXML**

---

Hy vọng tài liệu này giúp anh/chị **hiểu rõ từng bước** và có thể **giải thích cho người khác** một cách dễ dàng! 🎵
