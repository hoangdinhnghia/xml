# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

## 2.1 Nhận Dạng Ký Hiệu Nhạc (Optical Music Recognition - OMR)

### 2.1.1 Định Nghĩa và Tầm Quan Trọng
Nhận dạng ký hiệu nhạc (OMR) là quá trình chuyển đổi tự động các hình ảnh của bản nhạc in hoặc viết tay thành biểu diễn kỹ thuật số có thể xử lý được, chẳng hạn như tệp MusicXML. Hệ thống OMR đóng vai trò quan trọng trong:

- **Số hóa tài liệu lịch sử**: Bảo tồn các bản nhạc cổ điển qua dạng kỹ thuật số.
- **Tối ưu hóa quy trình biên tập nhạc**: Giảm thời gian nhập liệu thủ công.
- **Hỗ trợ giáo dục âm nhạc**: Cung cấp công cụ tương tác cho học viên.
- **Xử lý tài liệu nhạc lớn**: Quản lý các bộ sưu tập nhạc quy mô lớn.

### 2.1.2 Thách Thức Chính
Hệ thống OMR phải đối mặt với các thách thức như:

- **Độ phức tạp của ký hiệu nhạc**: Nhiều ký hiệu khác nhau với các biến thể về kích thước, hình dạng.
- **Chất lượng ảnh**: Các bản nhạc có thể bị xóa mờ, méo, hoặc có các vết nước.
- **Nền phức tạp**: Các đường kẻ khuông có thể không đều, các ký hiệu có thể chồng lấp.
- **Yêu cầu độ chính xác cao**: Lỗi nhỏ trong nhận dạng có thể dẫn đến sai lệch lớn về mặt âm nhạc.

---

## 2.2 Xử Lý Ảnh Kỹ Thuật Số (Digital Image Processing)

### 2.2.1 Khái Niệm Cơ Bản
Xử lý ảnh kỹ thuật số liên quan đến việc thao tác các biểu diễn kỹ thuật số của hình ảnh bằng máy tính. Các khái niệm cơ bản bao gồm:

- **Pixel**: Đơn vị cơ bản của hình ảnh kỹ thuật số, mỗi pixel chứa thông tin về màu sắc và độ sáng.
- **Không gian màu**: Cách biểu diễn màu (RGB, HSV, Grayscale, Binary).
- **Độ phân giải**: Số lượng pixel theo chiều ngang và chiều dọc (width × height).
- **Độ sâu bit**: Số bit dùng để biểu diễn mỗi pixel (8-bit = 0-255, 1-bit = nhị phân).

### 2.2.2 Các Kỹ Thuật Tiền Xử Lý (Preprocessing)

#### a) Chuyển đổi Không Gian Màu
- **Chuyển đổi sang Grayscale**: Giảm kích thước dữ liệu từ 3 channel (RGB) xuống 1 channel, vẫn giữ thông tin sáng tối.
- **Chuyển đổi sang Binary (nhị phân)**: Chuyển đổi mỗi pixel thành 0 (đen/nền) hoặc 1 (trắng/ký hiệu), thường sử dụng ngưỡng (thresholding).

#### b) Cải Thiện Chất Lượng Ảnh
- **Làm mịn (Blurring)**: Giảm nhiễu bằng cách trung bình hóa các pixel lân cận.
  ```
  Công thức Gaussian Blur:
  G(x,y) = (1 / 2πσ²) × exp(-(x²+y²) / 2σ²)
  ```
- **Tăng Độ Tương Phản (Contrast)**: Kéo rộng khoảng cách giữa giá trị sáng tối.
  ```
  G'(x,y) = (G(x,y) - mean) × factor + mean
  ```
- **Điều Chỉnh Độ Sáng (Brightness)**: Thêm/trừ giá trị hằng số từ tất cả pixel.

#### c) Tăng Cường Dữ Liệu (Data Augmentation)
Để cải thiện khả năng tổng quát hóa của mô hình máy học:

- **Xoay ảnh (Rotation)**: Xoay ảnh các góc ngẫu nhiên (0-360°).
- **Thu Phóng (Scaling/Resizing)**: Thay đổi kích thước ảnh để mô phỏng các bản nhạc khác nhau.
- **Dịch Chuyển (Translation)**: Dịch chuyển ảnh theo chiều X/Y.
- **Biến Dạng Phối Cảnh (Perspective Transform)**: Mô phỏng các bản nhạc bị chụp từ các góc khác nhau.
- **Xáo Trộn Pixel (Pixel Shuffle)**: Thêm vào các pixel ngẫu nhiên để mô phỏng nhiễu.
- **Thay Đổi Chất Lượng Ảnh (Encoding Quality)**: Nén ảnh JPEG với các mức chất lượng khác nhau.
- **Pixelize**: Giảm độ phân giải cục bộ để mô phỏng các ảnh mờ.

### 2.2.3 Xác Định Các Thành Phần Hình Ảnh
- **Bounding Box (BBox)**: Hình chữ nhật nhỏ nhất bao quanh một đối tượng, định nghĩa bằng (x₁, y₁, x₂, y₂).
- **Contour/Outline**: Đường bao quanh một đối tượng.
- **Connected Components**: Các tập hợp pixel kết nối với nhau theo tiêu chí khoảng cách.

---

## 2.3 Mạng Nơ-Ron Sâu và Học Máy (Deep Learning & Machine Learning)

### 2.3.1 Mạng Nơ-Ron Nhân Tạo (Artificial Neural Networks)

#### a) Cấu Trúc Cơ Bản
Mạng nơ-ron nhân tạo bao gồm các lớp:

- **Lớp Đầu Vào (Input Layer)**: Nhận dữ liệu đầu vào.
- **Lớp Ẩn (Hidden Layers)**: Xử lý và biến đổi dữ liệu.
- **Lớp Đầu Ra (Output Layer)**: Tạo ra dự đoán cuối cùng.

Công thức toán học cho một nơ-ron:
```
y = σ(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```
Trong đó:
- `w`: trọng số (weights)
- `x`: đầu vào (inputs)
- `b`: độ lệch (bias)
- `σ`: hàm kích hoạt (activation function)

#### b) Hàm Kích Hoạt (Activation Functions)
- **ReLU (Rectified Linear Unit)**:
  ```
  f(x) = max(0, x)
  ```
  Ưu điểm: Giảm bão hòa gradient, tính toán nhanh.

- **Sigmoid**:
  ```
  f(x) = 1 / (1 + e^(-x))
  ```
  Giá trị đầu ra nằm trong [0, 1], thích hợp cho xác suất.

- **Softmax**:
  ```
  f(x_i) = e^(x_i) / Σⱼ e^(x_j)
  ```
  Chuyển đổi mảng điểm thành phân phối xác suất.

### 2.3.2 Mạng Tích Chập (Convolutional Neural Networks - CNN)

#### a) Nguyên Lý Hoạt Động
CNN được thiết kế đặc biệt để xử lý dữ liệu hình ảnh thông qua:

- **Tích Chập (Convolution)**: Áp dụng bộ lọc (kernel) trên ảnh để trích xuất đặc trưng cục bộ.
  ```
  Output(i,j) = Σₘ Σₙ Image(i+m, j+n) × Kernel(m,n) + bias
  ```
  
- **Gộp (Pooling)**: Giảm kích thước đặc trưng bằng cách lấy giá trị max hoặc trung bình.
  ```
  Max Pooling: Output = max(Image region)
  Average Pooling: Output = mean(Image region)
  ```

- **Chuẩn Hóa Batch (Batch Normalization)**:
  ```
  ŷ = γ × (y - μ_batch) / √(σ_batch² + ε) + β
  ```
  Giúp tăng tốc độ hội tụ và ổn định huấn luyện.

#### b) Kiến Trúc U-Net
U-Net là một kiến trúc phân đoạn hình ảnh gồm:

- **Encoder (Đường xuống)**: Các lớp tích chập và pooling để giảm độ phân giải.
- **Bottleneck**: Lớp biểu diễn nén ở giữa mạng.
- **Decoder (Đường lên)**: Các lớp transposed convolution để tăng độ phân giải.
- **Skip Connections**: Kết nối trực tiếp từ encoder sang decoder để giữ lại chi tiết.

Ưu điểm: Hiệu quả với dữ liệu hạn chế, đầu ra có độ phân giải cao.

#### c) Kiến Trúc SegNet
SegNet là một kiến trúc phân đoạn ngữ cảnh gồm:

- **Encoder**: Tương tự VGG, giảm kích thước và tăng độ sâu.
- **Decoder**: Tăng kích thước lại bằng unpooling với các chỉ số lưu từ encoder.

Ưu điểm: Tiết kiệm bộ nhớ hơn U-Net, phù hợp cho xử lý ảnh lớn.

### 2.3.3 Hàm Mất Mát (Loss Functions)

#### a) Binary Focal Cross-Entropy
Được sử dụng cho các bài toán phân lớp nhị phân với dữ liệu không cân bằng:

```
BCE(p,y) = -[y × log(p) + (1-y) × log(1-p)]
FocalCE(p,y) = -[(1-p)^γ × y × log(p) + p^γ × (1-y) × log(1-p)]
```

Với γ (focusing parameter) để cân nhắc các mẫu khó.

#### b) Tversky Loss
Một biến thể của Dice Loss, được thiết kế để cân bằng false positives và false negatives:

```
Tversky(p,y) = TP / (TP + α×FN + (1-α)×FP)
Focal Tversky = (1 - Tversky)^γ
```

#### c) Kết Hợp Các Hàm Mất Mát
```
Loss = w₁ × FocalCE + w₂ × FocalTversky
```

Giúp tối ưu hóa đồng thời cả độ chính xác và sự cân bằng lớp.

### 2.3.4 Tối Ưu Hóa (Optimization)

#### a) Cơ Chế Lên Lịch Tỷ Lệ Học (Learning Rate Schedule)

**Warm-up Learning Rate**:
- Giai đoạn đầu: Tăng từ từ từ `min_lr` đến `init_lr`.
- Giai đoạn sau: Giảm dần với hệ số `decay_rate`.

```
Warm-up phase (t < warm_up_steps):
  lr(t) = min_lr + (init_lr - min_lr) × t / warm_up_steps

Decay phase (t ≥ warm_up_steps):
  cycle = (t - warm_up_steps) / decay_step
  lr(t) = init_lr × (decay_rate)^cycle - (offset % decay_step) / decay_step
```

Lợi ích: Tránh gradient explosion ở đầu, hội tụ nhanh hơn.

#### b) Tối Ưu Hóa Adam
```
m_t = β₁ × m_{t-1} + (1-β₁) × ∇f(θ)
v_t = β₂ × v_{t-1} + (1-β₂) × (∇f(θ))²
θ_t = θ_{t-1} - α × m_t / (√v_t + ε)
```

Ưu điểm: Tự động điều chỉnh tỷ lệ học cho từng tham số.

#### c) Early Stopping
Dừng huấn luyện khi độ chính xác trên tập validation không cải thiện sau `patience` epochs.

---

## 2.4 Phân Đoạn Hình Ảnh (Image Segmentation)

### 2.4.1 Định Nghĩa
Phân đoạn hình ảnh là quá trình chia ảnh thành các vùng hoặc pixel có các thuộc tính tương tự. Có hai loại chính:

- **Phân đoạn Ngữ Cảnh (Semantic Segmentation)**: Mỗi pixel được gán nhãn lớp, tất cả đối tượng của cùng lớp có cùng màu.
- **Phân đoạn Thể Thể (Instance Segmentation)**: Mỗi thể hiện của đối tượng được gán nhãn riêng, ngay cả khi cùng lớp.

### 2.4.2 Phương Pháp Dự Đoán

#### a) Argmax Classification
Với ma trận xác suất đa lớp P(x,y) = [p₀, p₁, ..., pₙ]:
```
class_map(x,y) = argmax_c P_c(x,y)
```

Mỗi pixel được gán lớp có xác suất cao nhất.

#### b) Manual Thresholding
Áp dụng ngưỡng tùy chỉnh cho từng lớp:
```
class_map(x,y) = {1 if P_c(x,y) > threshold_c, 0 otherwise}
```

### 2.4.3 Xử Lý Ảnh Sau Phân Đoạn (Post-processing)

#### a) Morphological Operations
- **Erosion (Xói mòn)**: Loại bỏ các pixel biên từ đối tượng.
  ```
  Erosion(x,y) = min(Image(x+i, y+j)) for all (i,j) in kernel
  ```
  
- **Dilation (Giãn nở)**: Thêm pixel vào biên của đối tượng.
  ```
  Dilation(x,y) = max(Image(x+i, y+j)) for all (i,j) in kernel
  ```

- **Opening**: Erosion sau đó Dilation, loại bỏ các chi tiết nhỏ.
- **Closing**: Dilation sau đó Erosion, lấp đầy các lỗ nhỏ.

#### b) Connected Components Analysis
Tìm các tập hợp pixel kết nối:
```
Label(x,y) = ID của connected component chứa pixel (x,y)
```

#### c) Bounding Box Extraction
Từ mask hoặc contours, xác định hình chữ nhật nhỏ nhất:
```
BBox = (x_min, y_min, x_max, y_max)
```

---

## 2.5 Xử Lý Dữ Liệu Đầu Vào Hình Ảnh (Data Processing)

### 2.5.1 Chia Nhỏ Ảnh (Image Tiling)
Chia ảnh lớn thành các patch nhỏ kích thước `win_size × win_size` với bước trượt `step_size`:

```
Quy trình:
1. Duyệt từ (0,0) đến (height, width) với bước step_size
2. Với mỗi vị trí (y,x), trích xuất patch [y:y+win_size, x:x+win_size]
3. Nếu patch vượt biên, dịch chuyển để patch nằm gọn trong ảnh
```

Lợi ích:
- Giảm yêu cầu bộ nhớ.
- Áp dụng mô hình nhỏ hơn.
- Tránh overfitting trên ảnh lớn.

### 2.5.2 Ghép Kết Quả Patch (Patch Merging)

#### a) Trung Bình Theo Đếm (Count Averaging)
```
Output(x,y) = Sum of predictions / Count of overlapping patches
```

#### b) Ghép Có Trọng Số Gaussian (Gaussian-Weighted Merging)
Áp dụng trọng số Gaussian để các vùng giữa patch có độ tin cậy cao hơn:

```
W(x,y) = exp(- ((x-x_c)² + (y-y_c)²) / (2σ²))
Output(x,y) = Sum(prediction × W) / Sum(W)
```

Lợi ích: Làm giảm các seam artifacts tại biên patch.

### 2.5.3 Caching và Tối Ưu Hóa Hiệu Suất
- **Cache Key**: Tạo hash từ model path, image path, và tham số để tái sử dụng kết quả.
- **Checksum Model**: Lưu SHA256 của file model để phát hiện thay đổi.
- **Batch Processing**: Xử lý nhiều patch cùng lúc để tối ưu hóa GPU/CPU.

---

## 2.6 Lý Thuyết Thông Tin (Information Theory)

### 2.6.1 Entropy Thông Tin
Đo độ không chắc chắn của một phân phối xác suất:

```
H(x,y) = - Σ_c P_c(x,y) × log₂(P_c(x,y))
```

- `H = 0`: Xác suất tập trung hoàn toàn (chắc chắn).
- `H = log₂(n)`: Xác suất phân tán đều (hoàn toàn không chắc chắn).

Ứng dụng: Tạo heatmap entropy để đánh dấu các vùng dự đoán yếu.

### 2.6.2 Độ Tin Cậy (Confidence)
```
Confidence(x,y) = max(P_c(x,y))
```

Độ tin cậy thấp (< 0.6) biểu thị các vùng cần xem xét lại.

---

## 2.7 Bộ Dữ Liệu (Datasets)

### 2.7.1 CVC-MUSCIMA Dataset
Bộ dữ liệu công khai cho phân đoạn và nhận dạng ký hiệu nhạc:

- **Nguồn**: Bộ sưu tập hình ảnh bản nhạc được quét.
- **Biến Thể**: Gồm các loại bản nhạc bị méo (curvature, rotated, staffline-thickness-variation, v.v.).
- **Cấu Trúc**:
  ```
  dataset/
  ├── curvature/
  ├── ideal/
  ├── interrupted/
  ├── kanungo/
  ├── rotated/
  ├── staffline-thickness-variation-v1/
  ├── ...
  └── whitespeckles/
  
  Mỗi folder chứa:
  ├── image/          (ảnh gốc)
  ├── gt/            (ground truth stafflines)
  └── symbol/        (ground truth symbols)
  ```

### 2.7.2 DeepScore Dataset
Bộ dữ liệu cho phân đoạn ngữ cảnh của ký hiệu nhạc:

- **Cấu Trúc**:
  ```
  dataset/
  ├── images/        (ảnh gốc)
  └── segmentation/  (bản đồ phân đoạn)
  ```

- **Đặc điểm**: Chứa hơn 2000 bản nhạc với chú thích chi tiết.

---

## 2.8 Định Dạng Dữ Liệu Âm Nhạc (Music Data Format)

### 2.8.1 MusicXML
Định dạng XML tiêu chuẩn để biểu diễn ký hiệu nhạc:

- **Cấu trúc cơ bản**:
  ```xml
  <score-partwise>
    <part-list>
      <score-part id="P1"/>
    </part-list>
    <part id="P1">
      <measure number="1">
        <note>
          <pitch>
            <step>C</step>
            <octave>4</octave>
          </pitch>
          <duration>4</duration>
        </note>
      </measure>
    </part>
  </score-partwise>
  ```

- **Thông tin chứa**:
  - Độ cao của nốt (step + octave + alter)
  - Trường độ (duration)
  - Các ký hiệu (clef, time signature, key signature)
  - Các chỉ dẫn diễn tấu

### 2.8.2 Biểu Diễn Nốt Nhạc
- **Step**: Tên nốt (C, D, E, F, G, A, B).
- **Octave**: Số octave (thường 3-6).
- **Alter**: Hóa biểu (# = +1, b = -1, không có = 0).
- **Duration**: Thời lượng theo đơn vị phân chia (quarter note = 4).

Ví dụ: C# ở octave 4 với trường độ nửa quãng (eighth note):
```xml
<note>
  <pitch>
    <step>C</step>
    <alter>1</alter>
    <octave>4</octave>
  </pitch>
  <duration>2</duration>
</note>
```

---

## Tóm Tắt Chương 2

Chương này đã trình bày các nền tảng lý thuyết cần thiết cho việc xây dựng hệ thống OMR:

1. **OMR** cung cấp cách tiếp cận vấn đề của số hóa ký hiệu nhạc.
2. **Xử lý ảnh** cung cấp các kỹ thuật tiền xử lý, tăng cường và trích xuất đặc trưng.
3. **Mạng nơ-ron sâu** cung cấp khả năng học và dự đoán tự động.
4. **Phân đoạn hình ảnh** cho phép trích xuất các thành phần cụ thể từ ảnh.
5. **Xử lý dữ liệu** giúp tối ưu hóa hiệu suất trên ảnh lớn.
6. **Lý thuyết thông tin** giúp đánh giá độ tin cậy của dự đoán.
7. **Bộ dữ liệu** và **định dạng âm nhạc** cung cấp dữ liệu thực tế và cách biểu diễn kết quả.

Tổng hợp tất cả, chúng tạo thành một hệ thống toàn diện để chuyển đổi hình ảnh bản nhạc thành biểu diễn kỹ thuật số.
