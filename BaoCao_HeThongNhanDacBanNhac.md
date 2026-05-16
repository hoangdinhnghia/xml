# BÁO CÁO ĐỒ ÁN TỐT NGHIỆP
## ỨNG DỤNG HỌC MÁY XÂY DỰNG HỆ THỐNG NHẬN DẠNG BẢN NHẠC SỬ DỤNG MÔ HÌNH U-NET

---

**XÂY DỰNG HỆ THỐNG CHUYỂN ĐỔI ẢNH BẢN NHẠC THÀNH DỮ LIỆU MUSICXML SỬ DỤNG MÔ HÌNH U-NET**

Tác giả: [Tên sinh viên]

Tóm tắt

Đồ án này xây dựng và đánh giá một hệ thống tự động chuyển ảnh bản nhạc in sang định dạng MusicXML. Hệ thống kết hợp các bước tiền xử lý ảnh, mô hình phân đoạn U-Net cho nhận diện ký hiệu, và hậu xử lý để chuyển kết quả phân đoạn thành dữ liệu nhạc có cấu trúc. Mục tiêu là tạo ra một quy trình thực tiễn, có thể minh họa trong buổi bảo vệ tốt nghiệp.

1. Giới thiệu

- Động lực: số hóa kho bản nhạc, hỗ trợ soạn nhạc và nghiên cứu âm nhạc.
- Mục tiêu: xây dựng pipeline từ ảnh đến MusicXML, thử nghiệm trên tập dữ liệu mẫu, báo cáo kết quả và hạn chế.

2. Dữ liệu và tài nguyên

- Dữ liệu nguồn: chứa trong `train_data`, `test_data` và thư mục `CvcMuscima`.
- Công cụ chính: mã nguồn trong thư mục `oemer`, mô hình U-Net, script chạy thử `oemer/ete.py`.

3. Phương pháp

- Tiền xử lý: chuẩn hóa kích thước, chuyển thang xám, loại nhiễu và làm thẳng khuông nhạc.
- Mô hình phân đoạn: U-Net được huấn luyện để phân biệt các lớp như nền, khuông nhạc, nốt, thân nốt, ký hiệu.
- Hậu xử lý: gom vùng, xác định vị trí nốt trên khuông, suy luận tiết tấu và ánh xạ sang MusicXML.

4. Triển khai và thử nghiệm

- Huấn luyện: cấu hình trong `oemer/train.py`, dữ liệu tăng cường được áp dụng để tăng tính bền vững.
- Suy luận: sử dụng `oemer/ete.py` để chuyển một ảnh mẫu sang MusicXML và lưu kết quả trong `output`.

5. Kết quả và đánh giá

- Chỉ số đánh giá: độ chính xác phân đoạn theo lớp, tỉ lệ ký hiệu ánh xạ đúng vào MusicXML, độ tin cậy vị trí nốt.
- Nhận xét: mô hình cho kết quả tốt với ảnh in chuẩn, giảm hiệu năng với ảnh mờ hoặc có xoắn nếp mạnh.

6. Kết luận

- Đồ án khẳng định tính khả thi của phương pháp kết hợp phân đoạn ảnh và hậu xử lý để chuyển ảnh bản nhạc sang MusicXML.
- Hướng phát triển: mở rộng nhận dạng ký hiệu phức tạp, cải thiện xử lý ảnh kém chất lượng, tích hợp kiểm tra ngữ cảnh âm nhạc.

Từ khóa: U-Net, phân đoạn ảnh, MusicXML, nhận dạng bản nhạc
