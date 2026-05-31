from typing import List

import numpy as np
from numpy import ndarray


# Từ điển lưu các lớp dữ liệu (numpy.ndarray) do từng bước trích xuất sinh ra.
# Mỗi khóa là tên lớp (ví dụ: staff_pred, symbols_pred...),
# mỗi giá trị là ma trận ảnh/đặc trưng tương ứng.
_layers = {}

_access_count = {}


def register_layer(name: str, layer: ndarray) -> None:
    """Đăng ký một lớp dữ liệu mới vào bộ nhớ dùng chung.

    Thuật toán:
    - Kiểm tra trùng tên để tránh ghi đè ngoài ý muốn.
    - Xác thực kiểu dữ liệu phải là `numpy.ndarray`.
    - Lưu layer vào `_layers` và khởi tạo bộ đếm truy cập bằng 0.

    Args:
        name: Tên định danh của lớp dữ liệu.
        layer: Ma trận dữ liệu cần lưu.
    """
    if name in _layers:
        print("Tên đã được đăng ký! Hãy chọn tên khác hoặc xóa tên cũ trước.")
        return

    assert isinstance(layer, np.ndarray)
    _layers[name] = layer
    _access_count[name] = 0


def get_layer(name: str) -> ndarray:
    """Lấy một lớp dữ liệu theo tên và tăng bộ đếm truy cập.

    Thuật toán:
    - Kiểm tra sự tồn tại của khóa trong `_layers`.
    - Nếu không tồn tại, ném `KeyError` để báo lỗi rõ ràng.
    - Nếu có, tăng `_access_count[name]` rồi trả về dữ liệu.

    Args:
        name: Tên lớp dữ liệu cần lấy.

    Returns:
        numpy.ndarray: Lớp dữ liệu tương ứng với tên đã cho.
    """
    if name not in _layers:
        raise KeyError(f"Tên layer chưa được đăng ký: {name}")
    _access_count[name] += 1
    return _layers[name]


def delete_layer(name):
    """Xóa một lớp dữ liệu và bộ đếm truy cập đi kèm (nếu có).

    Hàm xóa theo cơ chế an toàn: chỉ thao tác khi khóa tồn tại,
    giúp tránh phát sinh lỗi khi gọi xóa lặp lại.

    Args:
        name: Tên lớp dữ liệu cần xóa.
    """
    if name in _layers:
        del _layers[name]
        del _access_count[name]


def list_layers() -> List:
    """Trả về danh sách tên tất cả layer đang được lưu.

    Returns:
        List: Danh sách khóa hiện có trong `_layers`.
    """
    return list(_layers.keys())


def show_access_count():
    """In ra thống kê số lần truy cập của từng layer.

    Dùng để theo dõi mức độ sử dụng dữ liệu trung gian trong pipeline,
    hỗ trợ gỡ lỗi hoặc tối ưu luồng xử lý.
    """
    print(_access_count)
