
# 🖐️ AI Sign Language Translator (Hệ Thống Phiên Dịch Thủ Ngữ AI)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6)
![Tech](https://img.shields.io/badge/AI-MediaPipe%20%26%20Sklearn-orange)
![License](https://img.shields.io/badge/License-Commercial-green)

**AI Sign Language Translator** là giải pháp phần mềm hỗ trợ giao tiếp cho người khiếm thính, sử dụng công nghệ Thị giác máy tính (Computer Vision) để chuyển đổi ngôn ngữ ký hiệu tay thành văn bản Tiếng Việt theo thời gian thực.

Phần mềm hoạt động **hoàn toàn Offline**, không cần internet và được tối ưu hóa để chạy mượt mà trên các máy tính văn phòng phổ thông (không yêu cầu Card đồ họa rời).

---

## 🌟 Chức Năng Nổi Bật

1.  **Nhận diện thời gian thực (Real-time):** Phản hồi tức thì với độ trễ thấp (< 0.1s).
2.  **Hỗ trợ Tiếng Việt:** Hiển thị kết quả và giao diện hướng dẫn 100% Tiếng Việt có dấu.
3.  **Cơ chế Tự học (Auto-Train):** Cho phép người dùng tự thêm từ vựng mới thông qua Camera mà không cần biết lập trình.
4.  **Giao diện Thông minh (Smart UI):** Tự động phát hiện lỗi (không thấy tay, thiếu sáng) và đưa ra hướng dẫn khắc phục ngay trên màn hình.

---

## 📸 Giao Diện & Demo

### 1. Bảng Điều Khiển Trung Tâm (Dashboard)
Giao diện chính hiện đại, cho phép truy cập nhanh vào 3 chức năng cốt lõi: Nhận diện, Thu thập dữ liệu và Huấn luyện AI. Hệ thống Log bên dưới giúp theo dõi trạng thái phần mềm.

> **[CHÈN HÌNH ẢNH GIAO DIỆN CHÍNH (software_launcher) TẠI ĐÂY]**
> *Hình 1: Màn hình khởi động phần mềm.*

---

### 2. Chức Năng Nhận Diện (Detection)
Phần mềm tự động phát hiện khung xương tay và hiển thị chữ cái/câu từ tương ứng.
*   **Chế độ Fullscreen:** Tối ưu trải nghiệm nhìn.
*   **Smart Guide:** Tự động ẩn hướng dẫn khi phát hiện tay để màn hình thoáng đãng.

> **[CHÈN VIDEO HOẶC HÌNH ẢNH KHI ĐANG NHẬN DIỆN (main_app) TẠI ĐÂY]**
> *Hình 2: AI nhận diện chữ "Xin Chào" với độ tin cậy 98%.*

---

### 3. Chức Năng Thu Thập Dữ Liệu (Data Collection)
Công cụ giúp người dùng dạy từ mới cho máy. Có các chỉ dẫn trực quan về cách đặt tay, xoay cổ tay để đạt hiệu quả cao nhất.

> **[CHÈN HÌNH ẢNH MÀN HÌNH THU THẬP (collect_extra_data) TẠI ĐÂY]**
> *Hình 3: Giao diện thu thập dữ liệu với hiệu ứng thông báo "ĐÃ LƯU".*

---

### 4. Chức Năng Huấn Luyện Mô Hình (Training)
Sau khi thu thập dữ liệu, chức năng này sẽ kích hoạt thuật toán Machine Learning để học các mẫu mới. Quá trình được hiển thị qua thanh tiến độ chi tiết.

> **[CHÈN HÌNH ẢNH LOG HUẤN LUYỆN (train_model đang chạy trên launcher) TẠI ĐÂY]**
> *Hình 4: Quá trình huấn luyện mô hình với thanh tiến độ thời gian thực.*

---

## 📖 Hướng Dẫn Sử Dụng Chi Tiết

### Bước 1: Khởi động
Chạy file `software_launcher.exe` trong thư mục cài đặt.

### Bước 2: Sử dụng các chức năng

#### 🅰️ Chế độ Nhận diện (Detect)
1.  Nhấn nút **"📷 BẮT ĐẦU NHẬN DIỆN"**.
2.  Một bảng chọn sẽ hiện ra:
    *   **1 Tay:** Độ chính xác cao nhất, tốc độ nhanh nhất (Khuyên dùng).
    *   **2 Tay:** Dành cho các ký hiệu phức tạp cần phối hợp 2 tay.
3.  Đưa tay vào khung hình Camera.
4.  Nhấn phím **`Q`** để thoát và quay lại menu chính.

#### 🅱️ Dạy từ mới cho AI (Collect Data)
1.  Nhấn nút **"➕ THU THẬP DỮ LIỆU"**.
2.  Nhập tên từ/chữ cái bạn muốn dạy (Ví dụ: `CamOn`, `TamBiet`, `A`, `B`...).
3.  Cửa sổ Camera hiện lên:
    *   Tạo dáng tay tương ứng trước Camera.
    *   **Nhấn giữ phím `S`**: Để lưu mẫu liên tục. Hãy xoay nhẹ cổ tay, đưa tay xa/gần để máy học được nhiều góc độ.
    *   *Khuyên dùng:* Thu thập khoảng **50 - 100 mẫu** cho một từ.
4.  Nhấn phím **`Q`** để hoàn tất.

#### 🅾️ Cập nhật trí tuệ (Train Model)
*Lưu ý: Thực hiện bước này sau khi bạn đã Thu thập dữ liệu mới.*
1.  Nhấn nút **"🧠 HUẤN LUYỆN MODEL"**.
2.  Quan sát nhật ký hệ thống (Logs) bên dưới.
3.  Chờ thanh tiến độ chạy đến 100% và thông báo **"✅ HUẤN LUYỆN HOÀN TẤT!"**.
4.  Lúc này, bạn có thể quay lại chế độ Nhận diện để kiểm tra từ mới học.

---

## ⚙️ Yêu Cầu Hệ Thống

*   **Hệ điều hành:** Windows 10 hoặc Windows 11 (64-bit).
*   **CPU:** Intel Core i3 (thế hệ 4 trở lên) hoặc tương đương.
*   **RAM:** Tối thiểu 4GB.
*   **Camera:** Webcam Laptop hoặc Webcam USB rời (Độ phân giải HD 720p trở lên).
*   **Dung lượng ổ cứng:** 200MB trống.

---

## 🛠️ Dành Cho Nhà Phát Triển (Developer)

Nếu bạn muốn tùy chỉnh mã nguồn, vui lòng cài đặt môi trường như sau:

1.  **Clone Repo:**
    ```bash
    git clone https://github.com/your-username/sign-language-ai.git
    ```
2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Cấu trúc thư mục:**
    *   `main_app.py`: Core nhận diện.
    *   `collect_extra_data.py`: Core thu thập dữ liệu.
    *   `train_model.py`: Core huấn luyện (Random Forest).
    *   `software_launcher.py`: Giao diện điều khiển (Tkinter).

---

**Liên hệ hỗ trợ:**
*   Email: [Email của bạn]
*   Website: [Website của bạn nếu có]

*Copyright © 2026 AI Sign Language Project. All rights reserved.*