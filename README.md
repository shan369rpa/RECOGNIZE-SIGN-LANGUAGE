# 🖐️ AI Sign Language Detection (Vietnamese Alphabet)
# Nhận diện Thủ ngữ Việt Nam bằng AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-RandomForest-green)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

[🇬🇧 English Instructions](#english) | [🇻🇳 Hướng dẫn Tiếng Việt](#vietnamese)

---

<a name="english"></a>
## 🇬🇧 English Description

This project demonstrates a real-time **Sign Language Detection System** using Computer Vision and Machine Learning. It is optimized for the **Vietnamese Sign Language (VSL)** alphabet but can be easily retrained for any hand gestures.

Unlike traditional heavy CNN models, this project utilizes **MediaPipe Hands** to extract 21 skeletal landmarks of the hand. These coordinates are processed by a **Random Forest Classifier** (Scikit-learn), resulting in a lightweight, high-performance application that runs smoothly on CPUs without requiring a dedicated GPU.

### Key Features
*   **🚀 High Performance:** Real-time detection with high FPS on standard CPUs.
*   **🧠 Smart UI:** Interactive interface with Vietnamese support (via Pillow) and intelligent guidance.
*   **🛠️ Easy Training:** Includes tools to collect custom data and visualize training progress (`tqdm`).
*   **📊 Robustness:** Uses skeletal tracking, making it resilient to background noise and lighting changes.

### 1. Prerequisites (For Beginners)

Before running the code, ensure you have the following installed:

1.  **Python (3.8 - 3.11):**
    *   Download from [python.org](https://www.python.org/downloads/).
    *   ⚠️ **IMPORTANT:** During installation, check the box **"Add Python to PATH"**.
2.  **Git:**
    *   Download from [git-scm.com](https://git-scm.com/).
3.  **Code Editor:**
    *   Recommended: [Visual Studio Code](https://code.visualstudio.com/).

### 2. Installation

Open your Terminal (Command Prompt/PowerShell) and follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sign-language-demo.git
    cd sign-language-demo
    ```

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    # Activate:
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Usage Guide

#### ▶️ Run the Application
To start detecting sign language immediately:
```bash
python main_app.py
```
*   Select mode: `1` (One hand) or `2` (Two hands).
*   Press `Q` to exit.

#### 🔄 Train New/Custom Gestures
If you want to add your own gestures (e.g., "Like", "Hello") or improve accuracy:

1.  **Collect Data:**
    ```bash
    python collect_extra_data.py
    ```
    *   Enter the label name (e.g., `Like`).
    *   Hold `S` to save samples (capture ~50-100 frames).

2.  **Retrain Model:**
    ```bash
    python train_model.py
    ```
    *   Wait for the progress bar to finish.
    *   The new `model.p` will be saved automatically.

---

<a name="vietnamese"></a>
## 🇻🇳 Hướng dẫn Tiếng Việt

Dự án demo hệ thống **Nhận diện Thủ ngữ (Ngôn ngữ ký hiệu)** thời gian thực. Dự án tập trung vào bảng chữ cái **Thủ ngữ Việt Nam**, sử dụng công nghệ nhận diện khung xương tay (Hand Landmarks).

Hệ thống kết hợp **MediaPipe** (Google) để bắt tọa độ tay và **Random Forest** (Scikit-learn) để phân loại. Nhờ đó, ứng dụng cực kỳ nhẹ, chạy mượt trên mọi máy tính văn phòng mà không cần Card màn hình rời (GPU).

### Tính năng nổi bật
*   **🚀 Siêu nhẹ & Nhanh:** Chạy mượt mà thời gian thực (Real-time).
*   **🧠 Giao diện Thông minh:** Hỗ trợ hiển thị Tiếng Việt có dấu, tự động hướng dẫn khi không thấy tay.
*   **🛠️ Dễ dàng tùy biến:** Tự thêm dữ liệu tay của bạn để model học thêm.
*   **📊 Trực quan:** Có thanh tiến độ (loading) khi huấn luyện model.

### 1. Cài đặt Môi trường (Cho người mới)

Nếu bạn chưa từng lập trình Python, hãy làm theo các bước sau:

#### Bước 1: Cài đặt Python
1.  Truy cập [python.org/downloads](https://www.python.org/downloads/).
2.  Tải bản **Python 3.10** hoặc **3.11** (Ổn định nhất).
3.  Chạy file cài đặt.
    *   ⚠️ **QUAN TRỌNG:** Phải tích vào ô vuông **"Add Python to PATH"** ở màn hình đầu tiên. Nếu quên bước này, bạn sẽ không chạy được lệnh `python`.
4.  Bấm *Install Now* và đợi xong.

#### Bước 2: Cài đặt Git (Để tải code)
1.  Truy cập [git-scm.com](https://git-scm.com/).
2.  Tải và cài đặt (Cứ bấm Next liên tục là được).

#### Bước 3: Tải mã nguồn về máy
1.  Tạo một thư mục trống trên máy tính.
2.  Nhấn chuột phải vào thư mục đó, chọn **"Open Git Bash Here"** (hoặc mở CMD).
3.  Gõ lệnh:
    ```bash
    git clone https://github.com/TEN-GITHUB-CUA-BAN/sign-language-demo.git
    ```
    *(Thay link trên bằng link repo của bạn)*.

### 2. Cài đặt Thư viện

Mở Terminal (CMD hoặc PowerShell) tại thư mục dự án vừa tải về và chạy lệnh:

```bash
pip install -r requirements.txt
```
*Lệnh này sẽ tự động cài: OpenCV, MediaPipe, Scikit-learn, Pillow, Tqdm...*

### 3. Hướng dẫn Sử dụng

#### ▶️ Chạy ứng dụng nhận diện (Demo)
```bash
python main_app.py
```
*   Nhập `1` để chọn chế độ 1 tay (Chính xác cao).
*   Nhập `2` để chọn chế độ 2 tay.
*   Đưa tay lên Camera để trải nghiệm.
*   Nhấn phím `Q` để thoát.

#### 🔄 Dạy thêm chữ mới (Hoặc sửa lỗi nhận diện sai)
Nếu máy nhận diện tay của bạn không chuẩn, hoặc bạn muốn thêm ký hiệu mới (ví dụ: thả tim, like):

1.  **Bước 1: Thu thập dữ liệu**
    ```bash
    python collect_extra_data.py
    ```
    *   Nhập tên chữ muốn dạy (ví dụ: `Tim`).
    *   Cửa sổ Camera hiện lên: Đưa tay tạo dáng "Thả tim".
    *   **Nhấn giữ phím `S`**: Máy sẽ chụp liên tục (Làm khoảng 50 - 100 tấm).
    *   Nhấn `Q` để thoát.

2.  **Bước 2: Huấn luyện lại não cho AI**
    ```bash
    python train_model.py
    ```
    *   Nhìn thanh tiến độ chạy đến 100%.
    *   Xong! Chạy lại `main_app.py` để tận hưởng kết quả.

### Cấu trúc dự án
*   `main_app.py`: Chương trình chính (Camera nhận diện).
*   `collect_extra_data.py`: Công cụ thu thập thêm dữ liệu (có giao diện hướng dẫn).
*   `train_model.py`: Công cụ dạy học cho AI (có thanh loading).
*   `landmark_data.csv`: File chứa dữ liệu tọa độ tay (Dữ liệu gốc).
*   `model.p`: File bộ não AI đã được học (được sinh ra từ file CSV).

---
**Credits / Nguồn tham khảo:**
*   Dataset gốc: Viet Nam Sign Language Detection v6 (Roboflow Universe - HCMUT).
*   Công nghệ: Google MediaPipe & Scikit-learn.
```

### Lưu ý cuối cùng trước khi Push lên Git:

1.  Đảm bảo bạn đã có đầy đủ các file code mới nhất (`main_app.py`, `collect_extra_data.py`, `train_model.py`) trong thư mục.
2.  Đảm bảo file `.gitignore` đã chặn các thư mục rác (`venv`, `__pycache__`).
3.  Thực hiện bộ lệnh Git "thần thánh":
    ```bash
    git add .
    git commit -m "Update full features: Smart UI, Vietnamese support, Tqdm training"
    git push origin main
    ```