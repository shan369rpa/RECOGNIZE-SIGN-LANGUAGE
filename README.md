# AI Sign Language Detection (Vietnamese Alphabet)
# Nhận diện Thủ ngữ Việt Nam bằng AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-RandomForest-green)

[English](#english) | [Tiếng Việt](#tiếng-việt)

---

<a name="english"></a>
## 🇬🇧 English Description

This project demonstrates a real-time **Sign Language Detection System** utilizing Computer Vision and Machine Learning. It specifically targets the **Vietnamese Sign Language (VSL)** alphabet.

Instead of using heavy Convolutional Neural Networks (CNNs) on raw images, this project uses **MediaPipe Hands** to extract 21 key landmarks (skeletal points) of the hand. These coordinates are then fed into a **Random Forest Classifier** (Scikit-learn) for extremely fast and accurate prediction, even on low-end hardware (CPU only).

### Features
*   **High Performance:** Runs smoothly on CPU (Real-time FPS).
*   **Robustness:** Works reasonably well with complex backgrounds thanks to skeletal tracking.
*   **Dataset:** Trained on the *Viet Nam Sign Language Detection v6* dataset from Roboflow (HCMUT).

### Project Structure
*   `process_data_csv.py`: Converts raw images from the Roboflow dataset into a CSV file containing hand landmarks.
*   `train_model.py`: Trains a Random Forest model using the CSV data and saves it as `model.p`.
*   `main_app.py`: The main application that runs the webcam and performs real-time detection.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/sign-language-demo.git
    cd sign-language-demo
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Run the application (using the pre-trained model):**
    ```bash
    python main_app.py
    ```
    *Press 'Q' to exit.*

2.  **(Optional) Retrain the model:**
    *   Download the dataset from Roboflow (Format: Folder Structure or Multi-class CSV).
    *   Update the path in `process_data_csv.py`.
    *   Run `python process_data_csv.py` to generate `landmark_data.csv`.
    *   Run `python train_model.py` to generate `model.p`.

---

<a name="tiếng-việt"></a>
## 🇻🇳 Mô tả Tiếng Việt

Dự án demo hệ thống **Nhận diện Thủ ngữ (Ngôn ngữ ký hiệu)** thời gian thực, sử dụng Thị giác máy tính và Học máy. Dự án tập trung vào bảng chữ cái **Thủ ngữ Việt Nam**.

Thay vì xử lý trực tiếp hình ảnh nặng nề, dự án sử dụng **MediaPipe Hands** để trích xuất tọa độ 21 khớp xương bàn tay. Các tọa độ này sau đó được đưa vào thuật toán **Random Forest** (Rừng ngẫu nhiên) để phân loại. Phương pháp này giúp ứng dụng chạy cực nhẹ, nhanh và chính xác ngay cả trên máy tính cấu hình thấp không có GPU rời.

### Tính năng nổi bật
*   **Hiệu năng cao:** Chạy mượt mà thời gian thực (Real-time) trên CPU.
*   **Ổn định:** Không bị ảnh hưởng nhiều bởi phông nền phía sau (do sử dụng khung xương tay).
*   **Dữ liệu:** Được huấn luyện trên bộ dữ liệu *Viet Nam Sign Language Detection v6* (nguồn: ĐH Bách Khoa TP.HCM - Roboflow).

### Cấu trúc dự án
*   `process_data_csv.py`: Code xử lý ảnh thô tải từ Roboflow, chuyển đổi thành file CSV chứa tọa độ khớp tay.
*   `train_model.py`: Code huấn luyện mô hình AI từ file CSV và lưu ra file `model.p`.
*   `main_app.py`: Ứng dụng chính, bật Camera và nhận diện cử chỉ tay.

### Cài đặt

1.  **Tải mã nguồn:**
    ```bash
    git clone https://github.com/tên-của-bạn/sign-language-demo.git
    cd sign-language-demo
    ```

2.  **Cài đặt thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

### Hướng dẫn sử dụng

1.  **Chạy ứng dụng (đã có sẵn model):**
    ```bash
    python main_app.py
    ```
    *Nhấn phím 'Q' để thoát.*

2.  **(Tùy chọn) Huấn luyện lại mô hình:**
    *   Tải dataset từ Roboflow về máy.
    *   Cập nhật đường dẫn thư mục ảnh trong file `process_data_csv.py`.
    *   Chạy `python process_data_csv.py` để tạo file dữ liệu `landmark_data.csv`.
    *   Chạy `python train_model.py` để tạo file model mới `model.p`.

---
**Credits:**
*   Dataset: Ho Chi Minh University of Technology (Roboflow Universe).
*   Libraries: Google MediaPipe, Scikit-learn, OpenCV.