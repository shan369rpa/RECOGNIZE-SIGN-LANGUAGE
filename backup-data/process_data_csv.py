import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# --- CẤU HÌNH ĐƯỜNG DẪN (Sửa lại cho đúng máy bạn) ---
# Đường dẫn đến thư mục TRAIN chứa ảnh và file _classes.csv
DATASET_FOLDER = r"C:\Users\Son\Downloads\Viet Nam Sign Language Detection.v6i.multiclass\train"
CSV_FILE_NAME = "_classes.csv"

OUTPUT_FILE = "landmark_data.csv"

# --- KHỞI TẠO MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

# --- CHUẨN BỊ FILE OUTPUT ---
# Ghi header cho file CSV kết quả
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header: label, x0, y0, ..., x20, y20
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}'])
    writer.writerow(header)

print(f"Bắt đầu xử lý dữ liệu tại: {DATASET_FOLDER}")

# --- XỬ LÝ DỮ LIỆU ---
csv_path = os.path.join(DATASET_FOLDER, CSV_FILE_NAME)

if not os.path.exists(csv_path):
    print(f"Lỗi: Không tìm thấy file {csv_path}")
    exit()

count = 0
errors = 0
skipped = 0

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    
    # Lấy danh sách tên các cột (trừ cột filename) để biết có những nhãn nào (A, B, C...)
    # Cấu trúc reader.fieldnames sẽ là ['filename', ' A', ' B', ...] (chú ý khoảng trắng nếu có)
    class_columns = [col for col in reader.fieldnames if col.strip() != 'filename']
    
    for row in reader:
        # 1. Xác định nhãn của ảnh này
        image_label = None
        for col in class_columns:
            if row[col].strip() == '1': # Nếu cột nào có giá trị 1, đó là nhãn
                image_label = col.strip()
                break
        
        if image_label is None:
            continue # Không tìm thấy nhãn nào là 1
            
        # 2. Lấy đường dẫn ảnh
        image_filename = row['filename'].strip()
        image_path = os.path.join(DATASET_FOLDER, image_filename)
        
        # Kiểm tra ảnh có tồn tại không
        if not os.path.exists(image_path):
            skipped += 1
            continue

        # 3. Đọc và xử lý ảnh bằng MediaPipe
        img = cv2.imread(image_path)
        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Tạo dòng dữ liệu: [Label, x0, y0, ... y20]
                data_row = [image_label]
                
                # Lấy gốc tọa độ là cổ tay (điểm số 0)
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                
                for landmark in hand_landmarks.landmark:
                    # Lưu tọa độ tương đối (Relative Coordinates)
                    data_row.append(landmark.x - base_x)
                    data_row.append(landmark.y - base_y)
                
                # Ghi vào file CSV kết quả
                with open(OUTPUT_FILE, 'a', newline='') as output_f:
                    writer = csv.writer(output_f)
                    writer.writerow(data_row)
                
                count += 1
                if count % 100 == 0:
                    print(f"Đã xử lý {count} ảnh... (Đang ở nhãn {image_label})")
        else:
            errors += 1 # MediaPipe không tìm thấy tay trong ảnh này

print("------------------------------------------------")
print(f"HOÀN THÀNH!")
print(f"- Đã lưu được: {count} mẫu dữ liệu vào '{OUTPUT_FILE}'")
print(f"- Không tìm thấy tay (MediaPipe lỗi): {errors} ảnh")
print(f"- Không tìm thấy file ảnh hoặc lỗi đọc: {skipped} ảnh")
print("Bây giờ hãy chạy file 'train_model.py'!")