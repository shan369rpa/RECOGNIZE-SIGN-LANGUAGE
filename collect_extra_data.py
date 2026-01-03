import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# --- CẤU HÌNH ---
DATA_FILE = 'landmark_data.csv'

# --- 1. HÀM HỖ TRỢ TIẾNG VIỆT ---
def put_text_vietnamese(img, text, position, font_size, color):
    """
    Hàm vẽ chữ tiếng Việt lên ảnh OpenCV
    color: (B, G, R)
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # Font Arial thường có sẵn trên Windows
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            # Thử đường dẫn tuyệt đối nếu đường dẫn ngắn không được
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default() # Fallback

    # Đổi màu từ BGR sang RGB
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 2. GIAO DIỆN CONSOLE (HƯỚNG DẪN TRƯỚC KHI CHẠY) ---
print("\n" + "="*60)
print("  CHƯƠNG TRÌNH THU THẬP DỮ LIỆU HUẤN LUYỆN (DATA COLLECTION)  ")
print("="*60)

# Nhập nhãn
while True:
    target_label = input(">> Nhập tên KÝ TỰ / CỬ CHỈ bạn muốn dạy (ví dụ: A, Like, So1): ").strip()
    if target_label:
        break
    print("Lỗi: Tên không được để trống! Vui lòng nhập lại.")

print(f"\n[XÁC NHẬN] Bạn đã chọn dạy chữ: '{target_label}'")
print(f"-> Dữ liệu sẽ được lưu vào: {DATA_FILE}")

print("\n" + "-"*60)
print("  HƯỚNG DẪN QUY TRÌNH THU THẬP  ")
print("-"*60)
print("1. Một cửa sổ Camera sẽ hiện lên.")
print("2. Đưa tay vào khung hình. Nếu máy chưa thấy tay, màn hình sẽ tối đi.")
print("3. Khi máy nhận diện được khung xương tay (các đường nối):")
print("   - Nhấn giữ phím 'S' để LƯU mẫu liên tục.")
print("   - Vừa nhấn 'S', vừa xoay nhẹ cổ tay, nghiêng trái/phải.")
print("   - Đưa tay ra xa/gần để máy học được kích thước to nhỏ.")
print("4. Thu thập khoảng 50 - 100 mẫu cho mỗi chữ là tốt nhất.")
print("5. Nhấn phím 'Q' để hoàn tất và thoát.")
print("-"*60)

input(">> Nhấn phím ENTER để bật Camera và bắt đầu...")

# --- 3. KHỞI TẠO HỆ THỐNG ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              # Chỉ học 1 tay cho chuẩn
    min_detection_confidence=0.5
)

# Kiểm tra tạo file CSV
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
        writer.writerow(header)
    print("-> Đã tạo file CSV mới.")
else:
    print("-> Đã tìm thấy file CSV cũ. Đang ghi nối đuôi (Append).")

# --- 4. VÒNG LẶP CAMERA ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0
last_save_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi Camera!")
        break
    
    # Lật ảnh gương
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    hand_detected = False

    # --- A. XỬ LÝ KHI PHÁT HIỆN TAY ---
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ khung xương
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Kiểm tra phím bấm
            key = cv2.waitKey(1)
            
            # Nhấn 'S' để lưu
            if key == ord('s') or key == ord('S'):
                data_row = [target_label]
                
                # Tính toán tọa độ tương đối (Relative Coordinates)
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                
                for landmark in hand_landmarks.landmark:
                    data_row.append(landmark.x - base_x)
                    data_row.append(landmark.y - base_y)
                
                # Ghi vào file
                with open(DATA_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
                
                counter += 1
                last_save_time = time.time()
                print(f"-> Đã lưu mẫu thứ {counter}")

    # --- B. VẼ GIAO DIỆN NGƯỜI DÙNG (SMART UI) ---
    
    # TRƯỜNG HỢP 1: KHÔNG THẤY TAY -> HIỆN HƯỚNG DẪN
    if not hand_detected:
        # Làm tối màn hình
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Vẽ hộp thông báo
        cx, cy = W // 2, H // 2
        cv2.rectangle(frame, (cx - 250, cy - 130), (cx + 250, cy + 130), (255, 255, 255), 2)
        
        # Viết chữ Tiếng Việt
        frame = put_text_vietnamese(frame, "KHÔNG TÌM THẤY TAY!", (cx - 200, cy - 90), 35, (0, 0, 255))
        frame = put_text_vietnamese(frame, "1. Đưa tay vào giữa khung hình", (cx - 220, cy - 20), 22, (255, 255, 255))
        frame = put_text_vietnamese(frame, "2. Xoay nhẹ cổ tay để đa dạng góc", (cx - 220, cy + 20), 22, (255, 255, 255))
        frame = put_text_vietnamese(frame, "3. Giữ khoảng cách ~50cm", (cx - 220, cy + 60), 22, (255, 255, 255))

    # TRƯỜNG HỢP 2: THẤY TAY -> HIỆN THÔNG SỐ
    else:
        # Thanh Header (Thông tin)
        cv2.rectangle(frame, (0, 0), (W, 70), (0, 0, 0), -1)
        frame = put_text_vietnamese(frame, f"Đang dạy: {target_label}", (20, 20), 30, (0, 255, 255))
        frame = put_text_vietnamese(frame, f"Số lượng: {counter}", (W - 250, 20), 30, (255, 255, 255))

        # Thanh Footer (Hướng dẫn phím)
        cv2.rectangle(frame, (0, H - 50), (W, H), (0, 0, 0), -1)
        frame = put_text_vietnamese(frame, "Giữ 'S': Lưu liên tục  |  'Q': Thoát", (30, H - 40), 20, (200, 200, 200))

        # Hiệu ứng "ĐÃ LƯU" màu xanh lá
        if time.time() - last_save_time < 0.5:
            frame = put_text_vietnamese(frame, "ĐÃ LƯU!", (W//2 - 100, H//2), 50, (0, 255, 0))

    cv2.imshow('Thu thap du lieu (Data Collection)', frame)
    
    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- 5. KẾT THÚC ---
print("\n" + "="*60)
print(f"  HOÀN THÀNH! TỔNG SỐ MẪU ĐÃ THÊM: {counter}")
print("="*60)
print("⚠️  LƯU Ý QUAN TRỌNG:")
print("Dữ liệu mới đã được thêm vào file CSV, nhưng AI chưa học nó.")
print("👉 Hãy chạy lệnh này ngay:  python train_model.py")
print("="*60)