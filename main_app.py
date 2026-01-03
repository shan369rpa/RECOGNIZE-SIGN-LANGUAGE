import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- CẤU HÌNH ---
MODEL_PATH = 'model.p'
CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng độ tin cậy để hiển thị (0.5 = 50%)

# --- 1. TẢI MÔ HÌNH ---
print("dang tai model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
    print("Model tai thanh cong!")
except FileNotFoundError:
    print(f"LOI: Khong tim thay file '{MODEL_PATH}'. Hay chay train_model.py truoc.")
    exit()
except Exception as e:
    print(f"LOI khi doc model: {e}")
    exit()

# --- 2. KHỞI TẠO MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Cấu hình phát hiện tay
hands = mp_hands.Hands(
    static_image_mode=False,      # Chế độ video (False) giúp tracking nhanh hơn
    max_num_hands=1,              # Chỉ nhận diện 1 tay (để tránh loạn)
    min_detection_confidence=0.5, # Độ nhạy phát hiện
    min_tracking_confidence=0.5   # Độ nhạy theo dõi
)

# --- 3. MỞ CAMERA ---
cap = cv2.VideoCapture(0)

# Cài đặt độ phân giải (nếu cần)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Da mo Camera. Nhan phim 'Q' de thoat.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong doc duoc frame tu Camera.")
        break

    # Lật ngược ảnh (hiệu ứng gương)
    frame = cv2.flip(frame, 1)
    
    H, W, _ = frame.shape
    
    # Chuyển đổi màu sang RGB (MediaPipe cần RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Xử lý hình ảnh
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Vẽ khung xương tay (Skeleton)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # 2. Thu thập tọa độ để dự đoán
            data_aux = []
            x_ = []
            y_ = []

            # Lấy tất cả tọa độ x, y để tính khung bao (Bounding Box)
            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            # Lấy gốc tọa độ là Cổ tay (điểm số 0)
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            # Tính toán tọa độ tương đối (Relative Coordinates)
            # Logic này PHẢI GIỐNG HỆT lúc bạn chạy process_data_csv.py
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - base_x)
                data_aux.append(hand_landmarks.landmark[i].y - base_y)

            # 3. Đưa vào Model để dự đoán
            # Model Scikit-Learn yêu cầu mảng 2 chiều [[...]]
            prediction = model.predict([np.asarray(data_aux)])
            
            # Lấy độ tin cậy (Confidence score)
            try:
                probs = model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(probs)
            except:
                confidence = 1.0 # Fallback nếu model không hỗ trợ proba

            predicted_char = str(prediction[0])

            # 4. Vẽ Giao diện đẹp (UI)
            # Tính toán tọa độ khung chữ nhật bao quanh tay
            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            # Giới hạn không vẽ ra ngoài màn hình
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            # Chỉ hiển thị nếu độ tin cậy cao hơn ngưỡng
            if confidence > CONFIDENCE_THRESHOLD:
                # Màu sắc dựa trên độ tin cậy (Xanh lá: Cao, Vàng: Trung bình)
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                
                # Vẽ khung bao
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Vẽ nền cho chữ (Header)
                cv2.rectangle(frame, (x1, y1 - 60), (x2, y1), color, -1)
                
                # Viết ký tự dự đoán (Chữ to, đậm)
                text_display = f"{predicted_char}"
                cv2.putText(frame, text_display, (x1 + 10, y1 - 15), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
                
                # Viết % độ tin cậy (Chữ nhỏ)
                conf_display = f"{int(confidence * 100)}%"
                cv2.putText(frame, conf_display, (x2 - 60, y1 - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Hiển thị cửa sổ
    cv2.imshow('AI Sign Language Translator (Level 3)', frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()
hands.close()