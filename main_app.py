import cv2
import mediapipe as mp
import pickle
import numpy as np
import warnings
from PIL import ImageFont, ImageDraw, Image # Import thêm cái này

# --- HÀM HỖ TRỢ VIẾT TIẾNG VIỆT ---
def put_text_vietnamese(img, text, position, font_size, color):
    """
    img: Ảnh OpenCV (BGR)
    text: Nội dung chữ tiếng Việt
    position: Tọa độ (x, y)
    font_size: Cỡ chữ
    color: Màu chữ (B, G, R) - theo chuẩn OpenCV
    """
    # 1. Chuyển ảnh OpenCV (BGR) sang PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 2. Tải Font chữ (Windows thường có sẵn Arial)
    try:
        # Đường dẫn font trên Windows
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
    except IOError:
        # Nếu không tìm thấy thì dùng font mặc định (xấu hơn chút)
        font = ImageFont.load_default()
    
    # 3. Vẽ chữ (Lưu ý: PIL dùng màu RGB, OpenCV dùng BGR nên phải đổi lại màu)
    # color input là (Blue, Green, Red) -> Đổi thành (Red, Green, Blue)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    
    # 4. Chuyển lại từ PIL sang OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# --- TẮT CÁC CẢNH BÁO KHÔNG CẦN THIẾT ---
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- CẤU HÌNH ---
MODEL_PATH = 'model.p'
CONFIDENCE_THRESHOLD = 0.5

# --- 1. HỎI NGƯỜI DÙNG CHỌN CHẾ ĐỘ (TÍNH NĂNG MỚI) ---
print("\n" + "="*40)
print("  CÀI ĐẶT CHẾ ĐỘ NHẬN DIỆN  ")
print("="*40)
print("Lưu ý: Model hiện tại được train trên từng tay đơn lẻ.")
print(" - Chọn '1': Nhanh nhất, chính xác nhất (Khuyên dùng).")
print(" - Chọn '2': Nhận diện cả 2 tay cùng lúc (Có thể chậm hơn).")
print("-" * 40)

NUM_HANDS = 1 # Mặc định

while True:
    try:
        user_input = input(">> Bạn muốn nhận diện mấy tay? (Nhập 1 hoặc 2): ").strip()
        
        # Chuyển đổi thành số nguyên
        if user_input.isdigit():
            val = int(user_input)
            if val == 1 or val == 2:
                NUM_HANDS = val
                print(f"-> [OK] Đã kích hoạt chế độ: {NUM_HANDS} tay.")
                break
            else:
                print("Lỗi: Chỉ được nhập số 1 hoặc 2. Vui lòng nhập lại!")
        else:
            print("Lỗi: Vui lòng chỉ nhập con số (1 hoặc 2).")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

print("="*40)
print("Đang khởi động Camera...")

# --- 2. TẢI MÔ HÌNH ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']
except FileNotFoundError:
    print(f"LOI: Khong tim thay file '{MODEL_PATH}'. Hay chay train_model.py truoc.")
    exit()

# --- 3. KHỞI TẠO MEDIAPIPE (VỚI SỐ LƯỢNG TAY ĐÃ CHỌN) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=NUM_HANDS,      # <-- CẤU HÌNH SỐ LƯỢNG TAY TẠI ĐÂY
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 4. MỞ CAMERA ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Biến kiểm tra để hiển thị UI hướng dẫn
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        
        # Duyệt qua từng bàn tay (Nếu chọn 2 tay thì vòng lặp chạy tối đa 2 lần)
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            # --- A. VẼ KHUNG XƯƠNG ---
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # --- B. XỬ LÝ DỮ LIỆU ĐỂ DỰ ĐOÁN ---
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - base_x)
                data_aux.append(hand_landmarks.landmark[i].y - base_y)

            # --- C. DỰ ĐOÁN ---
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_char = str(prediction[0])
                
                probs = model.predict_proba([np.asarray(data_aux)])
                confidence = np.max(probs)
            except:
                predicted_char = "?"
                confidence = 0.0

            # --- D. VẼ UI KẾT QUẢ ---
            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if confidence > CONFIDENCE_THRESHOLD:
                color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                
                # Vẽ khung
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Vẽ nền chữ
                cv2.rectangle(frame, (x1, y1 - 60), (x2, y1), color, -1)
                
                # Hiện chữ cái
                cv2.putText(frame, predicted_char, (x1 + 10, y1 - 15), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
                
                # Hiện % tin cậy
                cv2.putText(frame, f"{int(confidence * 100)}%", (x2 - 60, y1 - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                # In ra Console (Dùng \r để không bị trôi dòng)
                # Nếu là 2 tay thì in thêm chỉ số tay
                hand_msg = f"[Tay {hand_idx + 1}]" if NUM_HANDS == 2 else "[Tay]"
                print(f"{hand_msg} Nhan dien: {predicted_char} ({int(confidence * 100)}%)      ", end="\r")

    # --- E. UI THÔNG MINH (HIỆN KHI KHÔNG THẤY TAY) ---
    if not hand_detected:
        # Làm tối màn hình
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Hộp thông báo
        center_x, center_y = W // 2, H // 2
        cv2.rectangle(frame, (center_x - 220, center_y - 120), (center_x + 220, center_y + 120), (255, 255, 255), 2)
        
        # Tiêu đề đỏ (TIẾNG VIỆT)
        frame = put_text_vietnamese(frame, "KHÔNG TÌM THẤY TAY!", (center_x - 190, center_y - 80), 35, (0, 0, 255))
        
        # Hướng dẫn (TIẾNG VIỆT)
        frame = put_text_vietnamese(frame, f"Chế độ: {NUM_HANDS} Tay", (center_x - 190, center_y - 30), 25, (0, 255, 255))
        frame = put_text_vietnamese(frame, "1. Đưa tay vào giữa khung hình", (center_x - 190, center_y + 10), 20, (255, 255, 255))
        frame = put_text_vietnamese(frame, "2. Đảm bảo đủ ánh sáng", (center_x - 190, center_y + 40), 20, (255, 255, 255))
        frame = put_text_vietnamese(frame, "3. Giữ khoảng cách ~50cm", (center_x - 190, center_y + 70), 20, (255, 255, 255))

    else:
        # Footer
        frame = put_text_vietnamese(frame, "Q: Thoát chương trình", (10, H - 30), 20, (200, 200, 200))

    cv2.imshow('AI Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()