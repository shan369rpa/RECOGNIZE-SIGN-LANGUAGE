import cv2
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model # Dùng tf_keras nếu bạn cài thêm thư viện đó
import cvzone

# --- CÀI ĐẶT CHUNG ---
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.75  # Ngưỡng độ tin cậy để hiển thị kết quả
SMOOTHING_WINDOW = 10       # Số frame để lấy trung bình làm mượt

# --- KHỞI TẠO CÁC THỨ ---

# 1. Tải Model
try:
    # Nếu bạn cài 'tf_keras' thì đổi dòng này thành:
    # from tf_keras.models import load_model
    model = load_model(MODEL_PATH, compile=False)
    print("Model đã tải thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file model '{MODEL_PATH}'. Hãy đảm bảo file nằm cùng thư mục với script.")
    exit()
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit()

# 2. Đọc nhãn (tên các lớp - số 1, 2, 3...)
try:
    class_names = open(LABELS_PATH, "r").readlines()
    print("Nhãn đã được đọc thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file nhãn '{LABELS_PATH}'.")
    exit()

# 3. Khởi tạo Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Đặt độ phân giải nhỏ cho nhanh
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 4. Khởi tạo bộ đệm làm mượt và biến game
history = deque(maxlen=SMOOTHING_WINDOW)
last_correct_time = 0
question_num1 = 1
question_num2 = 1
current_answer = str(question_num1 + question_num2)
question_text = f"{question_num1} + {question_num2}"
score = 0
show_result_message = ""
message_timer = 0
MESSAGE_DURATION = 2 # Giây hiển thị thông báo đúng/sai

# --- VÒNG LẶP CHÍNH ---
while True:
    success, frame = cap.read()
    if not success:
        print("Lỗi: Không đọc được frame từ camera.")
        break

    # Lật ảnh camera cho giống nhìn gương
    frame = cv2.flip(frame, 1)
    
    # --- XỬ LÝ DỰ ĐOÁN ---
    # Chuẩn bị ảnh cho model
    image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    # Chuẩn hóa giá trị pixel về khoảng -1 đến 1
    image_normalized = (image_array / 127.5) - 1

    # Dự đoán
    try:
        prediction = model.predict(image_normalized, verbose=0) # verbose=0 để không in log dự đoán
        index = np.argmax(prediction)
        confidence = prediction[0][index]
        predicted_class_name = class_names[index].strip()
    except Exception as e:
        print(f"Lỗi dự đoán: {e}")
        predicted_class_name = "..."
        confidence = 0

    # --- LÀM MƯỢT KẾT QUẢ & LOGIC GAME ---
    current_time = time.time()

    # Chỉ thêm vào history nếu độ tin cậy đủ cao
    if confidence > CONFIDENCE_THRESHOLD:
        history.append(predicted_class_name)
    else:
        # Nếu độ tin cậy thấp, có thể xóa hết history hoặc chỉ thêm 1 ký tự placeholder
        # Ở đây mình không thêm gì, để kết quả cũ dần dần biến mất
        pass 

    # Lấy kết quả được dự đoán nhiều nhất trong cửa sổ làm mượt
    if history:
        final_decision = max(set(history), key=history.count)
    else:
        final_decision = "..." # Nếu chưa đủ dữ liệu hoặc không có gì được nhận diện
         # In ra để kiểm tra
    if final_decision != "...":
        print(f"May nhin thay: '{final_decision}' | Dap an can thiet: '{current_answer}'")
    is_correct = False
    # Logic hiển thị thông báo đúng/sai
    if final_decision == current_answer and confidence > 0.8: # Cần độ tin cậy cao để xác nhận
        is_correct = True
        show_result_message = "Dung roi! :)"
        message_color = (0, 255, 0)
        message_timer = current_time
        
        # Đặt câu hỏi mới sau 2 giây
        if current_time - last_correct_time > MESSAGE_DURATION:
            score += 1
            a = random.randint(1, 3) # Số ngón tay có thể giơ
            b = random.randint(1, 3) # Số ngón tay có thể giơ
            question_num1 = a
            question_num2 = b
            current_answer = str(a + b)
            question_text = f"{a} + {b}"
            last_correct_time = current_time
            history.clear() # Xóa history để bắt đầu câu mới
            
    elif current_time - message_timer > MESSAGE_DURATION and show_result_message != "":
        # Xóa thông báo sau MESSAGE_DURATION giây
        show_result_message = ""
        
    # Nếu người dùng vẫn giữ 1 cử chỉ cũ (không có gì thay đổi)
    elif final_decision != current_answer and show_result_message == "":
        # Có thể thêm thông báo nhắc nhở ở đây
        pass

    # --- VẼ GIAO DIỆN LÊN KHUNG HÌNH ---
    
    # 1. Hiển thị điểm số
    cvzone.putTextRect(frame, f"Diem: {score}", (20, 40), scale=1.5, thickness=2, colorT=(255,255,255), colorR=(0, 150, 255))

    # 2. Hiển thị câu hỏi
    cvzone.putTextRect(frame, f"Cau hoi: {question_text} = ?", (20, 100), scale=1.5, thickness=2, colorT=(255,255,255), colorR=(100, 0, 200))
    
    # 3. Hiển thị thông báo Đúng/Sai
    if show_result_message:
        cvzone.putTextRect(frame, show_result_message, (20, 180), scale=1.5, thickness=2, colorT=(0,0,0), colorR=message_color)

    # 4. Hiển thị kết quả dự đoán được làm mượt
    if final_decision != "...":
        cvzone.putTextRect(frame, f"Cu chi: {final_decision}", (20, 260), scale=2, thickness=2, colorT=(255,255,255), colorR=(255, 0, 255))

# 5. Thanh độ tin cậy (Tự vẽ bằng OpenCV)
    # Xác định màu sắc
    color = (0, 255, 0) if confidence > 0.8 else (0, 150, 255) if confidence > 0.6 else (0, 0, 255)
    
    # Vẽ chữ thông báo độ tin cậy
    cvzone.putTextRect(frame, f"Do tin cay: {int(confidence*100)}%", (20, 310), scale=1.5, thickness=2, offset=5)
    
    # Vẽ thanh nền (khung viền trắng)
    # Tọa độ: (20, 320) đến (340, 350) -> Rộng 320px, Cao 30px
    cv2.rectangle(frame, (20, 320), (340, 350), (255, 255, 255), 2)
    
    # Tính độ dài thanh màu bên trong
    bar_value = int(confidence * 320)
    
    # Vẽ thanh màu (đã fill đầy)
    # Lưu ý: Giới hạn không vẽ vượt quá khung
    if bar_value > 320: bar_value = 320
    cv2.rectangle(frame, (20, 320), (20 + bar_value, 350), color, -1)
        
    # Hiển thị frame đã vẽ
    cv2.imshow("AI Sign Language Math Game", frame)

    # Điều kiện thoát (nhấn phím 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- DỌN DẸP SAU KHI THOÁT ---
cap.release()
cv2.destroyAllWindows()
print("Đã thoát chương trình.")