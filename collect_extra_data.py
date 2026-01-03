import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

DATA_FILE = 'landmark_data.csv'
WINDOW_NAME = "Data Collection Mode"

def put_text_vietnamese(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except: 
        try: font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except: font = ImageFont.load_default()
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

target_label = sys.argv[1] if len(sys.argv) > 1 else ""
if not target_label:
    target_label = input(">> Nhập tên KÝ TỰ: ").strip()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21): header.extend([f'x{i}', f'y{i}'])
        writer.writerow(header)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- CẤU HÌNH CỬA SỔ RESIZABLE ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1000, 600)

counter = 0
last_save_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                data_row = [target_label]
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                for lm in hand_landmarks.landmark:
                    data_row.append(lm.x - base_x)
                    data_row.append(lm.y - base_y)
                with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(data_row)
                counter += 1
                last_save_time = time.time()

    if not hand_detected:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        cx, cy = W // 2, H // 2
        cv2.rectangle(frame, (cx - 300, cy - 160), (cx + 300, cy + 160), (255, 255, 255), 2)
        
        frame = put_text_vietnamese(frame, "KHÔNG TÌM THẤY TAY!", (cx - 220, cy - 110), 40, (0, 0, 255))
        frame = put_text_vietnamese(frame, "1. Đưa tay vào giữa khung hình", (cx - 240, cy - 30), 25, (255, 255, 255))
        frame = put_text_vietnamese(frame, "2. Xoay nhẹ cổ tay để đa dạng góc", (cx - 240, cy + 20), 25, (255, 255, 255))
        frame = put_text_vietnamese(frame, "3. Giữ khoảng cách ~50cm", (cx - 240, cy + 70), 25, (255, 255, 255))
        frame = put_text_vietnamese(frame, "➤ Nhấn phím 'Q' để Thoát về Menu", (cx - 240, cy + 120), 25, (200, 200, 200))

    else:
        cv2.rectangle(frame, (0, 0), (W, 70), (0, 0, 0), -1)
        frame = put_text_vietnamese(frame, f"Đang dạy: {target_label}", (20, 20), 30, (0, 255, 255))
        frame = put_text_vietnamese(frame, f"Số lượng: {counter}", (W - 250, 20), 30, (255, 255, 255))
        cv2.rectangle(frame, (0, H - 50), (W, H), (0, 0, 0), -1)
        frame = put_text_vietnamese(frame, "Giữ 'S': Lưu liên tục  |  Nhấn 'Q': Thoát về Menu", (30, H - 40), 20, (200, 200, 200))

        if time.time() - last_save_time < 0.5:
            frame = put_text_vietnamese(frame, "ĐÃ LƯU!", (W//2 - 100, H//2), 50, (0, 255, 0))

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()