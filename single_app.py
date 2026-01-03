import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog
import cv2
import mediapipe as mp
import csv
import os
import time
import sys
import pickle
import threading
import numpy as np
import io
from PIL import ImageFont, ImageDraw, Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # <--- Import thư viện TQDM
import warnings

# --- CẤU HÌNH HỆ THỐNG ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

CONST_DATA_FILE = 'landmark_data.csv'
CONST_MODEL_FILE = 'model.p'
CONST_FONT_PATH = "arial.ttf"

# =============================================================================
# CLASS 1: BASE HAND APP (Lớp Cha - Chứa logic vẽ UI đẹp)
# =============================================================================
class BaseHandApp:
    def __init__(self, window_name="Camera App", num_hands=1):
        self.window_name = window_name
        self.num_hands = num_hands
        self.is_running = False
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def draw_vietnamese_text(self, img, text, pos, size, color):
        """Vẽ chữ tiếng Việt dùng Pillow"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try: font = ImageFont.truetype(CONST_FONT_PATH, size)
        except: 
            try: font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
            except: font = ImageFont.load_default()
        
        rgb_color = (color[2], color[1], color[0])
        draw.text(pos, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def draw_modern_ui(self, frame, header_text, footer_text):
        """Vẽ Header và Footer trong suốt (Glassmorphism)"""
        H, W, _ = frame.shape
        overlay = frame.copy()
        
        # Header & Footer Background
        cv2.rectangle(overlay, (0, 0), (W, 70), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, H - 50), (W, H), (0, 0, 0), -1)
        
        # Alpha blending
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text
        frame = self.draw_vietnamese_text(frame, header_text, (20, 20), 30, (0, 255, 255))
        frame = self.draw_vietnamese_text(frame, footer_text, (20, H - 35), 20, (200, 200, 200))
        return frame

    def draw_instruction_box(self, frame, title, lines):
        """Vẽ hộp hướng dẫn CĂN GIỮA, KHÔNG BỊ CẮT CHỮ"""
        H, W, _ = frame.shape
        
        # 1. Làm tối màn hình
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 2. Tính toán hộp giữa (Rộng 640px, Cao 360px)
        box_w, box_h = 640, 360
        cx, cy = W // 2, H // 2
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2

        # 3. Vẽ hộp
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2) # Viền trắng
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)      # Nền đen

        # 4. Viết Tiêu đề (Căn giữa theo chiều ngang của hộp)
        # Tự ước lượng vị trí x để căn giữa: title_x = x1 + (box_w - text_width) / 2
        # Ở đây set cứng tương đối
        frame = self.draw_vietnamese_text(frame, title, (x1 + 40, y1 + 30), 35, (0, 100, 255))

        # 5. Viết các dòng hướng dẫn
        start_y = y1 + 100
        for i, line in enumerate(lines):
            frame = self.draw_vietnamese_text(frame, line, (x1 + 50, start_y + i*45), 22, (255, 255, 255))

        # 6. Dòng Footer trong hộp
        frame = self.draw_vietnamese_text(frame, "➤ Nhấn phím 'Q' để Thoát về Menu", (x1 + 50, y2 - 40), 20, (150, 150, 150))
        
        return frame

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1000, 600)
        self.is_running = True

    def stop_camera(self):
        self.is_running = False
        if self.cap: self.cap.release()
        cv2.destroyWindow(self.window_name)

    def get_landmarks_list(self, hand_landmarks):
        data_aux = []
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - base_x)
            data_aux.append(lm.y - base_y)
        return data_aux

# =============================================================================
# CLASS 2: DATA COLLECTOR
# =============================================================================
class DataCollector(BaseHandApp):
    def __init__(self, target_label):
        super().__init__(window_name="Thu Thap Du Lieu", num_hands=1)
        self.target_label = target_label
        self.counter = 0
        self.last_save_time = 0
        
        if not os.path.exists(CONST_DATA_FILE):
            with open(CONST_DATA_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ['label'] + [f'{a}{i}' for i in range(21) for a in ['x', 'y']]
                writer.writerow(header)

    def run(self):
        self.start_camera()
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        data_row = [self.target_label] + self.get_landmarks_list(hand_landmarks)
                        with open(CONST_DATA_FILE, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow(data_row)
                        self.counter += 1
                        self.last_save_time = time.time()

            # --- GIAO DIỆN CHUẨN ĐẸP ---
            if not hand_detected:
                frame = self.draw_instruction_box(frame, 
                    title="KHÔNG TÌM THẤY TAY!",
                    lines=[
                        "1. Đưa tay vào vùng giữa camera.",
                        "2. Giữ khoảng cách 40-50cm.",
                        "3. Xoay nhẹ cổ tay để lấy nhiều góc.",
                        "4. Đảm bảo ánh sáng tốt."
                    ]
                )
            else:
                frame = self.draw_modern_ui(frame, 
                    header_text=f"Đang dạy: {self.target_label}", 
                    footer_text="Giữ 'S' để LƯU mẫu  |  Nhấn 'Q' để THOÁT"
                )
                
                # Số lượng mẫu
                frame = self.draw_vietnamese_text(frame, f"Số mẫu: {self.counter}", (W - 220, 20), 25, (255, 255, 255))
                
                # Hiệu ứng SAVED
                if time.time() - self.last_save_time < 0.5:
                    cx, cy = W // 2, H // 2
                    frame = self.draw_vietnamese_text(frame, "ĐÃ LƯU!", (cx - 100, cy), 60, (0, 255, 0))

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.stop_camera()

# =============================================================================
# CLASS 3: SIGN DETECTOR
# =============================================================================
class SignDetector(BaseHandApp):
    def __init__(self, num_hands=1):
        super().__init__(window_name="Nhan Dien Thu Ngu AI", num_hands=num_hands)
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            with open(CONST_MODEL_FILE, 'rb') as f:
                self.model = pickle.load(f)['model']
        except: pass

    def run(self):
        if not self.model:
            messagebox.showerror("Lỗi", "Chưa có Model! Hãy huấn luyện trước.")
            return

        self.start_camera()
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    data_aux = self.get_landmarks_list(hand_landmarks)
                    try:
                        char = str(self.model.predict([np.asarray(data_aux)])[0])
                        conf = np.max(self.model.predict_proba([np.asarray(data_aux)]))
                    except: char, conf = "?", 0.0

                    x_vals = [lm.x for lm in hand_landmarks.landmark]
                    y_vals = [lm.y for lm in hand_landmarks.landmark]
                    x1, y1 = max(0, int(min(x_vals)*W)-20), max(0, int(min(y_vals)*H)-20)
                    x2, y2 = min(W, int(max(x_vals)*W)+20), min(H, int(max(y_vals)*H)+20)
                    
                    if conf > 0.5:
                        color = (0,255,0) if conf > 0.8 else (0,255,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
                        cv2.rectangle(frame, (x1,y1-60), (x2,y1), color, -1)
                        frame = self.draw_vietnamese_text(frame, char, (x1+10, y1-55), 40, (0,0,0))
                        cv2.putText(frame, f"{int(conf*100)}%", (x2-60, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

            # --- GIAO DIỆN CHUẨN ĐẸP ---
            if not hand_detected:
                frame = self.draw_instruction_box(frame,
                    title="CHƯA PHÁT HIỆN TAY!",
                    lines=[
                        f"Chế độ hiện tại: {self.num_hands} Tay.",
                        "1. Đưa tay vào vùng giữa camera.",
                        "2. Giữ khoảng cách (50cm).",
                        "3. Tránh ngược sáng."
                    ]
                )
            else:
                frame = self.draw_modern_ui(frame, 
                    header_text="AI SIGN LANGUAGE TRANSLATOR", 
                    footer_text="Nhấn phím 'Q' để Thoát chương trình"
                )

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.stop_camera()

# =============================================================================
# CLASS 4: MODEL TRAINER (Đã thêm TQDM)
# =============================================================================
class ModelTrainer:
    def __init__(self, log_callback):
        self.log = log_callback

    def train(self):
        self.log("="*30, "INFO")
        self.log("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN...", "INFO")
        
        try:
            import pandas as pd
            # 1. Đọc dữ liệu
            try:
                data = pd.read_csv(CONST_DATA_FILE, on_bad_lines='skip')
            except FileNotFoundError:
                self.log(f"❌ Không tìm thấy file {CONST_DATA_FILE}", "ERROR")
                return

            if data.empty:
                self.log("❌ File dữ liệu trống! Hãy thu thập trước.", "ERROR")
                return

            samples = len(data)
            try:
                classes_list = data['label'].unique()
                classes_count = len(classes_list)
            except KeyError:
                self.log("❌ File dữ liệu lỗi (Thiếu cột label)", "ERROR")
                return

            self.log(f"-> Tìm thấy: {samples} mẫu dữ liệu", "INFO")
            self.log(f"-> Số lượng từ vựng: {classes_count}", "INFO")

            # 2. Chuẩn bị
            X = data.drop('label', axis=1)
            y = data['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.log(f"-> Đang huấn luyện Random Forest...", "WARNING")
            
            # 3. Train với TQDM (Sử dụng Warm Start để cập nhật từng cây)
            # Khởi tạo model với warm_start=True
            model = RandomForestClassifier(n_estimators=0, warm_start=True, n_jobs=-1, random_state=42)
            total_trees = 100
            
            # Sử dụng tqdm để tạo vòng lặp (nhưng ta in log ra GUI chứ không in console)
            for i in tqdm(range(total_trees), desc="Training"):
                model.n_estimators += 1
                # Dùng .values để tránh warning sklearn
                model.fit(X_train.values, y_train)
                
                # Cập nhật GUI mỗi 5% (5 cây)
                if (i+1) % 5 == 0:
                    percent = i + 1
                    self.log(f"⏳ Đang huấn luyện... {percent}%", "WARNING")
                    time.sleep(0.01) # Nghỉ xíu để UI kịp vẽ

            # 4. Đánh giá
            acc = accuracy_score(y_test, model.predict(X_test.values)) * 100
            self.log(f"✅ HUẤN LUYỆN HOÀN TẤT!", "SUCCESS")
            self.log(f"🎯 ĐỘ CHÍNH XÁC: {acc:.2f}%", "SUCCESS")

            with open(CONST_MODEL_FILE, 'wb') as f:
                pickle.dump({'model': model}, f)
            
            self.log(f"💾 Đã lưu model vào '{CONST_MODEL_FILE}'", "SUCCESS")

        except Exception as e:
            self.log(f"❌ LỖI NGHIÊM TRỌNG: {e}", "ERROR")

# =============================================================================
# CLASS 5: MAIN GUI
# =============================================================================
class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI SIGN LANGUAGE SYSTEM - PRO")
        self.root.geometry("900x650")
        self.root.configure(bg="#2C3E50")

        # Header
        header_frame = tk.Frame(root, bg="#2C3E50")
        header_frame.pack(pady=20)
        tk.Label(header_frame, text="PHẦN MỀM HỖ TRỢ NGƯỜI KHIẾM THÍNH", font=("Arial", 14), fg="#95A5A6", bg="#2C3E50").pack()
        tk.Label(header_frame, text="AI SIGN LANGUAGE TRANSLATOR", font=("Arial", 24, "bold"), fg="#ECF0F1", bg="#2C3E50").pack()

        # Buttons Frame
        btn_frame = tk.Frame(root, bg="#2C3E50")
        btn_frame.pack(pady=20)

        # Buttons
        self.btn_detect = tk.Button(btn_frame, text="📷 NHẬN DIỆN (DETECT)", font=("Arial", 11, "bold"), 
                                    bg="#3498DB", fg="white", width=22, height=2, command=self.on_detect_click)
        self.btn_detect.grid(row=0, column=0, padx=15)

        self.btn_collect = tk.Button(btn_frame, text="➕ THU THẬP (COLLECT)", font=("Arial", 11, "bold"), 
                                     bg="#E67E22", fg="white", width=22, height=2, command=self.on_collect_click)
        self.btn_collect.grid(row=0, column=1, padx=15)

        self.btn_train = tk.Button(btn_frame, text="🧠 HUẤN LUYỆN MODEL (TRAIN)", font=("Arial", 13, "bold"), 
                                   bg="#27AE60", fg="white", width=48, height=2, command=self.on_train_click)
        self.btn_train.grid(row=1, column=0, columnspan=2, pady=25)

        # Logs
        log_frame = tk.Frame(root, bg="#2C3E50")
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        tk.Label(log_frame, text="NHẬT KÝ HỆ THỐNG:", fg="#F1C40F", bg="#2C3E50", font=("Consolas", 10, "bold")).pack(anchor="w")
        
        self.log_area = scrolledtext.ScrolledText(log_frame, height=10, bg="#1E272E", fg="#ECF0F1", font=("Consolas", 10))
        self.log_area.pack(fill="both", expand=True)
        
        # Tags màu
        self.log_area.tag_config("INFO", foreground="#ECF0F1")
        self.log_area.tag_config("SUCCESS", foreground="#2ECC71")
        self.log_area.tag_config("WARNING", foreground="#F1C40F")
        self.log_area.tag_config("ERROR", foreground="#E74C3C")
        
        self.log(">> Hệ thống đã sẵn sàng...", "INFO")

    def log(self, msg, tag="INFO"):
        self.log_area.insert(tk.END, str(msg) + "\n", tag)
        self.log_area.see(tk.END)

    def on_detect_click(self):
        choice_win = tk.Toplevel(self.root)
        choice_win.title("Chọn chế độ")
        choice_win.geometry("300x150")
        tk.Label(choice_win, text="Bạn muốn nhận diện mấy tay?", font=("Arial", 11)).pack(pady=10)
        
        def run(n):
            choice_win.destroy()
            def thread_task():
                self.btn_detect.config(state="disabled")
                app = SignDetector(num_hands=n)
                app.run()
                self.btn_detect.config(state="normal")
            threading.Thread(target=thread_task).start()

        tk.Button(choice_win, text="1 Tay (Chính xác)", bg="#2ECC71", width=15, command=lambda: run(1)).pack(pady=5)
        tk.Button(choice_win, text="2 Tay (Thử nghiệm)", bg="#F39C12", width=15, command=lambda: run(2)).pack(pady=5)

    def on_collect_click(self):
        label = simpledialog.askstring("Thu thập", "Nhập tên Ký tự / Cử chỉ muốn dạy:")
        if not label: return
        
        def thread_task():
            self.btn_collect.config(state="disabled")
            app = DataCollector(target_label=label)
            app.run()
            self.btn_collect.config(state="normal")
            self.log(f"Đã thu thập xong nhãn: {label}", "SUCCESS")
        threading.Thread(target=thread_task).start()

    def on_train_click(self):
        def thread_task():
            self.btn_train.config(state="disabled", text="⏳ ĐANG CHẠY...", bg="#7F8C8D")
            trainer = ModelTrainer(self.run_log_on_main)
            trainer.train()
            self.btn_train.config(state="normal", text="🧠 HUẤN LUYỆN MODEL (TRAIN)", bg="#27AE60")
        threading.Thread(target=thread_task).start()

    def run_log_on_main(self, msg, tag):
        self.root.after(0, lambda: self.log(msg, tag))

if __name__ == "__main__":
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()