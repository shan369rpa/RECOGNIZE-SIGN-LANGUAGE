import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Thư viện thanh tiến độ
import time

# --- CẤU HÌNH ---
DATA_PATH = 'landmark_data.csv'
MODEL_PATH = 'model.p'
N_ESTIMATORS = 100 # Số lượng cây trong rừng (càng nhiều càng chính xác nhưng lâu hơn)

print("\n" + "="*50)
print("  HUẤN LUYỆN MÔ HÌNH AI (TRAINING)  ")
print("="*50)

# 1. ĐỌC DỮ LIỆU
print(f"[1/4] Đang đọc dữ liệu từ '{DATA_PATH}'...")
try:
    data = pd.read_csv(DATA_PATH)
    if data.empty:
        print("❌ Lỗi: File dữ liệu trống!")
        exit()
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{DATA_PATH}'")
    exit()

# Hiển thị thống kê nhỏ
num_samples = len(data)
num_classes = len(data['label'].unique())
print(f"   -> Tìm thấy {num_samples} mẫu dữ liệu.")
print(f"   -> Bao gồm {num_classes} nhãn (ký tự): {data['label'].unique()}")

# 2. CHUẨN BỊ DỮ LIỆU
print(f"[2/4] Đang chia tập dữ liệu (Train/Test)...")
X = data.drop('label', axis=1)
y = data['label']

# Chia 80% học, 20% thi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"   -> Train set: {len(X_train)} mẫu | Test set: {len(X_test)} mẫu")

# 3. HUẤN LUYỆN (CÓ THANH TIẾN ĐỘ)
print(f"[3/4] Bắt đầu huấn luyện ({N_ESTIMATORS} cây)...")

# Cấu hình Warm Start để train từng bước
model = RandomForestClassifier(
    n_estimators=0,      # Bắt đầu với 0 cây
    warm_start=True,     # Cho phép giữ lại kết quả cũ để train tiếp
    n_jobs=-1,           # Dùng tất cả nhân CPU
    random_state=42
)

# Vòng lặp train với thanh tqdm
with tqdm(total=N_ESTIMATORS, desc="   -> Tiến độ", unit="tree", ncols=100) as pbar:
    for i in range(N_ESTIMATORS):
        # Tăng thêm 1 cây vào rừng
        model.n_estimators += 1
        
        # Train tiếp (chỉ train cây mới thêm vào)
        # .values để tránh cảnh báo UserWarning về feature names
        model.fit(X_train.values, y_train)
        
        # Cập nhật thanh tiến độ
        pbar.update(1)
        # time.sleep(0.01) # (Tùy chọn) Bỏ comment nếu máy chạy quá nhanh không kịp nhìn :D

# 4. ĐÁNH GIÁ VÀ LƯU
print(f"[4/4] Đang đánh giá và lưu model...")

y_pred = model.predict(X_test.values)
score = accuracy_score(y_test, y_pred)
accuracy_percent = score * 100

print("-" * 50)
print(f"✅ ĐỘ CHÍNH XÁC: {accuracy_percent:.2f}%")
print("-" * 50)

if accuracy_percent < 50:
    print("⚠️  Cảnh báo: Độ chính xác thấp. Hãy thu thập thêm dữ liệu!")

# Lưu model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"💾 Đã lưu model vào: {MODEL_PATH}")
print("="*50)
print("HOÀN TẤT! Bây giờ bạn có thể chạy 'python main_app.py'")