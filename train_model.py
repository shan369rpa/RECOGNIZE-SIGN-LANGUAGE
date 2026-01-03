import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Thư viện thanh tiến độ
import time
import sys
import io
# --- ÉP BUỘC UTF-8 CHO FILE EXE ---
# Thêm đoạn này vào đầu file để sửa lỗi Unicode trên Windows Console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# --- CẤU HÌNH ---
DATA_PATH = 'landmark_data.csv'
MODEL_PATH = 'model.p'
N_ESTIMATORS = 100 

print("\n" + "="*50)
print("  HUẤN LUYỆN MÔ HÌNH AI (TRAINING)  ")
print("="*50)

# 1. ĐỌC DỮ LIỆU (ĐÃ FIX LỖI CRASH)
print(f"[1/4] Đang đọc dữ liệu từ '{DATA_PATH}'...")
try:
    # on_bad_lines='skip': Tự động bỏ qua các dòng lỗi (dòng có 169 cột)
    data = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    
    if data.empty:
        print("❌ Lỗi: File dữ liệu trống!")
        exit()
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file '{DATA_PATH}'")
    exit()
except Exception as e:
    print(f"❌ Lỗi không xác định: {e}")
    exit()

# Hiển thị thống kê
num_samples = len(data)
try:
    num_classes = len(data['label'].unique())
    print(f"   -> Đã đọc thành công: {num_samples} mẫu.")
    print(f"   -> Số lượng nhãn: {num_classes} ({data['label'].unique()})")
except KeyError:
    print("❌ Lỗi: File CSV không có cột 'label'. Hãy kiểm tra lại file dữ liệu.")
    exit()

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

model = RandomForestClassifier(
    n_estimators=0,      
    warm_start=True,     
    n_jobs=-1,           
    random_state=42
)

# Vòng lặp train với thanh tqdm
with tqdm(total=N_ESTIMATORS, desc="   -> Tiến độ", unit="tree", ncols=100, colour='green',file=sys.stdout) as pbar:
    for i in range(N_ESTIMATORS):
        model.n_estimators += 1
        model.fit(X_train.values, y_train)
        pbar.update(1)

# 4. ĐÁNH GIÁ VÀ LƯU
print(f"[4/4] Đang đánh giá và lưu model...")

y_pred = model.predict(X_test.values)
score = accuracy_score(y_test, y_pred)
accuracy_percent = score * 100

print("-" * 50)
print(f"✅ ĐỘ CHÍNH XÁC: {accuracy_percent:.2f}%")
print("-" * 50)

# Lưu model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"💾 Đã lưu model vào: {MODEL_PATH}")
print("="*50)
print("HOÀN TẤT! Bây giờ bạn có thể chạy 'python main_app.py'")