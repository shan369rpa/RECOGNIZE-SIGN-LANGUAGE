import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# 1. Đọc dữ liệu từ file CSV mới
print("Đang đọc dữ liệu...")
data = pd.read_csv('landmark_data.csv')

# Kiểm tra dữ liệu
if data.empty:
    print("Lỗi: File CSV trống! Hãy kiểm tra lại quá trình xử lý ảnh.")
    exit()

print(f"Tổng số mẫu dữ liệu: {len(data)}")
print(f"Các ký tự đã học: {data['label'].unique()}")

# 2. Tách dữ liệu
X = data.drop('label', axis=1) # Tọa độ (Input)
y = data['label']              # Nhãn (Output - A, B, C...)

# Chia tập train/test (80% học, 20% thi)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 3. Huấn luyện (Dùng Random Forest - Cực nhanh và chính xác)
print("Đang huấn luyện mô hình...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Đánh giá
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f"--------------------------------")
print(f"ĐỘ CHÍNH XÁC: {score * 100:.2f}%")
print(f"--------------------------------")

# 5. Lưu mô hình
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f) # Lưu dưới dạng từ điển để an toàn hơn

print("Đã lưu mô hình vào file 'model.p'. Xong!")