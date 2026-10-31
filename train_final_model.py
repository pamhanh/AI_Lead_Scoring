"""
train_final_model.py

Giai đoạn 6, bước 6.1 (phần chuẩn bị): train Logistic Regression LẦN CUỐI
trên TOÀN BỘ 33 lead hợp lệ (không chia train/test nữa) và lưu model +
scaler ra file, để API không phải train lại mỗi lần có request.

Lý do dùng Logistic Regression (không phải Decision Tree): kết quả 5-fold
CV ở Giai đoạn 5 cho thấy Logistic Regression ổn định hơn hẳn (độ lệch
chuẩn 0.076 so với 0.166 của Decision Tree), dù trung bình accuracy không
khác biệt có ý nghĩa thống kê.

Lý do train trên TOÀN BỘ dữ liệu (không giữ lại test set): ở giai đoạn
train/test hay k-fold, mục đích là SO SÁNH và ĐÁNH GIÁ model. Bây giờ đã
chọn xong model (Logistic Regression), mục đích chuyển sang "dùng thật" -
nên tận dụng tối đa dữ liệu để model học được nhiều nhất trước khi đưa vào
phục vụ (không cần giữ lại phần nào để "thi" nữa, vì bài thi đã xong ở
Giai đoạn 5).

Cách dùng:
    python train_final_model.py

Output: model.pkl, scaler.pkl (dùng bởi app.py)
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "labeled.csv"
TARGET_COL = "lead_label"
BLOG_DAYS_MISSING_SENTINEL = 9999

# QUAN TRỌNG: đúng thứ tự và tên cột như lúc train/đánh giá ở Giai đoạn 5
# (train_and_evaluate.py / kfold_evaluate.py) - app.py phải tạo feature
# vector theo ĐÚNG thứ tự này khi gọi model.predict(), nếu không kết quả
# sẽ sai mà không báo lỗi gì (model vẫn chạy được, chỉ là hiểu sai ý nghĩa
# từng cột).
FEATURE_COLS = [
    "has_contact_form", "has_working_hours", "has_online_booking",
    "has_price_list", "has_blog", "has_blog_date", "blog_is_recent",
    "blog_updated_days",
]


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df[df[TARGET_COL] != "Không đủ điều kiện"].copy()

    for col in FEATURE_COLS:
        if col == "blog_updated_days":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(BLOG_DAYS_MISSING_SENTINEL)
        else:
            df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(FEATURE_COLS, "feature_cols.pkl")  # lưu kèm thứ tự cột, tránh app.py lệch thứ tự

    train_acc = model.score(X_scaled, y)
    print(f"Đã train xong trên toàn bộ {len(df)} mẫu.")
    print(f"Accuracy trên chính tập train (chỉ để tham khảo, KHÔNG phải đánh giá thật): {train_acc:.3f}")
    print("-> model.pkl, scaler.pkl, feature_cols.pkl")


if __name__ == "__main__":
    main()
