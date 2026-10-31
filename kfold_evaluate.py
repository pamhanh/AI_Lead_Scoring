"""
kfold_evaluate.py

Bổ sung cho Giai đoạn 5 - dùng K-Fold Cross-Validation thay vì 1 lần chia
train/test, để so sánh Logistic Regression vs Decision Tree đáng tin cậy
hơn khi chỉ có ~33 mẫu (như đã bàn: test set 1 lần chỉ 7 mẫu, sai số ước
lượng quá lớn ~19 điểm % để kết luận model nào tốt hơn).

Cách k-fold hoạt động (k=5 mặc định):
  1. Chia toàn bộ 33 mẫu thành 5 phần gần bằng nhau (StratifiedKFold giữ
     đúng tỉ lệ Cold/Warm/Hot trong mỗi phần).
  2. Lặp 5 lần: mỗi lần lấy 1 phần làm test, 4 phần còn lại làm train.
  3. Mỗi mẫu trong 33 mẫu ĐỀU được dùng làm test đúng 1 lần (không có mẫu
     nào "may mắn" luôn nằm trong train hay luôn nằm trong test).
  4. Kết quả cuối = trung bình 5 lần accuracy -> ổn định hơn nhiều so với
     1 lần chia ngẫu nhiên.

Cách dùng:
    python kfold_evaluate.py

Yêu cầu: pip install scikit-learn pandas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "labeled.csv"
RANDOM_STATE = 42
N_FOLDS = 5

FEATURE_COLS = [
    "has_contact_form", "has_working_hours", "has_online_booking",
    "has_price_list", "has_blog", "has_blog_date", "blog_is_recent",
    "blog_updated_days",
]
TARGET_COL = "lead_label"
BLOG_DAYS_MISSING_SENTINEL = 9999


def load_data():
    df = pd.read_csv(INPUT_FILE)
    df = df[df[TARGET_COL] != "Không đủ điều kiện"].copy()

    for col in FEATURE_COLS:
        if col == "blog_updated_days":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(BLOG_DAYS_MISSING_SENTINEL)
        else:
            df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def main():
    X, y = load_data()
    n = len(X)
    print(f"Tổng số lead hợp lệ: {n}")
    print(f"Phân bố nhãn: {y.value_counts().to_dict()}")

    min_class_count = y.value_counts().min()
    k = min(N_FOLDS, min_class_count)
    if k < N_FOLDS:
        print(
            f"\nLưu ý: lớp ít mẫu nhất chỉ có {min_class_count} mẫu -> giảm "
            f"số fold xuống {k} (mỗi fold cần ít nhất 1 mẫu/lớp)."
        )
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    # Logistic Regression cần chuẩn hoá dữ liệu -> đóng gói scaler + model
    # thành 1 pipeline để mỗi fold tự fit scaler riêng trên đúng phần train
    # của fold đó (không rò rỉ thông tin từ fold test sang lúc chuẩn hoá).
    logreg_pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )
    tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=RANDOM_STATE)

    print(f"\n=== {k}-Fold Cross-Validation ===")

    logreg_scores = cross_val_score(logreg_pipeline, X, y, cv=skf, scoring="accuracy")
    tree_scores = cross_val_score(tree_clf, X, y, cv=skf, scoring="accuracy")

    def report(name, scores):
        print(f"\n{name}:")
        print(f"  Accuracy từng fold: {[round(s, 3) for s in scores]}")
        print(f"  Trung bình: {scores.mean():.3f}  (độ lệch chuẩn: {scores.std():.3f})")
        return scores.mean(), scores.std()

    mean_logreg, std_logreg = report("Logistic Regression", logreg_scores)
    mean_tree, std_tree = report("Decision Tree", tree_scores)

    print("\n=== So sánh (đáng tin cậy hơn bản 1-lần-chia-train/test) ===")
    print(f"  Logistic Regression: {mean_logreg:.3f} ± {std_logreg:.3f}")
    print(f"  Decision Tree:       {mean_tree:.3f} ± {std_tree:.3f}")

    diff = abs(mean_logreg - mean_tree)
    combined_std = (std_logreg + std_tree) / 2
    if diff < combined_std:
        print(
            f"\n-> Chênh lệch trung bình ({diff:.3f}) NHỎ HƠN độ lệch chuẩn trung bình "
            f"({combined_std:.3f}) -> 2 model KHÔNG khác biệt rõ rệt, sự chênh lệch quan "
            f"sát được nhiều khả năng chỉ do nhiễu mẫu nhỏ, không nên kết luận model nào "
            f"'tốt hơn' một cách chắc chắn với dữ liệu hiện tại."
        )
    else:
        better = "Logistic Regression" if mean_logreg > mean_tree else "Decision Tree"
        print(f"\n-> {better} có xu hướng nhỉnh hơn rõ rệt hơn qua {k} fold.")


if __name__ == "__main__":
    main()
