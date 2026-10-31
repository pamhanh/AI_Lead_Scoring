"""
train_and_evaluate.py

Giai đoạn 5 - Huấn luyện & đánh giá Model.

  5.1 Chia train/test 80/20 (stratify theo nhãn để giữ tỉ lệ Cold/Warm/Hot)
  5.2 Train Logistic Regression
  5.3 Train Decision Tree (criterion='entropy'), vẽ cây, in Information Gain
      từng node
  5.4 Đánh giá cả 2 model: accuracy, precision/recall, confusion matrix
  5.5 So sánh 2 model

QUAN TRỌNG (ghi vào báo cáo, mục "giới hạn phương pháp"):
Vì lead_label được tính TRỰC TIẾP từ chính các đặc trưng này qua 1 công thức
tuyến tính có ngưỡng (bootstrap rule ở Giai đoạn 4), nên 2 model ở đây thực
chất đang HỌC LẠI (approximate) công thức thủ công đó, không phải học từ
hành vi mua hàng thực tế (vì không có dữ liệu thực tế đó). Điều này là bản
chất của "bootstrap labeling" - mục tiêu không phải để có độ chính xác cao
bất ngờ, mà để chứng minh: (1) pipeline ML hoạt động đúng end-to-end, và
(2) model có thể tổng quát hoá để chấm điểm công ty MỚI mà không cần chạy
lại thủ công bộ luật mỗi lần - đây mới là giá trị thực tế khi tích hợp vào
n8n ở Giai đoạn 6.

Cách dùng:
    python train_and_evaluate.py

Yêu cầu: pip install scikit-learn matplotlib pandas
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

INPUT_FILE = "labeled.csv"
RANDOM_STATE = 42

FEATURE_COLS = [
    "has_contact_form", "has_working_hours", "has_online_booking",
    "has_price_list", "has_blog", "has_blog_date", "blog_is_recent",
    "blog_updated_days",
]
TARGET_COL = "lead_label"
BLOG_DAYS_MISSING_SENTINEL = 9999  # không xác định được ngày blog -> coi như "rất cũ"


def load_data():
    df = pd.read_csv(INPUT_FILE)
    # Bỏ các công ty không đủ điều kiện (đã loại ở Giai đoạn 4, không có lead_score)
    df = df[df[TARGET_COL] != "Không đủ điều kiện"].copy()

    for col in FEATURE_COLS:
        if col == "blog_updated_days":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(BLOG_DAYS_MISSING_SENTINEL)
        else:
            df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, df


def evaluate(name, y_test, y_pred):
    print(f"\n=== {name} ===")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("\nPrecision/Recall/F1 (per class):")
    print(classification_report(y_test, y_pred, zero_division=0))
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"Confusion matrix (labels={labels}):")
    print(pd.DataFrame(cm, index=[f"thực:{l}" for l in labels], columns=[f"đoán:{l}" for l in labels]))
    return acc


def print_tree_information_gain(tree_clf, feature_names):
    """Bước 5.3: in Entropy/Information Gain cho từng node nội bộ của cây."""
    t = tree_clf.tree_
    print("\n--- Entropy/Information Gain từng node (Decision Tree) ---")
    for node_id in range(t.node_count):
        if t.children_left[node_id] == t.children_right[node_id]:  # leaf node
            continue
        feat = feature_names[t.feature[node_id]]
        thresh = t.threshold[node_id]
        parent_entropy = t.impurity[node_id]
        n_parent = t.n_node_samples[node_id]

        left, right = t.children_left[node_id], t.children_right[node_id]
        n_left, n_right = t.n_node_samples[left], t.n_node_samples[right]
        weighted_child_entropy = (
            n_left / n_parent * t.impurity[left] + n_right / n_parent * t.impurity[right]
        )
        info_gain = parent_entropy - weighted_child_entropy

        print(
            f"  Node {node_id}: split trên '{feat}' <= {thresh:.2f} "
            f"| entropy cha = {parent_entropy:.3f} (n={n_parent}) "
            f"| entropy con (weighted) = {weighted_child_entropy:.3f} "
            f"| Information Gain = {info_gain:.3f}"
        )


def main():
    X, y, df = load_data()
    print(f"Tổng số lead hợp lệ dùng để train/test: {len(df)}")
    print(f"Phân bố nhãn: {y.value_counts().to_dict()}")

    # --- 5.1: chia train/test 80/20 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # --- 5.2: Logistic Regression ---
    # chuẩn hoá dữ liệu trước khi đưa vào Logistic Regression (blog_updated_days
    # có scale rất khác các cột 0/1 còn lại -> cần StandardScaler để hội tụ tốt)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train_scaled, y_train)
    y_pred_logreg = logreg.predict(X_test_scaled)
    acc_logreg = evaluate("Logistic Regression", y_test, y_pred_logreg)

    # --- 5.3: Decision Tree (criterion=entropy) ---
    tree_clf = DecisionTreeClassifier(
        criterion="entropy", max_depth=4, random_state=RANDOM_STATE
    )
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    acc_tree = evaluate("Decision Tree", y_test, y_pred_tree)

    print_tree_information_gain(tree_clf, FEATURE_COLS)

    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_clf, feature_names=FEATURE_COLS, class_names=tree_clf.classes_,
        filled=True, rounded=True, fontsize=9,
    )
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=150)
    print("\n-> decision_tree.png (hình cây quyết định)")

    # --- 5.5: so sánh 2 model ---
    print("\n=== Bước 5.5: So sánh 2 model ===")
    print(f"  Logistic Regression accuracy: {acc_logreg:.3f}")
    print(f"  Decision Tree accuracy:       {acc_tree:.3f}")
    better = "Logistic Regression" if acc_logreg > acc_tree else (
        "Decision Tree" if acc_tree > acc_logreg else "Ngang nhau"
    )
    print(f"  -> Model tốt hơn trên tập test này: {better}")
    print(
        "\n  Lưu ý: với chỉ ~33 mẫu, kết quả accuracy dễ dao động mạnh tuỳ lần "
        "chia train/test - nên báo cáo cả 2 model thay vì chỉ chọn 1, đúng "
        "tinh thần bước 5.5 'giữ cả 2 để đối chiếu trong báo cáo'."
    )


if __name__ == "__main__":
    main()
