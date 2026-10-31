"""
bootstrap_labeling.py

Giai đoạn 4 - Bootstrap Labeling.

--- 4.1: Bộ luật chấm điểm (đã sửa sau khi thảo luận kỹ về ý nghĩa business) ---

QUAN TRỌNG - lý do đổi cách tính so với bản đầu tiên:
Bản đầu tiên chấm "điểm cao = website càng đầy đủ tính năng" rồi dùng điểm
đó để tìm khách hàng cho dịch vụ automation. Đây là SAI LOGIC kiểu thiên
kiến sống sót: công ty đã có sẵn online booking + blog cập nhật đều thì
KHÔNG còn nhu cầu mua automation nữa - chấm họ "Hot" là gõ nhầm cửa.

Cách tính mới - 2 tầng:

  TẦNG 1 - GATE (điều kiện bắt buộc, không phải điểm số):
    Phải có has_phone VÀ has_address mới được coi là 1 lead hợp lệ (bằng
    chứng đây là doanh nghiệp thật, liên hệ được). Thiếu 1 trong 2 -> loại
    khỏi danh sách hoàn toàn, gán nhãn "Không đủ điều kiện" (KHÔNG phải
    "Cold" - vì "Cold" vẫn ngụ ý là 1 lead hợp lệ, chỉ ít tiềm năng hơn).

  TẦNG 2 - GAP SCORE (chỉ tính trên các công ty đã qua gate), 2 nhóm:
    a) Nhóm "uy tín cơ bản" - tính THUẬN, weight thấp:
         has_contact_form   +1
         has_working_hours  +1
       (thông tin cơ bản, không liên quan trực tiếp automation, ít phân hoá)

    b) Nhóm "công cụ automation nâng cao" - tính NGHỊCH, THIẾU mới cộng điểm:
         KHÔNG có has_online_booking  +3
         KHÔNG có has_price_list      +2
         KHÔNG blog_is_recent         +3
       (đây là các tính năng dịch vụ automation có thể bán được -> thiếu =
       có cơ hội bán, không phải thiếu = kém)

    Lưu ý: KHÔNG dùng has_blog riêng trong công thức (chỉ dùng
    blog_is_recent) để tránh đếm trùng 1 tín hiệu 2 lần - vì has_blog=False
    thì blog_is_recent chắc chắn cũng False.

  Điểm tối đa: 1+1+3+2+3 = 10
  Điểm càng cao = doanh nghiệp thật (đã qua gate) NHƯNG thiếu nhiều công cụ
  automation -> cơ hội bán tốt nhất.

--- 4.3: Nhãn phân loại (Hot/Warm/Cold) ---
Dùng ngưỡng tam phân vị của phân bố điểm (chỉ tính trên các công ty đã qua
gate) để tự động tránh lệch nhóm quá nhiều.

Cách dùng:
    python bootstrap_labeling.py
"""

import csv
from collections import Counter

INPUT_FILE = "features.csv"
OUTPUT_FILE = "labeled.csv"

GATE_FIELDS = ["has_phone", "has_address"]

BASIC_WEIGHTS = {  # tính thuận: có -> cộng điểm
    "has_contact_form": 1,
    "has_working_hours": 1,
}

GAP_WEIGHTS = {  # tính nghịch: KHÔNG có -> cộng điểm
    "has_online_booking": 3,
    "has_price_list": 2,
    "blog_is_recent": 3,
}

MAX_SCORE = sum(BASIC_WEIGHTS.values()) + sum(GAP_WEIGHTS.values())


def to_bool(val) -> bool:
    return str(val).strip().lower() == "true"


def passes_gate(row: dict) -> bool:
    return all(to_bool(row.get(fld, "")) for fld in GATE_FIELDS)


def compute_score(row: dict) -> int:
    score = 0
    for fld, w in BASIC_WEIGHTS.items():
        if to_bool(row.get(fld, "")):
            score += w
    for fld, w in GAP_WEIGHTS.items():
        if not to_bool(row.get(fld, "")):
            score += w
    return score


def tertile_thresholds(scores: list[int]) -> tuple[float, float]:
    s = sorted(scores)
    n = len(s)
    low_idx = n // 3
    high_idx = (2 * n) // 3
    return s[low_idx], s[high_idx]


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    qualified = []
    disqualified = []

    for row in rows:
        if passes_gate(row):
            row["lead_score"] = compute_score(row)
            qualified.append(row)
        else:
            row["lead_score"] = ""
            row["lead_label"] = "Không đủ điều kiện"
            disqualified.append(row)

    scores = [row["lead_score"] for row in qualified]
    if scores:
        low_th, high_th = tertile_thresholds(scores)
    else:
        low_th, high_th = 0, 0

    for row in qualified:
        s = row["lead_score"]
        if s <= low_th:
            row["lead_label"] = "Cold"
        elif s <= high_th:
            row["lead_label"] = "Warm"
        else:
            row["lead_label"] = "Hot"

    all_rows = qualified + disqualified

    fieldnames = list(rows[0].keys())
    for k in ("lead_score", "lead_label"):
        if k in fieldnames:
            fieldnames.remove(k)
    fieldnames += ["lead_score", "lead_label"]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Bước 4.3: kiểm tra phân bố nhãn ---
    dist = Counter(row["lead_label"] for row in all_rows)
    n_total = len(all_rows)
    n_qualified = len(qualified)

    print(f"Điểm tối đa lý thuyết: {MAX_SCORE}")
    if scores:
        print(f"Ngưỡng tam phân vị (tính trên {n_qualified} lead hợp lệ): "
              f"Cold <= {low_th}, Warm <= {high_th}, Hot > {high_th}")
    print()
    print(f"=== Bước 4.3: Phân bố nhãn (tổng {n_total} công ty) ===")
    for label in ("Không đủ điều kiện", "Cold", "Warm", "Hot"):
        cnt = dist.get(label, 0)
        pct = 100 * cnt / n_total
        bar = "#" * round(pct / 2)
        print(f"  {label:20s}: {cnt:3d}/{n_total} ({pct:5.1f}%) {bar}")

    if n_qualified:
        max_pct_qualified = max(
            dist.get(l, 0) for l in ("Cold", "Warm", "Hot")
        ) / n_qualified * 100
        if max_pct_qualified > 60:
            print(
                "\nCẢNH BÁO: trong số lead hợp lệ, 1 nhóm chiếm >60% -> phân bố lệch, "
                "nên xem lại trọng số ở GAP_WEIGHTS/BASIC_WEIGHTS trước khi train model."
            )
        else:
            print("\nPhân bố (trong số lead hợp lệ) tương đối cân bằng, có thể dùng để train model.")

    print(f"\n-> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
