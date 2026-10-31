"""
feature_engineering.py

Giai đoạn 3 - Feature Engineering. Đọc leads_raw.csv (đầu ra Giai đoạn 2),
thực hiện đúng 4 bước trong kế hoạch dự án:

  3.1 Dữ liệu thiếu: các trường has_X đã là boolean 0/1 sẵn từ scraper
      (KHÔNG nội suy/vá giá trị - "không tìm thấy X" = has_X: 0, đúng yêu cầu).
  3.2 Chuẩn hoá định dạng: blog_updated_days đã là số ngày (không phải text)
      từ scraper_template_A.py; ở đây bổ sung 2 đặc trưng phái sinh từ nó
      (has_blog_date, blog_is_recent) để tránh dùng thẳng 1 cột có nhiều
      giá trị thiếu (12/42 site không xác định được ngày blog) làm input
      trực tiếp cho model.
  3.3 Bảng đặc trưng cuối cùng: in ra danh sách cột dùng cho Giai đoạn 4-5.
  3.4 Loại công ty scrape lỗi/thiếu quá nhiều dữ liệu: loại các dòng có
      error, hoặc có ít hơn MIN_VALID_FEATURES / 7 đặc trưng gốc được tìm
      thấy (đặt ngưỡng ở đây, có thể chỉnh MIN_VALID_FEATURES bên dưới).

Cách dùng:
    python feature_engineering.py

Output: features.csv (bảng đặc trưng sạch, dùng cho Giai đoạn 4 - gán nhãn)
        dropped_companies.csv (log các công ty bị loại + lý do, để minh bạch
        trong báo cáo - đúng tinh thần bước 7.2 "ghi rõ minh bạch")
"""

import csv

INPUT_FILE = "leads_raw.csv"
OUTPUT_FEATURES = "features.csv"
OUTPUT_DROPPED = "dropped_companies.csv"

# Bước 3.4: ngưỡng % đặc trưng gốc hợp lệ tối thiểu để giữ lại 1 công ty.
# 7 đặc trưng gốc: has_phone, has_address, has_working_hours,
# has_contact_form, has_online_booking, has_price_list, has_blog.
# Ngưỡng 3/7 (~43%) nghĩa là: nếu 1 site chỉ có <=2 trong 7 đặc trưng này là
# True, khả năng cao scraper đọc sai cấu trúc trang (không phải do công ty
# đó thực sự thiếu tất cả) -> loại để tránh nhiễu dữ liệu train.
MIN_VALID_FEATURES = 3

CORE_BOOL_FIELDS = [
    "has_phone", "has_address", "has_working_hours", "has_contact_form",
    "has_online_booking", "has_price_list", "has_blog",
]

BLOG_RECENT_THRESHOLD_DAYS = 180  # ~6 tháng - blog cập nhật trong 6 tháng gần đây coi là "active"


def to_bool(val: str) -> bool:
    return str(val).strip().lower() == "true"


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept = []
    dropped = []

    for row in rows:
        reasons = []

        if row.get("error"):
            reasons.append(f"lỗi scrape: {row['error']}")

        n_valid = sum(1 for fld in CORE_BOOL_FIELDS if to_bool(row.get(fld, "")))
        if n_valid < MIN_VALID_FEATURES:
            reasons.append(
                f"chỉ {n_valid}/{len(CORE_BOOL_FIELDS)} đặc trưng gốc hợp lệ "
                f"(ngưỡng tối thiểu {MIN_VALID_FEATURES})"
            )

        if reasons:
            dropped.append({
                "domain": row["domain"],
                "url": row["url"],
                "n_valid_features": n_valid,
                "reason": "; ".join(reasons),
            })
            continue

        # --- Bước 3.2: đặc trưng phái sinh từ blog_updated_days ---
        blog_days_raw = row.get("blog_updated_days", "")
        has_blog_date = blog_days_raw not in ("", "None", None)
        blog_is_recent = False
        blog_updated_days_val = ""
        if has_blog_date:
            try:
                blog_days_int = int(float(blog_days_raw))
                blog_updated_days_val = blog_days_int
                blog_is_recent = blog_days_int <= BLOG_RECENT_THRESHOLD_DAYS
            except ValueError:
                has_blog_date = False

        kept.append({
            "domain": row["domain"],
            "url": row["url"],
            "has_phone": to_bool(row["has_phone"]),
            "has_address": to_bool(row["has_address"]),
            "has_working_hours": to_bool(row["has_working_hours"]),
            "has_contact_form": to_bool(row["has_contact_form"]),
            "has_online_booking": to_bool(row["has_online_booking"]),
            "has_price_list": to_bool(row["has_price_list"]),
            "has_blog": to_bool(row["has_blog"]),
            "has_blog_date": has_blog_date,
            "blog_updated_days": blog_updated_days_val,
            "blog_is_recent": blog_is_recent,
        })

    feature_fieldnames = [
        "domain", "url",
        "has_phone", "has_address", "has_working_hours", "has_contact_form",
        "has_online_booking", "has_price_list", "has_blog",
        "has_blog_date", "blog_updated_days", "blog_is_recent",
    ]
    with open(OUTPUT_FEATURES, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    with open(OUTPUT_DROPPED, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "url", "n_valid_features", "reason"])
        writer.writeheader()
        writer.writerows(dropped)

    print("=== Bước 3.3: Bảng đặc trưng cuối cùng dùng cho model ===")
    for fld in feature_fieldnames[2:]:
        print(f"  - {fld}")
    print()
    print(f"Giữ lại: {len(kept)} công ty -> {OUTPUT_FEATURES}")
    print(f"Loại bỏ: {len(dropped)} công ty -> {OUTPUT_DROPPED}")
    if dropped:
        print("\nDanh sách bị loại:")
        for d in dropped:
            print(f"  - {d['domain']}: {d['reason']}")


if __name__ == "__main__":
    main()
