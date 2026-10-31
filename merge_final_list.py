"""
merge_final_list.py

Gộp site mẫu + tất cả các đợt candidates_filtered*.csv lại, chỉ giữ những
site keep=True, xuất ra final_company_list.csv - danh sách công ty cuối
cùng dùng để scraping hàng loạt ở Bước 2.3-2.4.

Cách dùng (chạy sau khi đã có đủ candidates_filtered.csv,
candidates_filtered_batch2.csv, candidates_filtered_batch3.csv, ... trong
cùng thư mục):

    python merge_final_list.py
"""

import csv
import glob
import sys

SAMPLE_SITE = {"domain": "nhakhoatrongrang.com", "url": "https://nhakhoatrongrang.com/"}


def main():
    files = sorted(glob.glob("candidates_filtered*.csv"))
    if not files:
        print("Không tìm thấy file candidates_filtered*.csv nào trong thư mục hiện tại.", file=sys.stderr)
        sys.exit(1)

    seen_domains = set()
    final_rows = [SAMPLE_SITE]
    seen_domains.add(SAMPLE_SITE["domain"])

    for path in files:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                domain = row["domain"]
                if domain in seen_domains:
                    continue
                if row.get("keep", "").strip().lower() == "true":
                    final_rows.append({"domain": domain, "url": row["url"]})
                    seen_domains.add(domain)

    with open("final_company_list.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "url"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Đã gộp {len(files)} file: {', '.join(files)}")
    print(f"Tổng số công ty trong danh sách cuối: {len(final_rows)}")
    print("-> final_company_list.csv")


if __name__ == "__main__":
    main()
