"""
run_batch_scrape.py

Bước 2.3-2.4: chạy scraper_template_A.py trên TOÀN BỘ danh sách công ty
trong final_company_list.csv (đầu ra của merge_final_list.py), log lại các
trường hợp lỗi/timeout, và lưu dữ liệu thô (mỗi dòng = 1 công ty) ra:
  - leads_raw.csv   (dùng cho Giai đoạn 3 - Feature Engineering)
  - leads_raw.json  (bản đầy đủ, dễ debug)
  - scrape_errors.log (log lỗi/timeout theo yêu cầu bước 2.3)

Cách dùng:
    python run_batch_scrape.py

Yêu cầu: scraper_template_A.py phải nằm cùng thư mục.
"""

import csv
import json
import sys
import time
from datetime import datetime

from scraper_template_A import scrape_site

REQUEST_DELAY_SEC = 2.0  # lịch sự với server, tránh bị chặn IP giữa chừng khi scrape hàng loạt

FIELDNAMES = [
    "domain", "url", "phone", "has_phone", "address", "has_address",
    "has_working_hours", "has_contact_form", "has_online_booking",
    "has_price_list", "blog_url", "has_blog", "blog_updated_days", "error",
]


def main():
    with open("final_company_list.csv", encoding="utf-8") as f:
        companies = list(csv.DictReader(f))

    results = []
    errors = []

    for i, row in enumerate(companies, 1):
        domain = row["domain"]
        url = row["url"]
        print(f"[{i}/{len(companies)}] Scraping {domain} ...", file=sys.stderr)

        try:
            data = scrape_site(url)
        except Exception as e:  # noqa: BLE001 - ghi log rồi tiếp tục site khác, không dừng cả batch
            data = {"url": url, "domain": domain, "error": str(e)}

        if data.get("error"):
            errors.append(f"[{datetime.now().isoformat(timespec='seconds')}] {domain}: {data['error']}")

        clean_row = {k: data.get(k) for k in FIELDNAMES}
        results.append(clean_row)

        time.sleep(REQUEST_DELAY_SEC)

    with open("leads_raw.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    with open("leads_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if errors:
        with open("scrape_errors.log", "w", encoding="utf-8") as f:
            f.write("\n".join(errors) + "\n")

    n_ok = sum(1 for r in results if not r.get("error"))
    print(f"\nXong. {n_ok}/{len(results)} công ty scrape thành công.", file=sys.stderr)
    print("-> leads_raw.csv, leads_raw.json" + (", scrape_errors.log" if errors else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
