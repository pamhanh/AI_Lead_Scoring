"""
filter_candidates.py

Bước 2.2-2.4 của kế hoạch dự án: chạy hàng loạt trên danh sách candidates.csv
(do Claude tìm ở bước 2.1), tự động:
  - kiểm tra robots.txt có chặn crawl toàn bộ không
  - phát hiện site JS-heavy (nội dung gần như trống nếu không chạy JS -
    requests/BeautifulSoup không render JS nên site loại này sẽ scrape sai)
  - chạy thử scraper_template_A.py (import lại các hàm) để xem trích xuất
    được bao nhiêu field cho mỗi site -> ước lượng % dữ liệu hợp lệ

Kết quả ghi ra candidates_filtered.csv, có cột "keep" (True/False) và "note"
giải thích lý do loại nếu keep=False, để bạn (Bạn - theo bảng phân công
Giai đoạn 2) duyệt lại lần cuối trước khi scraping hàng loạt thật sự.

Cách dùng:
    python filter_candidates.py candidates.csv candidates_filtered.csv

Yêu cầu: scraper_template_A.py phải nằm cùng thư mục.
Yêu cầu: pip install requests beautifulsoup4 lxml
"""

import sys
import csv
import time
import requests
from urllib.parse import urlparse

from scraper_template_A import scrape_site, fetch, HEADERS, TIMEOUT

MIN_TEXT_LEN_NOT_JS_HEAVY = 400  # dưới ngưỡng này coi như site JS-heavy/rỗng
REQUEST_DELAY_SEC = 1.5  # lịch sự với server, tránh bị chặn IP giữa chừng


def check_robots(base_url: str) -> tuple[bool, str]:
    """Trả về (robots_ok, ghi_chú). robots_ok=False nếu User-agent: * bị Disallow: /"""
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        resp = requests.get(robots_url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return True, "không có robots.txt (coi như cho phép)"
        lines = [l.strip() for l in resp.text.splitlines()]
        in_wildcard_block = False
        for line in lines:
            low = line.lower()
            if low.startswith("user-agent:"):
                in_wildcard_block = low.split(":", 1)[1].strip() == "*"
            elif in_wildcard_block and low.startswith("disallow:"):
                path = low.split(":", 1)[1].strip()
                if path == "/":
                    return False, "robots.txt chặn toàn bộ (Disallow: / cho User-agent: *)"
        return True, "robots.txt cho phép (hoặc chỉ chặn 1 phần)"
    except requests.RequestException as e:
        return True, f"không kiểm tra được robots.txt ({e}) - coi như cho phép, cần tự xem lại"


def check_js_heavy(base_url: str) -> tuple[bool, int]:
    """Trả về (is_js_heavy, so_ky_tu_text). Heuristic: nếu HTML tĩnh gần như
    không có text (nội dung được JS render sau khi load) thì requests/BS4 sẽ
    không lấy được gì -> loại khỏi danh sách scrape hàng loạt."""
    try:
        soup = fetch(base_url)
        text_len = len(soup.get_text(strip=True))
        return text_len < MIN_TEXT_LEN_NOT_JS_HEAVY, text_len
    except requests.RequestException:
        return True, 0


def run(input_csv: str, output_csv: str):
    with open(input_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(rows, 1):
        url = row["url"]
        domain = row["domain"]
        print(f"[{i}/{len(rows)}] Đang kiểm tra {domain} ...", file=sys.stderr)

        robots_ok, robots_note = check_robots(url)
        js_heavy, text_len = check_js_heavy(url)

        scraped = {}
        error = None
        if not js_heavy:
            try:
                scraped = scrape_site(url)
                error = scraped.get("error")
            except Exception as e:  # noqa: BLE001 - muốn log mọi lỗi rồi tiếp tục site khác
                error = str(e)

        n_fields_ok = sum(
            1
            for k in ("has_phone", "has_address", "has_working_hours", "has_contact_form")
            if scraped.get(k)
        )

        keep = robots_ok and not js_heavy and error is None and n_fields_ok >= 2
        reasons = []
        if not robots_ok:
            reasons.append(robots_note)
        if js_heavy:
            reasons.append(f"nghi ngờ JS-heavy (chỉ {text_len} ký tự text tĩnh)")
        if error:
            reasons.append(f"lỗi khi scrape: {error}")
        if not js_heavy and not error and n_fields_ok < 2:
            reasons.append(f"chỉ trích xuất được {n_fields_ok}/4 field cơ bản - cấu trúc khác nhiều so với site mẫu")

        results.append({
            "domain": domain,
            "url": url,
            "keep": keep,
            "robots_ok": robots_ok,
            "js_heavy": js_heavy,
            "text_len": text_len,
            "n_fields_ok": n_fields_ok,
            "phone": scraped.get("phone"),
            "has_address": scraped.get("has_address"),
            "has_working_hours": scraped.get("has_working_hours"),
            "has_contact_form": scraped.get("has_contact_form"),
            "note": "; ".join(reasons) if reasons else "OK",
        })

        time.sleep(REQUEST_DELAY_SEC)

    fieldnames = [
        "domain", "url", "keep", "robots_ok", "js_heavy", "text_len",
        "n_fields_ok", "phone", "has_address", "has_working_hours",
        "has_contact_form", "note",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    n_keep = sum(1 for r in results if r["keep"])
    print(f"\nXong. Giữ lại {n_keep}/{len(results)} site. Xem chi tiết trong {output_csv}", file=sys.stderr)


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "candidates.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "candidates_filtered.csv"
    run(inp, out)
