"""
scraper_template_A.py

Scraper mẫu cho Giai đoạn 1 (Bài tập lớn AI - Lead Scoring hệ thống nha khoa).

Site mẫu:      https://nhakhoatrongrang.com/  (Nha Khoa Trồng Răng Sài Gòn)
Nền tảng CMS:  WordPress + theme Flatsome (child theme) + Elementor page builder
Ghi chú CMS:   Elementor sinh class/id ngẫu nhiên riêng cho từng trang/site
               (vd. elementor-element-xxxxxxx) nên KHÔNG đáng tin cậy để dùng
               làm "khớp cấu trúc" khi áp dụng hàng loạt cho nhiều site khác
               nhau (Giai đoạn 2). Vì vậy scraper này dùng cách tiếp cận
               heuristic / pattern-based (regex + landmark HTML tags như
               <a href="tel:...">, <form>, thẻ ngày tháng trong URL blog...)
               thay vì phụ thuộc vào class CSS cụ thể của 1 site.
               Cách này bền hơn khi mở rộng sang các site WordPress/Elementor
               khác cùng ngành nha khoa ở Giai đoạn 2.

Các trường (feature) được trích xuất — khớp với bảng đặc trưng ở Giai đoạn 3
của kế hoạch dự án:
    - phone              : số điện thoại (từ href="tel:...")
    - has_phone          : bool
    - address            : chuỗi địa chỉ đầu tiên tìm được (heuristic)
    - has_address        : bool
    - has_working_hours  : bool (tìm pattern "Thứ 2 ... Chủ Nhật" hoặc giờ dạng HH:MM)
    - has_contact_form   : bool (có thẻ <form> với ít nhất 2 input)
    - has_online_booking : bool (form có field liên quan đặt lịch: ngày/giờ hẹn)
    - has_price_list     : bool (có trang/menu "bảng giá")
    - blog_url           : link tới trang blog/tin tức nếu tìm thấy trong menu
    - has_blog           : bool
    - blog_updated_days  : số ngày kể từ bài viết mới nhất trên blog (None nếu
                            không xác định được ngày) -- cần scrape thêm trang
                            blog riêng, hàm scrape_blog_latest() bên dưới.

Cách dùng:
    python scraper_template_A.py https://nhakhoatrongrang.com/

Yêu cầu: pip install requests beautifulsoup4 lxml
"""

import re
import sys
import json
import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 15

PHONE_RE = re.compile(r"tel:([+\d][\d\s\-\.]{6,})")
HOURS_RE = re.compile(
    r"(Th(ứ|u)\s*[2-7]|Chủ\s*Nh(ậ|a)t).{0,40}?\d{1,2}\s*[:h]\s*\d{0,2}", re.IGNORECASE
)
ADDRESS_HINT_RE = re.compile(
    r"(Địa chỉ|Đ/c)\s*[:\-]?\s*(.{5,120})", re.IGNORECASE
)
BOOKING_FIELD_HINTS = ("ngay-dat-lich", "gio-hen", "date", "booking", "appointment", "lich-hen")
PRICE_LIST_HINTS = ("bảng giá", "bang-gia", "price")
BLOG_HINTS = ("tin-tuc", "tin tức", "blog", "kien-thuc", "kiến thức")


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def extract_phone(soup: BeautifulSoup) -> str | None:
    for a in soup.select("a[href^='tel:']"):
        m = PHONE_RE.search(a.get("href", ""))
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    # fallback: quét text toàn trang cho pattern số điện thoại VN
    text = soup.get_text(" ", strip=True)
    m = re.search(r"0\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{3,4}", text)
    return m.group(0) if m else None


def extract_address(soup: BeautifulSoup) -> str | None:
    # Ưu tiên link Google Maps: text của link đó gần như luôn là địa chỉ thật,
    # đáng tin hơn regex theo từ khoá (vd "Cơ sở vật chất" không phải địa chỉ
    # dù chứa từ "Cơ sở").
    for a in soup.select(
        "a[href*='maps.app.goo.gl'], a[href*='google.com/maps'], a[href*='goo.gl/maps']"
    ):
        label = a.get_text(strip=True)
        if len(label) > 8 and re.search(r"\d", label):
            return label

    # fallback: regex theo từ khoá "Địa chỉ" / "Đ/c", nhưng chỉ chấp nhận nếu
    # đoạn sau đó có số nhà (tránh khớp nhầm tiêu đề section không phải địa chỉ)
    text = soup.get_text("\n", strip=True)
    for m in ADDRESS_HINT_RE.finditer(text):
        candidate = m.group(2).split("\n")[0].strip()
        if re.search(r"\d", candidate):
            return candidate
    return None


def has_working_hours(soup: BeautifulSoup) -> bool:
    text = soup.get_text(" ", strip=True)
    return bool(HOURS_RE.search(text))


def find_contact_form(soup: BeautifulSoup):
    """Quét TẤT CẢ <form> trên trang, không dừng lại ở form đầu tiên.

    Một site có thể có nhiều form (vd form liên hệ nhanh ở đầu trang + form
    đặt lịch đầy đủ ở dưới) - nếu chỉ xét form đầu tiên sẽ dễ bỏ sót field
    đặt lịch nằm ở form khác (đã gặp lỗi này khi test trên site mẫu).
    """
    forms = soup.find_all("form")
    has_any_form = False
    has_booking = False
    for form in forms:
        inputs = form.find_all(["input", "select", "textarea"])
        if len(inputs) >= 2:
            has_any_form = True
            names = " ".join((i.get("name") or "") for i in inputs).lower()
            placeholders = " ".join((i.get("placeholder") or "") for i in inputs).lower()
            combined = names + " " + placeholders
            if (
                any(h in combined for h in BOOKING_FIELD_HINTS)
                or any(i.get("type") == "date" for i in inputs)
                or "đặt lịch" in combined
                or "giờ hẹn" in combined
                or "giờ đặt" in combined
            ):
                has_booking = True
    return has_any_form, has_booking


def find_menu_link(soup: BeautifulSoup, hints, base_url):
    for a in soup.find_all("a", href=True):
        label = a.get_text(strip=True).lower()
        href = a["href"].lower()
        if any(h in label for h in hints) or any(h in href for h in hints):
            return urljoin(base_url, a["href"])
    return None


def _extract_date_days(soup: BeautifulSoup):
    """Tìm 1 ngày tháng hợp lệ trong trang và trả về số ngày kể từ hiện tại.

    Thứ tự ưu tiên: meta tag SEO (Yoast/RankMath thường có
    article:published_time / article:modified_time - đây là cách phổ biến
    nhất để lấy ngày trên site WordPress vì nhiều theme KHÔNG hiển thị ngày
    đăng công khai trong nội dung, dù vẫn có trong <head>), rồi mới tới thẻ
    <time>, rồi tới text hiển thị dạng dd/mm/yyyy.
    """
    for prop in ("article:modified_time", "article:published_time"):
        meta = soup.find("meta", attrs={"property": prop})
        if meta and meta.get("content"):
            try:
                dt = datetime.datetime.fromisoformat(meta["content"][:19])
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                return max((datetime.datetime.now() - dt).days, 0)
            except ValueError:
                continue

    # WordPress thường có thẻ <time datetime="..."> cho bài viết
    for time_tag in soup.find_all("time"):
        dt_attr = time_tag.get("datetime")
        if dt_attr:
            try:
                dt = datetime.datetime.fromisoformat(dt_attr[:10])
                return (datetime.datetime.now() - dt).days
            except ValueError:
                continue

    # fallback: tìm pattern ngày dạng dd/mm/yyyy trong text
    text = soup.get_text(" ", strip=True)
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if m:
        try:
            dt = datetime.datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            return (datetime.datetime.now() - dt).days
        except ValueError:
            pass
    return None


def scrape_blog_latest(blog_url: str):
    """Cố gắng lấy ngày bài viết mới nhất trên trang blog/tin tức.
    Trả về số ngày kể từ hiện tại, hoặc None nếu không xác định được.

    Trang danh mục blog (vd /kien-thuc-tong-quat/) thường KHÔNG có thẻ <time>
    ngay trên trang danh sách - phải mở 1-2 bài viết đầu tiên trong danh sách
    mới lấy được ngày đăng thật.
    """
    try:
        listing_soup = fetch(blog_url)
    except requests.RequestException:
        return None

    days = _extract_date_days(listing_soup)
    if days is not None:
        return days

    # Tìm link bài viết trong danh sách: ưu tiên các thẻ <a> có tiêu đề dài
    # (loại bỏ link menu/nav thường ngắn), cùng domain, khác chính blog_url.
    domain = urlparse(blog_url).netloc
    seen = set()
    candidates = []
    for a in listing_soup.find_all("a", href=True):
        href = urljoin(blog_url, a["href"])
        title = a.get_text(strip=True)
        if (
            urlparse(href).netloc == domain
            and href.rstrip("/") != blog_url.rstrip("/")
            and len(title) > 15
            and href not in seen
        ):
            seen.add(href)
            candidates.append(href)

    for href in candidates[:3]:
        try:
            post_soup = fetch(href)
        except requests.RequestException:
            continue
        days = _extract_date_days(post_soup)
        if days is not None:
            return days
    return None


def scrape_site(url: str) -> dict:
    domain = urlparse(url).netloc
    result = {"url": url, "domain": domain}

    try:
        soup = fetch(url)
    except requests.RequestException as e:
        result["error"] = str(e)
        return result

    phone = extract_phone(soup)
    address = extract_address(soup)
    has_form, has_booking = find_contact_form(soup)
    blog_url = find_menu_link(soup, BLOG_HINTS, url)
    price_url = find_menu_link(soup, PRICE_LIST_HINTS, url)

    result.update({
        "phone": phone,
        "has_phone": bool(phone),
        "address": address,
        "has_address": bool(address),
        "has_working_hours": has_working_hours(soup),
        "has_contact_form": has_form,
        "has_online_booking": has_booking,
        "has_price_list": bool(price_url),
        "blog_url": blog_url,
        "has_blog": bool(blog_url),
    })

    result["blog_updated_days"] = scrape_blog_latest(blog_url) if blog_url else None

    # Luôn ghé qua trang /lien-he/ nếu có: form đặt lịch đầy đủ (với field
    # ngày/giờ hẹn) thường KHÔNG nằm ở form rút gọn trên trang chủ, nên chỉ
    # kiểm tra trang chủ sẽ bỏ sót has_online_booking (đã gặp lỗi này khi
    # test trên site mẫu). Đồng thời bổ sung địa chỉ nếu trang chủ chưa có.
    contact_url = find_menu_link(soup, ("liên hệ", "lien-he", "contact"), url)
    if contact_url and contact_url != url:
        try:
            contact_soup = fetch(contact_url)
            if not result["has_address"]:
                addr2 = extract_address(contact_soup)
                if addr2:
                    result["address"] = addr2
                    result["has_address"] = True
            has_form2, has_booking2 = find_contact_form(contact_soup)
            result["has_contact_form"] = result["has_contact_form"] or has_form2
            result["has_online_booking"] = result["has_online_booking"] or has_booking2
        except requests.RequestException:
            pass

    return result


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "https://nhakhoatrongrang.com/"
    data = scrape_site(target)
    print(json.dumps(data, ensure_ascii=False, indent=2))
