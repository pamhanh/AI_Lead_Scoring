"""
app.py

Giai đoạn 6, bước 6.1: API nhỏ (Flask) nhận URL 1 website nha khoa, TỰ SCRAPE
(dùng lại scraper_template_A.py), trích đặc trưng (đúng logic
feature_engineering.py), áp gate (đúng logic bootstrap_labeling.py Giai
đoạn 4), rồi chạy model Logistic Regression đã train (model.pkl) để trả về
điểm số + nhãn Hot/Warm/Cold.

Đây là 1 nguồn logic DUY NHẤT cho việc scrape - n8n sẽ gọi API này thay vì
viết lại logic scraping bằng JavaScript trong Code node, để tránh 2 bản
logic lệch nhau khi sửa code sau này.

Cách chạy:
    pip install flask joblib scikit-learn pandas requests beautifulsoup4 lxml
    python app.py

Mặc định chạy ở http://localhost:5000
Endpoint chính: POST /score  body: {"url": "https://..."}
"""

import joblib
from flask import Flask, request, jsonify

from scraper_template_A import scrape_site

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURE_COLS = joblib.load("feature_cols.pkl")

GATE_FIELDS = ["has_phone", "has_address"]
BLOG_DAYS_MISSING_SENTINEL = 9999


def build_features(raw: dict) -> dict:
    """Đúng logic feature_engineering.py: bool 0/1 + đặc trưng phái sinh từ blog."""
    features = {}
    for fld in [
        "has_phone", "has_address", "has_working_hours", "has_contact_form",
        "has_online_booking", "has_price_list", "has_blog",
    ]:
        features[fld] = bool(raw.get(fld))

    blog_days = raw.get("blog_updated_days")
    has_blog_date = blog_days is not None
    blog_is_recent = False
    blog_updated_days_val = BLOG_DAYS_MISSING_SENTINEL
    if has_blog_date:
        try:
            blog_days_int = int(float(blog_days))
            blog_updated_days_val = blog_days_int
            blog_is_recent = blog_days_int <= 180
        except (TypeError, ValueError):
            has_blog_date = False

    features["has_blog_date"] = has_blog_date
    features["blog_is_recent"] = blog_is_recent
    features["blog_updated_days"] = blog_updated_days_val
    return features


def passes_gate(features: dict) -> bool:
    return all(features.get(fld) for fld in GATE_FIELDS)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/score", methods=["POST"])
def score():
    body = request.get_json(silent=True) or {}
    url = body.get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "Thiếu trường 'url' trong body"}), 400
    if not url.startswith("http"):
        url = "https://" + url

    try:
        raw = scrape_site(url)
    except Exception as e:  # noqa: BLE001 - trả lỗi rõ ràng cho n8n xử lý, không để API sập
        return jsonify({"ok": False, "url": url, "error": f"Lỗi scrape: {e}"}), 200

    if raw.get("error"):
        return jsonify({
            "ok": False, "url": url, "domain": raw.get("domain"),
            "error": raw["error"],
        }), 200

    features = build_features(raw)

    if not passes_gate(features):
        return jsonify({
            "ok": True,
            "qualified": False,
            "domain": raw.get("domain"),
            "url": url,
            "lead_label": "Không đủ điều kiện",
            "lead_score": None,
            "reason": "Thiếu has_phone hoặc has_address - không xác minh được là doanh nghiệp thật",
            "raw_features": {**features, "phone": raw.get("phone"), "address": raw.get("address")},
        })

    # cột numeric (blog_updated_days) giữ nguyên số, cột bool ép 0/1
    X_row = [[
        features[c] if c == "blog_updated_days" else (1 if features[c] else 0)
        for c in FEATURE_COLS
    ]]
    X_scaled = scaler.transform(X_row)

    lead_label = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    proba_by_class = dict(zip(model.classes_, [round(float(p), 3) for p in proba]))

    return jsonify({
        "ok": True,
        "qualified": True,
        "domain": raw.get("domain"),
        "url": url,
        "lead_label": lead_label,
        "probabilities": proba_by_class,
        "raw_features": {**features, "phone": raw.get("phone"), "address": raw.get("address")},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
