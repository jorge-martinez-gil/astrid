from __future__ import annotations

import math
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_images"
W, H = 3840, 2160

NAVY = (16, 24, 39)
INK = (28, 36, 52)
MUTED = (86, 97, 118)
WHITE = (255, 255, 255)
PAPER = (246, 248, 252)
BLUE = (48, 109, 246)
CYAN = (0, 181, 216)
GREEN = (18, 163, 112)
AMBER = (245, 158, 11)
ORANGE = (239, 113, 57)
VIOLET = (126, 87, 194)
PINK = (220, 72, 121)
RED = (225, 72, 72)
SLATE = (91, 109, 135)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    base = Path("C:/Windows/Fonts")
    candidates = {
        "regular": ["segoeui.ttf", "arial.ttf"],
        "bold": ["segoeuib.ttf", "arialbd.ttf"],
        "semilight": ["segoeuisl.ttf", "segoeui.ttf", "arial.ttf"],
    }[name]
    for item in candidates:
        path = base / item
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F = {
    "title": font("bold", 122),
    "subtitle": font("regular", 52),
    "h1": font("bold", 88),
    "h2": font("bold", 62),
    "h3": font("bold", 42),
    "body": font("regular", 35),
    "small": font("regular", 28),
    "tiny": font("regular", 23),
    "mono": font("regular", 28),
}


def rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, msg, fill=INK, f=None, anchor=None, spacing=8, align="left"):
    draw.multiline_text(xy, msg, fill=fill, font=f or F["body"], anchor=anchor, spacing=spacing, align=align)


def wrap_text(msg: str, width: int) -> str:
    return "\n".join(wrap(msg, width=width, break_long_words=False))


def fit_text(draw, box, msg, fill=INK, font_obj=None, anchor="mm"):
    x1, y1, x2, y2 = box
    fnt = font_obj or F["body"]
    lines = wrap(msg, max(12, int((x2 - x1) / (fnt.size * 0.52))))
    content = "\n".join(lines)
    text(draw, ((x1 + x2) / 2, (y1 + y2) / 2), content, fill=fill, f=fnt, anchor=anchor, align="center")


def gradient_bg(left=(245, 248, 252), right=(233, 240, 255), accent=(255, 246, 232)):
    sw, sh = 480, 270
    img = Image.new("RGB", (sw, sh), left)
    pix = img.load()
    for y in range(sh):
        for x in range(sw):
            t = x / sw
            base = tuple(int(left[i] * (1 - t) + right[i] * t) for i in range(3))
            glow = math.exp(-(((x - sw * 0.84) / 132) ** 2 + ((y - sh * 0.15) / 90) ** 2))
            color = tuple(int(base[i] * (1 - glow * 0.45) + accent[i] * glow * 0.45) for i in range(3))
            pix[x, y] = color
    return img.resize((W, H), Image.Resampling.BICUBIC)


def save(img: Image.Image, name: str):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    if path.exists():
        return
    img.save(path, "PNG", optimize=True)


def add_header(draw, eyebrow: str, title: str, subtitle: str | None = None):
    text(draw, (180, 145), eyebrow.upper(), fill=BLUE, f=F["small"])
    text(draw, (180, 210), title, fill=NAVY, f=F["h1"])
    if subtitle:
        text(draw, (180, 325), wrap_text(subtitle, 82), fill=MUTED, f=F["subtitle"], spacing=12)


def add_logo(img, pos=(180, 112), size=210):
    logo_path = ROOT / "logo.png"
    if not logo_path.exists():
        return
    logo = Image.open(logo_path).convert("RGBA")
    logo.thumbnail((size, size), Image.Resampling.LANCZOS)
    shadow = Image.new("RGBA", (logo.width + 40, logo.height + 40), (0, 0, 0, 0))
    mask = logo.split()[-1].filter(ImageFilter.GaussianBlur(14))
    shadow.paste((0, 0, 0, 45), (20, 24), mask)
    img.alpha_composite(shadow, (pos[0] - 20, pos[1] - 20))
    img.alpha_composite(logo, pos)


def shadow_card(base, box, radius=42, fill=WHITE, shadow=(30, 45, 70, 34)):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x1, y1, x2, y2 = box
    d.rounded_rectangle((x1 + 12, y1 + 18, x2 + 12, y2 + 18), radius=radius, fill=shadow)
    layer = layer.filter(ImageFilter.GaussianBlur(18))
    base.alpha_composite(layer)
    d = ImageDraw.Draw(base)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=(221, 228, 238), width=2)


def pill(draw, box, label, fill, text_fill=WHITE):
    rounded(draw, box, 28, fill)
    fit_text(draw, box, label, fill=text_fill, font_obj=F["small"])


def arrow(draw, start, end, color=SLATE, width=8):
    draw.line((start, end), fill=color, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 28
    pts = [
        end,
        (end[0] - size * math.cos(ang - 0.42), end[1] - size * math.sin(ang - 0.42)),
        (end[0] - size * math.cos(ang + 0.42), end[1] - size * math.sin(ang + 0.42)),
    ]
    draw.polygon(pts, fill=color)


def slide_01():
    img = gradient_bg().convert("RGBA")
    d = ImageDraw.Draw(img)
    add_logo(img, (190, 150), 270)
    text(d, (510, 174), "ASTRID", fill=NAVY, f=F["title"])
    text(d, (515, 310), "Advanced Software Tools for Reliable Industrial Datasets", fill=MUTED, f=F["subtitle"])
    text(
        d,
        (185, 610),
        wrap_text("A local-first Streamlit platform for auditing dataset quality, privacy risk, drift, robustness, and fairness before industrial AI training.", 67),
        fill=INK,
        f=font("regular", 68),
        spacing=18,
    )
    labels = [("Tabular", BLUE), ("Time Series", VIOLET), ("Images", ORANGE), ("Local Reports", GREEN)]
    x = 185
    for label, color in labels:
        pill(d, (x, 920, x + 420, 1015), label, color)
        x += 465

    shadow_card(img, (2140, 470, 3570, 1600), radius=55)
    d = ImageDraw.Draw(img)
    text(d, (2250, 590), "Dataset Health Score", fill=NAVY, f=F["h2"])
    d.arc((2450, 790, 3250, 1590), 190, 350, fill=(222, 228, 238), width=56)
    d.arc((2450, 790, 3250, 1590), 190, 310, fill=GREEN, width=56)
    text(d, (2850, 1110), "87", fill=GREEN, f=font("bold", 168), anchor="mm")
    text(d, (2850, 1255), "Grade A", fill=NAVY, f=F["h2"], anchor="mm")
    for i, (name, color) in enumerate([("Quality", BLUE), ("Security", AMBER), ("Reliability", VIOLET), ("Robustness", CYAN), ("Fairness", PINK)]):
        y = 1440 + i * 65
        rounded(d, (2260, y, 3135, y + 24), 12, (230, 236, 246))
        rounded(d, (2260, y, 2260 + [790, 690, 730, 610, 650][i], y + 24), 12, color)
        text(d, (3160, y - 8), name, fill=MUTED, f=F["tiny"])

    save(img, "01-astrid-project-overview.png")


def slide_02():
    img = gradient_bg((248, 250, 252), (239, 246, 255), (235, 252, 243)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(d, "Coverage", "Three analyzers, one reliability language", "ASTRID gives tabular files, temporal recordings, and image archives a shared audit vocabulary.")
    cards = [
        ("Tabular", "CSV, Parquet, Excel", ["Missingness", "Duplicate rows", "Split leakage", "PII columns"], BLUE),
        ("Time Series", "Temporal CSV, Parquet", ["Cadence gaps", "Timestamp validity", "First/last drift", "Entity analysis"], VIOLET),
        ("Images", "ZIP archives", ["Corrupt files", "Blur and resolution", "EXIF GPS", "Perceptual duplicates"], ORANGE),
    ]
    for i, (name, sub, bullets, color) in enumerate(cards):
        x1 = 230 + i * 1160
        shadow_card(img, (x1, 620, x1 + 1020, 1670), radius=45)
        d = ImageDraw.Draw(img)
        rounded(d, (x1 + 65, 700, x1 + 245, 880), 38, tuple(list(color) + [28]) if len(color) == 3 else color)
        text(d, (x1 + 155, 790), name[0], fill=color, f=font("bold", 86), anchor="mm")
        text(d, (x1 + 65, 960), name, fill=NAVY, f=F["h2"])
        text(d, (x1 + 65, 1040), sub, fill=MUTED, f=F["body"])
        for j, b in enumerate(bullets):
            y = 1165 + j * 105
            d.ellipse((x1 + 72, y + 12, x1 + 106, y + 46), fill=color)
            text(d, (x1 + 132, y), b, fill=INK, f=F["body"])
    text(d, (1920, 1905), "Each analyzer returns scores, verdicts, evidence, recommendations, and downloadable reports.", fill=SLATE, f=F["body"], anchor="mm")
    save(img, "02-multimodal-analyzers.png")


def slide_03():
    img = gradient_bg((250, 251, 253), (244, 247, 252), (255, 246, 232)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(d, "Scoring Model", "Five reliability dimensions", "A weighted score turns many low-level checks into a single dataset readiness signal.")
    center = (1920, 1160)
    radius = 520
    dims = [("Quality", 0.92, BLUE), ("Security", 0.78, AMBER), ("Reliability", 0.84, VIOLET), ("Robustness", 0.72, CYAN), ("Fairness", 0.69, PINK)]
    for r in [0.25, 0.5, 0.75, 1.0]:
        pts = []
        for i in range(5):
            a = -math.pi / 2 + i * 2 * math.pi / 5
            pts.append((center[0] + radius * r * math.cos(a), center[1] + radius * r * math.sin(a)))
        d.polygon(pts, outline=(211, 219, 232))
    pts = []
    for i, (_, val, _) in enumerate(dims):
        a = -math.pi / 2 + i * 2 * math.pi / 5
        pts.append((center[0] + radius * val * math.cos(a), center[1] + radius * val * math.sin(a)))
    d.polygon(pts, fill=(48, 109, 246, 50), outline=BLUE)
    d.line(pts + [pts[0]], fill=BLUE, width=8)
    for i, (name, val, color) in enumerate(dims):
        a = -math.pi / 2 + i * 2 * math.pi / 5
        x = center[0] + (radius + 230) * math.cos(a)
        y = center[1] + (radius + 150) * math.sin(a)
        rounded(d, (x - 230, y - 70, x + 230, y + 70), 34, WHITE, outline=(219, 226, 238), width=2)
        text(d, (x, y - 16), name, fill=NAVY, f=F["h3"], anchor="mm")
        text(d, (x, y + 33), f"{int(val*100)}%", fill=color, f=F["small"], anchor="mm")
    text(d, center, "Health\nScore", fill=NAVY, f=F["h2"], anchor="mm", align="center")
    save(img, "03-five-reliability-dimensions.png")


def slide_04():
    img = gradient_bg((247, 250, 252), (239, 250, 246), (246, 240, 255)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(d, "Workflow", "From dataset upload to audit evidence", "The application is designed as a fast validation gate before model training.")
    steps = [
        ("1", "Upload", "CSV, Parquet, Excel, temporal files, or image ZIP archives", BLUE),
        ("2", "Configure", "Column roles, threshold presets, score weights, and metadata", VIOLET),
        ("3", "Analyze", "Run modality-specific checks and aggregate evidence", ORANGE),
        ("4", "Review", "Verdict card, dimension tabs, prioritized fixes", GREEN),
        ("5", "Export", "JSON, Markdown, or self-contained HTML report", PINK),
    ]
    y = 1040
    xs = [320, 1030, 1740, 2450, 3160]
    for i, (num, title, body, color) in enumerate(steps):
        x = xs[i]
        d.ellipse((x - 130, y - 130, x + 130, y + 130), fill=color)
        text(d, (x, y), num, fill=WHITE, f=font("bold", 88), anchor="mm")
        if i < len(steps) - 1:
            arrow(d, (x + 160, y), (xs[i + 1] - 165, y), color=(150, 162, 184), width=10)
        shadow_card(img, (x - 300, y + 230, x + 300, y + 680), radius=36)
        d = ImageDraw.Draw(img)
        text(d, (x, y + 345), title, fill=NAVY, f=F["h3"], anchor="mm")
        fit_text(d, (x - 240, y + 420, x + 240, y + 625), body, fill=MUTED, font_obj=F["small"])
    save(img, "04-analysis-workflow.png")


def slide_05():
    img = gradient_bg((250, 251, 253), (244, 249, 255), (255, 247, 235)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(d, "Scoring Engine", "Configurable weights with automatic normalization", "Users can tune the audit to match security-critical, fairness-critical, or benchmark-style use cases.")
    shadow_card(img, (230, 600, 1620, 1680), radius=48)
    shadow_card(img, (1830, 600, 3610, 1680), radius=48)
    d = ImageDraw.Draw(img)
    text(d, (350, 735), "Default weights", fill=NAVY, f=F["h2"])
    weights = [("Quality", 35, BLUE), ("Security", 25, AMBER), ("Reliability", 20, VIOLET), ("Robustness", 10, CYAN), ("Fairness", 10, PINK)]
    for i, (name, value, color) in enumerate(weights):
        y = 880 + i * 135
        text(d, (360, y), name, fill=INK, f=F["body"])
        rounded(d, (820, y + 9, 1420, y + 47), 19, (229, 235, 244))
        rounded(d, (820, y + 9, 820 + value * 15, y + 47), 19, color)
        text(d, (1470, y - 3), f"{value}%", fill=color, f=F["body"])
    text(d, (2000, 750), "Composite health score", fill=NAVY, f=F["h2"])
    formula = "score = sum(component_score x normalized_weight)"
    rounded(d, (2000, 900, 3440, 1045), 34, (244, 247, 251), outline=(221, 228, 238), width=2)
    text(d, (2720, 972), formula, fill=INK, f=F["mono"], anchor="mm")
    grades = [("A", GREEN), ("B", BLUE), ("C", AMBER), ("D", ORANGE), ("F", RED)]
    for i, (g, color) in enumerate(grades):
        x = 2040 + i * 285
        rounded(d, (x, 1220, x + 200, 1420), 34, color)
        text(d, (x + 100, 1320), g, fill=WHITE, f=font("bold", 90), anchor="mm")
    text(d, (2000, 1530), "The score remains interpretable even when sliders do not add to exactly 100.", fill=MUTED, f=F["body"])
    save(img, "05-health-score-and-weights.png")


def slide_06():
    img = gradient_bg((249, 250, 252), (241, 248, 255), (255, 240, 244)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(d, "Risk Detection", "Security and fairness checks before training", "ASTRID surfaces dataset risks that are easy to miss in ordinary exploratory analysis.")
    items = [
        ("PII scan", "Email, phone, ID, and sensitive path patterns", AMBER),
        ("EXIF privacy", "GPS metadata and camera traces in image archives", ORANGE),
        ("Split leakage", "Duplicate row hashes and perceptual image hashes", BLUE),
        ("Group disparity", "Representation and positive-rate differences", PINK),
        ("Drift", "KS statistics across time slices or dataset splits", VIOLET),
        ("Anomalies", "MAD outliers, rare category-label patterns, blur", CYAN),
    ]
    for i, (title, body, color) in enumerate(items):
        col = i % 3
        row = i // 3
        x1 = 250 + col * 1175
        y1 = 665 + row * 560
        shadow_card(img, (x1, y1, x1 + 990, y1 + 405), radius=42)
        d = ImageDraw.Draw(img)
        d.ellipse((x1 + 70, y1 + 70, x1 + 190, y1 + 190), fill=color)
        text(d, (x1 + 130, y1 + 130), "!", fill=WHITE, f=font("bold", 76), anchor="mm")
        text(d, (x1 + 245, y1 + 80), title, fill=NAVY, f=F["h3"])
        text(d, (x1 + 245, y1 + 155), wrap_text(body, 34), fill=MUTED, f=F["body"], spacing=8)
    save(img, "06-security-fairness-risk-detection.png")


def slide_07():
    img = gradient_bg((249, 250, 252), (244, 252, 248), (237, 245, 255)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(d, "Evidence", "Reports for compliance and reproducibility", "After every analysis, ASTRID can export a shareable artifact for reviews, dataset cards, and audit trails.")
    shadow_card(img, (310, 620, 1810, 1690), radius=44)
    d = ImageDraw.Draw(img)
    text(d, (440, 760), "Self-contained HTML report", fill=NAVY, f=F["h2"])
    sections = [("Verdict", GREEN), ("Dimension Scores", BLUE), ("Evidence Tables", VIOLET), ("Recommendations", ORANGE)]
    for i, (label, color) in enumerate(sections):
        y = 910 + i * 160
        rounded(d, (460, y, 1660, y + 92), 24, (243, 246, 251), outline=(222, 229, 240), width=2)
        d.rectangle((460, y, 480, y + 92), fill=color)
        text(d, (530, y + 46), label, fill=INK, f=F["body"], anchor="lm")
    shadow_card(img, (2090, 620, 3530, 1690), radius=44)
    d = ImageDraw.Draw(img)
    text(d, (2220, 760), "Designed for regulated AI", fill=NAVY, f=F["h2"])
    badges = [("Local execution", BLUE), ("No external data upload", GREEN), ("Dataset version evidence", VIOLET), ("EU AI Act alignment", ORANGE), ("ISO/IEC 25012 signal", PINK)]
    for i, (label, color) in enumerate(badges):
        x = 2225 + (i % 2) * 590
        y = 940 + (i // 2) * 190
        pill(d, (x, y, x + 500, y + 96), label, color)
    text(d, (2225, 1530), "Exports can travel with model cards, experiment metadata, and dataset release notes.", fill=MUTED, f=F["body"])
    save(img, "07-reporting-and-compliance.png")


def slide_08():
    img = gradient_bg((248, 250, 252), (245, 246, 255), (241, 252, 246)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(d, "MLOps Placement", "A validation gate before model training", "ASTRID fits where dataset risk is cheapest to catch: before training jobs, deployment reviews, and production drift loops.")
    steps = [
        ("Raw Data", SLATE),
        ("ASTRID Audit", BLUE),
        ("Dataset Artifact", GREEN),
        ("Training Run", VIOLET),
        ("Model Review", ORANGE),
        ("Production Monitor", PINK),
    ]
    y = 1080
    for i, (label, color) in enumerate(steps):
        x = 275 + i * 620
        rounded(d, (x, y - 130, x + 440, y + 130), 42, WHITE, outline=(219, 226, 238), width=3)
        d.ellipse((x + 45, y - 60, x + 165, y + 60), fill=color)
        text(d, (x + 105, y), str(i + 1), fill=WHITE, f=font("bold", 62), anchor="mm")
        fit_text(d, (x + 190, y - 95, x + 420, y + 95), label, fill=NAVY, font_obj=F["h3"])
        if i < len(steps) - 1:
            arrow(d, (x + 465, y), (x + 585, y), color=(146, 160, 183), width=9)
    shadow_card(img, (620, 1515, 3220, 1790), radius=40)
    d = ImageDraw.Draw(img)
    text(d, (780, 1615), "Outcome", fill=BLUE, f=F["h3"])
    text(d, (780, 1690), "Reproducible dataset audits become part of the same lifecycle as model cards, experiments, and release approvals.", fill=INK, f=F["body"])
    save(img, "08-mlops-validation-gate.png")


def slide_09():
    img = gradient_bg((249, 250, 252), (241, 248, 255), (248, 242, 255)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(
        d,
        "Architecture",
        "A compact Streamlit application with shared scoring utilities",
        "Each analyzer owns its modality logic while reusing common scoring, styling, report, and privacy helpers.",
    )

    shadow_card(img, (260, 720, 980, 1110), radius=44)
    shadow_card(img, (1560, 575, 2300, 1260), radius=48)
    shadow_card(img, (2860, 720, 3580, 1110), radius=44)
    d = ImageDraw.Draw(img)

    text(d, (620, 840), "app.py", fill=NAVY, f=F["h2"], anchor="mm")
    text(d, (620, 930), "Streamlit home\nnavigation and landing UI", fill=MUTED, f=F["small"], anchor="mm", align="center")

    text(d, (1930, 720), "utils.py", fill=NAVY, f=F["h2"], anchor="mm")
    utility_items = ["compute_health_score", "DEFAULT_WEIGHTS", "PII patterns", "HTML reports", "Shared CSS"]
    for i, item in enumerate(utility_items):
        y = 825 + i * 70
        rounded(d, (1710, y, 2150, y + 42), 20, (241, 245, 251), outline=(219, 226, 238), width=1)
        text(d, (1930, y + 21), item, fill=INK, f=F["tiny"], anchor="mm")

    text(d, (3220, 840), "docs/", fill=NAVY, f=F["h2"], anchor="mm")
    text(d, (3220, 930), "Jekyll documentation\nand analyzer references", fill=MUTED, f=F["small"], anchor="mm", align="center")

    arrow(d, (1005, 915), (1535, 915), color=(139, 154, 177), width=9)
    arrow(d, (2325, 915), (2835, 915), color=(139, 154, 177), width=9)

    pages = [
        ("01_Tabular.py", "CSV, Parquet, Excel", BLUE),
        ("02_Time_Series.py", "Temporal data", VIOLET),
        ("03_Images.py", "ZIP image archives", ORANGE),
        ("04_Drift_experimental.py", "Drift tracker", GREEN),
    ]
    for i, (name, detail, color) in enumerate(pages):
        x1 = 330 + i * 840
        shadow_card(img, (x1, 1470, x1 + 650, 1740), radius=34)
        d = ImageDraw.Draw(img)
        d.rectangle((x1, 1470, x1 + 650, 1488), fill=color)
        text(d, (x1 + 325, 1562), name, fill=NAVY, f=F["h3"], anchor="mm")
        text(d, (x1 + 325, 1642), detail, fill=MUTED, f=F["small"], anchor="mm")
        arrow(d, (1930, 1290), (x1 + 325, 1440), color=(168, 180, 199), width=6)

    save(img, "09-application-architecture.png")


def slide_10():
    img = gradient_bg((250, 251, 253), (244, 252, 248), (255, 248, 235)).convert("RGBA")
    d = ImageDraw.Draw(img)
    add_header(
        d,
        "Recommendations",
        "Turning audit findings into remediation work",
        "ASTRID ranks plain-language actions so teams can move from diagnosis to dataset improvement.",
    )

    shadow_card(img, (245, 620, 1555, 1725), radius=48)
    shadow_card(img, (1785, 620, 3605, 1725), radius=48)
    d = ImageDraw.Draw(img)

    text(d, (385, 755), "Detected issues", fill=NAVY, f=F["h2"])
    issues = [
        ("High", "PII-like identifiers in column names", RED),
        ("High", "Leakage between train and test splits", ORANGE),
        ("Medium", "Distribution drift in recent slices", AMBER),
        ("Medium", "Class imbalance and group disparity", PINK),
        ("Low", "Missing documentation fields", SLATE),
    ]
    for i, (sev, issue, color) in enumerate(issues):
        y = 900 + i * 135
        pill(d, (395, y, 620, y + 76), sev, color)
        text(d, (670, y + 37), issue, fill=INK, f=F["body"], anchor="lm")

    text(d, (1935, 755), "Prioritized remediation queue", fill=NAVY, f=F["h2"])
    actions = [
        ("1", "Mask or remove sensitive identifiers before release", BLUE),
        ("2", "Rebuild train/test split after duplicate removal", GREEN),
        ("3", "Investigate drifted time windows and source batches", VIOLET),
        ("4", "Rebalance underrepresented groups or document limits", PINK),
        ("5", "Attach dataset provenance and schema notes", ORANGE),
    ]
    for i, (num, action, color) in enumerate(actions):
        y = 900 + i * 135
        d.ellipse((1950, y, 2026, y + 76), fill=color)
        text(d, (1988, y + 38), num, fill=WHITE, f=F["small"], anchor="mm")
        rounded(d, (2070, y, 3430, y + 76), 22, (243, 247, 251), outline=(221, 228, 238), width=2)
        text(d, (2115, y + 38), action, fill=INK, f=F["body"], anchor="lm")

    rounded(d, (1885, 1545, 3500, 1645), 28, (235, 248, 243), outline=(183, 225, 207), width=2)
    text(d, (2692, 1595), "Output: cleaner datasets, clearer evidence, fewer surprises before training", fill=GREEN, f=F["body"], anchor="mm")

    save(img, "10-remediation-recommendations.png")


def main():
    for fn in [slide_01, slide_02, slide_03, slide_04, slide_05, slide_06, slide_07, slide_08, slide_09, slide_10]:
        fn()
    print(f"Wrote {len(list(OUT.glob('*.png')))} PNGs to {OUT}")


if __name__ == "__main__":
    main()
