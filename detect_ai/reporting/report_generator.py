"""Helpers to assemble a text report for GUI display and export to PDF."""
import os
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image, ImageDraw, ImageFont

try:
    from fpdf import FPDF
except Exception:  # pragma: no cover - optional dependency
    FPDF = None


def _fmt_percent(value: float | None) -> str:
    """Format probability as percentage string."""
    if value is None:
        return "-"
    try:
        return f"{value * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _render_module_details(modules: Dict[str, Any]) -> str:
    """Render per-module blocks in a consistent way."""
    lines = []
    for name, data in modules.items():
        status = data.get("status")
        summary = []
        if status == "ok":
            summary.append("статус: ok")
            # Pull a short decision string if present
            decision = data.get("decision") or data.get("label") or data.get("final_decision")
            if decision:
                summary.append(f"решение: {decision}")
            prob_keys: Iterable[str] = ("probability", "prob_of_fake", "final_probability")
            for key in prob_keys:
                prob = data.get(key)
                if isinstance(prob, dict):
                    prob = prob.get("fake_prob")
                if isinstance(prob, (int, float, str)):
                    summary.append(f"вероятность: {prob}")
                    break
        elif status == "skipped":
            summary.append(f"пропущено: {data.get('reason', 'не указано')}")
        else:
            summary.append(f"ошибка: {data.get('message', 'неизвестно')}")
        lines.append(f"- {name}: {'; '.join(summary)}")
    return "\n".join(lines) if lines else "- нет данных"


def _render_metadata(metadata: Dict[str, Any] | None) -> str:
    """Render image metadata as bullet list."""
    if not metadata:
        return "- не предоставлено"
    parts = []
    for key, value in metadata.items():
        parts.append(f"- {key}: {value}")
    return "\n".join(parts)


def _default_conclusion(result: str) -> str:
    """Provide a short conclusion if caller did not supply one."""
    if result.lower() == "подделка":
        return (
            "Обнаружены признаки подделки. Рекомендуется провести дополнительную "
            "проверку и, при необходимости, запросить расширенный отчёт для судебной экспертизы."
        )
    if result.lower() == "оригинал":
        return (
            "Явных признаков подделки не выявлено. При критически важном использовании "
            "можно запросить подробный отчёт для независимой проверки."
        )
    return (
        "Результат неопределён. Для точной оценки стоит запросить подробный отчёт "
        "и выполнить повторный анализ."
    )


def generate_text_report(
    *,
    result_label: str,
    confidence: float | None,
    module_details: Dict[str, Any],
    image_metadata: Dict[str, Any] | None = None,
    conclusion: str | None = None,
) -> str:
    """
    Build a human-readable text report that GUI can display.

    Args:
        result_label: High-level decision, e.g. "Подделка" или "Оригинал".
        confidence: Confidence level in [0, 1] if available.
        module_details: Per-module raw data blocks to surface in report.
        image_metadata: Optional metadata dict (EXIF, размер, дата и т.п.).
        conclusion: Optional custom conclusion; if None auto text is used.
    """
    lines = [
        f"Результат: {result_label}",
        f"Вероятность: {_fmt_percent(confidence)}",
        "Подробности:",
        _render_module_details(module_details),
        "Метаданные изображения:",
        _render_metadata(image_metadata),
        "Заключение:",
        conclusion or _default_conclusion(result_label),
    ]
    return "\n".join(lines)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a UTF-8 capable font, fallback to default if missing."""
    font_candidates = [
        "DejaVuSans.ttf",  # bundled with Pillow
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS wide charset
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _find_font_path() -> str | None:
    """Return a path to a Unicode TTF font if available."""
    font_candidates = [
        "DejaVuSans.ttf",  # bundled with Pillow
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS wide charset
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in font_candidates:
        if Path(candidate).exists():
            return candidate
    return None


def _wrap_text(report_text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap text into lines that fit given pixel width."""
    lines: list[str] = []
    for paragraph in report_text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            trial = f"{current} {word}".strip()
            if font.getlength(trial) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _build_pdf_image(report_text: str, image_path: Path) -> Image.Image:
    """Create a Pillow image containing text report and thumbnail."""
    body_font = _load_font(16)
    heading_font = _load_font(18)
    max_text_width = 760
    lines = _wrap_text(report_text, body_font, max_text_width)

    # Measure text block size
    def line_height(font_obj: ImageFont.ImageFont, text: str) -> int:
        bbox = font_obj.getbbox(text or "Ag")
        return int(bbox[3] - bbox[1])

    heading_height = line_height(heading_font, "Ag")
    body_height = line_height(body_font, "Ag")
    text_width = max((body_font.getlength(line) for line in lines), default=max_text_width)
    text_height = heading_height + len(lines) * (body_height + 4) + 20

    # Load image thumbnail
    try:
        with Image.open(image_path).convert("RGB") as img:
            img.thumbnail((520, 520))
            image_block = img.copy()
    except Exception:
        image_block = Image.new("RGB", (400, 300), color="lightgray")
        d = ImageDraw.Draw(image_block)
        d.text((10, 10), "Изображение не доступно", font=body_font, fill="black")

    # Layout: text on left, image on right
    padding = 24
    column_gap = 24
    canvas_width = int(max(text_width + padding * 2 + image_block.width + column_gap, 900))
    canvas_height = int(max(text_height + padding * 2, image_block.height + padding * 2))
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Draw heading
    y = padding
    draw.text((padding, y), "Отчёт по анализу изображения", font=heading_font, fill="black")
    y += heading_height + 8
    # Draw body
    for line in lines:
        draw.text((padding, y), line, font=body_font, fill="black")
        y += body_height + 4

    # Place image on the right
    img_x = padding + int(max_text_width) + column_gap
    img_y = padding
    canvas.paste(image_block, (img_x, img_y))
    return canvas


def generate_pdf_report(report_text: str, image_path: Path | str, output_path: Path | str) -> None:
    """
    Render and save PDF report with selectable text and embedded image.

    Args:
        report_text: Prepared text report.
        image_path: Path to source image to embed.
        output_path: Destination PDF path.
    """
    if FPDF is None:
        raise ImportError("fpdf2 is required for PDF export with text. Install with 'pip install fpdf2'.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = _find_font_path()
    if font_path:
        pdf.add_font("Body", "", font_path, uni=True)
        pdf.set_font("Body", size=12)
    else:
        pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font(size=14)
    pdf.multi_cell(0, 8, "Отчёт по анализу изображения", align="L")
    pdf.ln(2)
    pdf.set_font(size=12)

    # Text body
    pdf.multi_cell(0, 8, report_text, align="L")
    pdf.ln(4)

    # Embed image scaled to page width
    try:
        with Image.open(image_path) as img:
            w, h = img.size
    except Exception:
        w, h = (640, 480)
        image_path = None

    available_width = getattr(pdf, "epw", pdf.w - pdf.l_margin - pdf.r_margin)
    img_width = min(available_width, 180)
    img_height = img_width * h / w if w else img_width

    # Ensure we have space, otherwise add a new page
    if pdf.get_y() + img_height + 10 > pdf.h - pdf.b_margin:
        pdf.add_page()
    if image_path:
        pdf.image(str(image_path), x=pdf.l_margin, y=pdf.get_y(), w=img_width)
    else:
        pdf.multi_cell(0, 8, "[Изображение недоступно]")

    pdf.output(str(output_path))
