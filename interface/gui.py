from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import cv2
import numpy as np

from interface import main as cli_main
from reporting import generate_pdf_report, generate_text_report


def _build_args(image_path: Path) -> argparse.Namespace:
    """Construct argparse-like namespace for single-image run."""
    return argparse.Namespace(
        image=image_path,
        dataset=None,
        modifying_model=None,
        modifying_ensemble=None,
        skip_modifying=False,
        generating_vit=None,
        generating_convnext=None,
        generating_classifier=None,
        generating_device=None,
        skip_generating=False,
        swapping_models_dir=cli_main.ROOT_DIR / "models" / "swapping",
        skip_swapping=False,
    )


class ModuleCard(ttk.Frame):
    def __init__(self, master: tk.Widget, title: str):
        """Small panel that shows status of a single detector."""
        super().__init__(master, padding=8)
        self.title = title

        self.columnconfigure(1, weight=1)

        self.status_bar = tk.Canvas(self, width=6, height=60, highlightthickness=0)
        self.status_bar.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 8))

        self.title_lbl = ttk.Label(self, text=title, font=("Helvetica", 12, "bold"))
        self.title_lbl.grid(row=0, column=1, sticky="w")

        self.result_var = tk.StringVar(value="—")
        self.result_lbl = ttk.Label(self, textvariable=self.result_var, font=("Helvetica", 11))
        self.result_lbl.grid(row=1, column=1, sticky="w")

        self.time_var = tk.StringVar(value="")
        self.time_lbl = ttk.Label(self, textvariable=self.time_var, foreground="#666")
        self.time_lbl.grid(row=2, column=1, sticky="w")

        self.detail_btn = ttk.Button(self, text="Детали", command=self._show_details, state="disabled")
        self.detail_btn.grid(row=0, column=2, rowspan=2, sticky="e", padx=(8, 0))

        self.details: dict | None = None
        self._set_status_color("#ccc")

    def _set_status_color(self, color: str) -> None:
        """Paint narrow status bar with a given color."""
        self.status_bar.delete("all")
        self.status_bar.create_rectangle(0, 0, 8, 120, fill=color, width=0)

    def update_state(self, result: dict | None) -> None:
        """Update text, color and details block based on detector result."""
        if result is None:
            self.result_var.set("—")
            self.time_var.set("")
            self.details = None
            self.detail_btn.configure(state="disabled")
            self._set_status_color("#ccc")
            return

        status = result.get("status")
        elapsed = result.get("data", {}).get("elapsed")
        color = "#ccc"
        summary = ""

        if status == "ok":
            summary = cli_main._format_data(self.title, result["data"], include_extras=False)
            self.detail_btn.configure(state="normal")
            color = "#2e7d32" if "F" not in summary else "#c62828"
        elif status == "skipped":
            summary = f"Пропущено: {result.get('reason')}"
            self.detail_btn.configure(state="disabled")
            color = "#999"
        else:
            summary = result.get("message", "Ошибка")
            self.detail_btn.configure(state="normal")
            color = "#c62828"

        self.result_var.set(summary)
        if elapsed is not None:
            self.time_var.set(f"Время: {elapsed:.3f}s")
        else:
            self.time_var.set("")
        self.details = result
        self._set_status_color(color)

    def _show_details(self) -> None:
        """Open modal window with full JSON result."""
        if not self.details:
            return
        top = tk.Toplevel(self)
        top.title(f"Детали: {self.title}")
        top.geometry("600x400")
        toolbar = ttk.Frame(top)
        toolbar.pack(fill="x", pady=(4, 0))
        ttk.Button(toolbar, text="Копировать", command=lambda: top.clipboard_append(json.dumps(self.details, ensure_ascii=False, indent=2))).pack(side="left", padx=4)
        text = tk.Text(top, wrap="word", font="TkFixedFont")
        text.pack(fill="both", expand=True, padx=6, pady=4)
        text.insert("1.0", json.dumps(self.details, ensure_ascii=False, indent=2))
        text.configure(state="disabled")


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        """Build main GUI frame and wire callbacks."""
        super().__init__(master, padding=10)
        master.title("Fake Detector GUI")
        master.geometry("900x650")
        self.pack(fill="both", expand=True)
        self.image_label = None
        self.image_obj = None
        self.last_results = None
        self.last_image_path: Path | None = None
        self.last_report_text: str | None = None
        self.last_metadata: dict[str, str] | None = None
        self.image_info_var = tk.StringVar(value="Файл не выбран")

        self._init_styles()
        self._build_ui()

    def _init_styles(self) -> None:
        """Configure base styles for a cleaner look."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#f8fafc")
        style.configure("Card.TFrame", background="#ffffff", relief="ridge", borderwidth=1)
        style.configure("Card.TLabel", background="#ffffff")
        style.configure("CardTitle.TLabel", font=("SF Pro Text", 14, "bold"), background="#ffffff")
        style.configure("Header.TLabel", font=("SF Pro Text", 12), background="#f8fafc")
        style.configure("Primary.TButton", padding=6)
        style.configure("TButton", padding=6)
        style.configure("Meta.TLabel", background="#f8fafc", font=("SF Pro Text", 10))

    def _build_ui(self) -> None:
        """Render all static widgets and layout."""
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 10))

        self.path_var = tk.StringVar()
        top_left = ttk.Frame(top)
        top_left.pack(side="left", fill="x", expand=True)
        file_row = ttk.Frame(top_left)
        file_row.pack(fill="x", pady=2)
        ttk.Button(file_row, text="Выбрать файл", command=self._choose_file).pack(side="left")
        ttk.Entry(file_row, textvariable=self.path_var, width=60).pack(side="left", padx=8, fill="x", expand=True)

        top_right = ttk.Frame(top)
        top_right.pack(side="right", anchor="e")
        buttons_row = ttk.Frame(top_right)
        buttons_row.pack(side="top", anchor="e", pady=2)
        self.run_btn = ttk.Button(buttons_row, text="Запустить анализ", command=self._run_async, style="Primary.TButton")
        self.run_btn.pack(side="left")
        self.report_btn = ttk.Button(buttons_row, text="Посмотреть отчёт", command=self._show_report_window, state="disabled")
        self.report_btn.pack(side="left", padx=(8, 0))
        self.pdf_btn = ttk.Button(buttons_row, text="Выгрузить отчёт в PDF", command=self._save_pdf_report, state="disabled")
        self.pdf_btn.pack(side="left", padx=(4, 0))
        self.status_var = tk.StringVar(value="Готов")
        ttk.Label(top_right, textvariable=self.status_var, foreground="#444", style="Header.TLabel").pack(side="left", padx=10)
        self.progress = ttk.Progressbar(top_right, mode="indeterminate", length=120)

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True)
        content.columnconfigure(1, weight=1)

        self.image_panel = ttk.Label(content, relief="sunken")
        self.image_panel.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 10))
        self.image_info_lbl = ttk.Label(content, textvariable=self.image_info_var, style="Meta.TLabel")
        self.image_info_lbl.grid(row=3, column=0, sticky="w", pady=(4, 0))

        cards_frame = ttk.Frame(content)
        cards_frame.grid(row=0, column=1, sticky="nsew")
        cards_frame.columnconfigure(0, weight=1)

        self.cards = {
            "Modifying": ModuleCard(cards_frame, "Modifying"),
            "Generating": ModuleCard(cards_frame, "Generating"),
            "Swapping": ModuleCard(cards_frame, "Swapping"),
        }
        self.cards["Modifying"].pack(fill="x", pady=4)
        self.cards["Generating"].pack(fill="x", pady=4)
        self.cards["Swapping"].pack(fill="x", pady=4)

        meta_frame = ttk.LabelFrame(self, text="Метаданные изображения")
        meta_frame.pack(fill="x", pady=(8, 0))
        self.meta_container = ttk.Frame(meta_frame, padding=6)
        self.meta_container.pack(fill="both", expand=True)
        self._set_meta_text("Метаданные появятся после анализа.")

    def _choose_file(self) -> None:
        """Show OS file picker and load preview if selected."""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")],
        )
        if file_path:
            self.path_var.set(file_path)
            self._load_preview(Path(file_path))

    def _load_preview(self, path: Path) -> None:
        """Load thumbnail to the left preview panel."""
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((360, 360))
            self.image_obj = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=self.image_obj)
            size_kb = path.stat().st_size / 1024
            self.image_info_var.set(f"{path.name} — {size_kb:.1f} KB — {img.width}x{img.height}")
        except Exception as exc:  # noqa: BLE001
            self.image_panel.configure(text=f"Не удалось загрузить изображение: {exc}", image="")
            self.image_obj = None
            self.image_info_var.set("Не удалось загрузить превью")

    def _run_async(self) -> None:
        """Validate path, pre-check face and kick off background inference."""
        path = Path(self.path_var.get())
        if not path.is_file():
            messagebox.showwarning("Файл не найден", "Выберите существующий файл изображения.")
            return

        has_face = _detect_face_with_yunet(path)
        if not has_face:
            messagebox.showinfo("Лицо не обнаружено", "На выбранном изображении лицо не найдено. Анализ пропущен.")
            for card in self.cards.values():
                card.update_state({"status": "skipped", "reason": "Лицо не найдено"})
            return

        self.run_btn.configure(state="disabled")
        self.report_btn.configure(state="disabled")
        self.pdf_btn.configure(state="disabled")
        self.progress.pack(side="left", padx=8)
        self.progress.start(10)
        self.status_var.set("Выполняется...")
        for card in self.cards.values():
            card.update_state(None)
        self._set_meta_text("Метаданные появятся после анализа.")
        threading.Thread(target=self._run_detection, args=(path,), daemon=True).start()

    def _run_detection(self, path: Path) -> None:
        """Run detectors in a worker thread and propagate results."""
        try:
            args = _build_args(path)
            results = cli_main._run_all(path, args)
            # update UI on main thread
            self.after(0, self._update_results, path, results, None)
        except Exception as exc:  # noqa: BLE001
            self.after(0, self._update_results, path, None, exc)

    def _update_results(self, path: Path, results, error: Exception | None) -> None:
        """Handle completion from worker: update UI or show error."""
        self.run_btn.configure(state="normal")
        self.progress.stop()
        self.progress.pack_forget()
        if error:
            self.status_var.set(f"Ошибка: {error}")
            messagebox.showerror("Ошибка анализа", str(error))
            return
        self.status_var.set("Готово")
        if results:
            for title, result in results:
                if title in self.cards:
                    self.cards[title].update_state(result)
            self.last_results = results
            self.last_image_path = path
            self.last_report_text = self._build_report_text()
            if self.last_metadata:
                self._set_meta_text(self._format_metadata_for_display(self.last_metadata))
            self.report_btn.configure(state="normal")
            self.pdf_btn.configure(state="normal")
        # reload preview in case it was not loaded
        if self.image_obj is None:
            self._load_preview(path)

    def _build_report_text(self) -> str | None:
        """Compose text report from latest results."""
        if not self.last_results or not self.last_image_path:
            return None
        modules = {title: result for title, result in self.last_results}
        label, conf = self._derive_overall_label_conf(modules)
        metadata = self._extract_image_metadata(self.last_image_path)
        self.last_metadata = metadata
        return generate_text_report(
            result_label=label,
            confidence=conf,
            module_details=modules,
            image_metadata=metadata,
        )

    def _derive_overall_label_conf(self, modules: dict) -> tuple[str, float | None]:
        """Heuristic overall decision from module outputs."""
        fake_votes = 0
        total_votes = 0
        probs: list[float] = []
        for name, res in modules.items():
            if not isinstance(res, dict):
                continue
            if res.get("status") != "ok":
                continue
            data = res.get("data", {})
            # Collect probability-like values
            prob_candidates = []
            if isinstance(data.get("probability"), (int, float)):
                prob_candidates.append(float(data["probability"]))
            if isinstance(data.get("prob_of_fake"), (int, float)):
                prob_candidates.append(float(data["prob_of_fake"]))
            final_prob = data.get("final_probability")
            if isinstance(final_prob, dict) and isinstance(final_prob.get("fake_prob"), (int, float)):
                prob_candidates.append(float(final_prob["fake_prob"]))
            if prob_candidates:
                probs.append(max(prob_candidates))
            # Collect fake/real decisions
            decision = None
            if name == "Modifying":
                decision = data.get("label")
                if isinstance(decision, (int, float)):
                    decision = "fake" if decision == 1 else "real"
            elif name == "Generating":
                decision = "fake" if data.get("is_fake") else "real"
            elif name == "Swapping":
                decision = data.get("final_decision")
            if isinstance(decision, str) and decision.lower() in {"fake", "modification"}:
                fake_votes += 1
            if decision is not None:
                total_votes += 1
        total_votes = max(total_votes, 1)  # avoid div by zero
        label = "Подделка" if fake_votes >= (total_votes / 2) else "Оригинал"
        conf = max(probs) if probs else None
        return label, conf

    def _extract_image_metadata(self, path: Path) -> dict[str, str]:
        """Gather lightweight metadata for report."""
        info: dict[str, str] = {"Путь": str(path)}
        try:
            stat = path.stat()
            info["Размер файла"] = f"{stat.st_size / 1024:.1f} KB"
            info["Создано"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime))
            info["Изменено"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        except OSError:
            info["Размер файла"] = "н/д"
        try:
            with Image.open(path) as img:
                info["Формат"] = img.format or "н/д"
                info["MIME"] = Image.MIME.get(img.format, "н/д") if hasattr(Image, "MIME") else "н/д"
                info["Размер"] = f"{img.width}x{img.height}"
                info["Цветовой режим"] = img.mode
                if img.info.get("dpi"):
                    dpi = img.info.get("dpi")
                    if isinstance(dpi, tuple) and len(dpi) >= 2:
                        info["DPI"] = f"{dpi[0]}x{dpi[1]}"
                exif = img.getexif()
                if exif:
                    from PIL import ExifTags

                    tag_map = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
                    field_map = {
                        "DateTimeOriginal": "EXIF Дата/время съёмки",
                        "DateTime": "EXIF Дата/время",
                        "Make": "Камера: производитель",
                        "Model": "Камера: модель",
                        "LensModel": "Объектив",
                        "Software": "ПО",
                        "Orientation": "Ориентация",
                        "ExposureTime": "Выдержка",
                        "FNumber": "Диафрагма",
                        "ISOSpeedRatings": "ISO",
                        "PhotographicSensitivity": "ISO",
                        "FocalLength": "Фокусное расстояние",
                        "ExposureProgram": "Программа экспозиции",
                        "Flash": "Вспышка",
                    }
                    for tag_key, label in field_map.items():
                        if tag_key in tag_map:
                            info[label] = str(tag_map[tag_key])
        except Exception:
            info.setdefault("Формат", "н/д")
        return info

    def _format_metadata_for_display(self, metadata: dict[str, str]) -> str:
        """Convert metadata dict into pretty text for the GUI panel."""
        if not metadata:
            return "Метаданные недоступны."
        return "\n".join(f"{key}: {value}" for key, value in metadata.items())

    def _set_meta_text(self, text: str) -> None:
        """Render metadata text area as read-only."""
        for child in self.meta_container.winfo_children():
            child.destroy()
        lines = text.splitlines()
        if not lines:
            lines = ["Метаданные недоступны."]
        for idx, line in enumerate(lines):
            ttk.Label(self.meta_container, text=line, anchor="w").grid(row=idx, column=0, sticky="w", pady=1)

    def _show_report_window(self) -> None:
        """Display text report in a separate window."""
        if not self.last_report_text:
            messagebox.showinfo("Отчёт недоступен", "Сначала выполните анализ изображения.")
            return
        top = tk.Toplevel(self)
        top.title("Текстовый отчёт")
        top.geometry("700x600")
        text = tk.Text(top, wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("1.0", self.last_report_text)
        text.configure(state="disabled")

    def _save_pdf_report(self) -> None:
        """Render report with image into a PDF file."""
        if not self.last_report_text or not self.last_image_path:
            messagebox.showinfo("Отчёт недоступен", "Сначала выполните анализ изображения.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Сохранить отчёт как PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not save_path:
            return
        try:
            generate_pdf_report(self.last_report_text, self.last_image_path, save_path)
            messagebox.showinfo("Успех", f"PDF сохранён: {save_path}")
        except ImportError as exc:
            messagebox.showerror(
                "Не хватает зависимости",
                f"{exc}\nУстановите пакет fpdf2 и повторите попытку.",
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка сохранения PDF", str(exc))


def _detect_face_with_yunet(image_path: Path) -> bool:
    """Lightweight face presence check using YuNet if available."""
    model_path = cli_main.ROOT_DIR / "swapping" / "face_detection_yunet_2023mar.onnx"
    if not model_path.is_file():
        # модель не найдена — считаем, что лица нет, чтобы явно уведомить пользователя
        return False

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        h, w = img.shape[:2]
        face_detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(w, h),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=1,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(img)
        return faces is not None and len(faces) > 0
    except Exception:
        return False



def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
