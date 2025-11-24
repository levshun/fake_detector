from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import cv2
import numpy as np

from interface import main as cli_main


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
        self.status_bar.delete("all")
        self.status_bar.create_rectangle(0, 0, 8, 120, fill=color, width=0)

    def update_state(self, result: dict | None) -> None:
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
        if not self.details:
            return
        top = tk.Toplevel(self)
        top.title(f"Детали: {self.title}")
        top.geometry("600x400")
        text = tk.Text(top, wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("1.0", json.dumps(self.details, ensure_ascii=False, indent=2))
        text.configure(state="disabled")


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=10)
        master.title("Fake Detector GUI")
        master.geometry("900x650")
        self.pack(fill="both", expand=True)
        self.image_label = None
        self.image_obj = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", pady=(0, 10))

        self.path_var = tk.StringVar()
        ttk.Button(top, text="Выбрать файл", command=self._choose_file).pack(side="left")
        ttk.Entry(top, textvariable=self.path_var, width=60).pack(side="left", padx=8, fill="x", expand=True)
        self.run_btn = ttk.Button(top, text="Запустить анализ", command=self._run_async)
        self.run_btn.pack(side="left")
        self.status_var = tk.StringVar(value="Готов")
        ttk.Label(top, textvariable=self.status_var, foreground="#555").pack(side="left", padx=10)

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True)
        content.columnconfigure(1, weight=1)

        self.image_panel = ttk.Label(content, relief="sunken")
        self.image_panel.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=(0, 10))

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

    def _choose_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")],
        )
        if file_path:
            self.path_var.set(file_path)
            self._load_preview(Path(file_path))

    def _load_preview(self, path: Path) -> None:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((360, 360))
            self.image_obj = ImageTk.PhotoImage(img)
            self.image_panel.configure(image=self.image_obj)
        except Exception as exc:  # noqa: BLE001
            self.image_panel.configure(text=f"Не удалось загрузить изображение: {exc}", image="")
            self.image_obj = None

    def _run_async(self) -> None:
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
        self.status_var.set("Выполняется...")
        for card in self.cards.values():
            card.update_state(None)
        threading.Thread(target=self._run_detection, args=(path,), daemon=True).start()

    def _run_detection(self, path: Path) -> None:
        try:
            args = _build_args(path)
            results = cli_main._run_all(path, args)
            # update UI on main thread
            self.after(0, self._update_results, path, results, None)
        except Exception as exc:  # noqa: BLE001
            self.after(0, self._update_results, path, None, exc)

    def _update_results(self, path: Path, results, error: Exception | None) -> None:
        self.run_btn.configure(state="normal")
        if error:
            self.status_var.set(f"Ошибка: {error}")
            messagebox.showerror("Ошибка анализа", str(error))
            return
        self.status_var.set("Готово")
        if results:
            for title, result in results:
                if title in self.cards:
                    self.cards[title].update_state(result)
        # reload preview in case it was not loaded
        if self.image_obj is None:
            self._load_preview(path)


def _detect_face_with_yunet(image_path: Path) -> bool:
    """Lightweight face presence check using YuNet if available."""
    model_path = cli_main.ROOT_DIR / "swapping" / "face_detection_yunet_2023mar.onnx"
    if not model_path.is_file():
        # модель не найдена, считаем что лицо есть, чтобы не блокировать анализ
        return True

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
        # в случае ошибки детектора не блокируем запуск
        return True



def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
