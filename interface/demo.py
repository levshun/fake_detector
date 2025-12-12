import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import io
import time
import csv
from typing import Any, Dict, Iterable, List, Tuple

import detect_ai as dai


os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ.pop("TF_USE_LEGACY_KERAS", None)

ROOT_DIR = os.path.join('..', 'models')
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_GENERATING_DIR = os.path.join(ROOT_DIR, 'generating')
DEFAULT_GENERATING_VIT = os.path.join(DEFAULT_GENERATING_DIR, "eva_model.pth")
DEFAULT_GENERATING_CONVNEXT = os.path.join(DEFAULT_GENERATING_DIR, "convnext_model.pth")
DEFAULT_GENERATING_CLASSIFIER = os.path.join(DEFAULT_GENERATING_DIR, "final_decisiontree.pkl")

DEFAULT_MODIFYING_DIR = os.path.join(ROOT_DIR, 'modifying')
DEFAULT_MODIFYING_MODEL = os.path.join(DEFAULT_MODIFYING_DIR, "binary", "beauty_gan", "eff_net_b3", "eff_net_b3.keras")
DEFAULT_MODIFYING_ENSEMBLE = dai.mod_ens_dict

DEFAULT_SWAPPING_DIR = os.path.join(ROOT_DIR, 'swapping')

DEFAULT_DATASET_DIR = os.path.join('..', 'datasets', 'generating')

if not os.path.exists('log'):
    os.mkdir('log')

LOG_PATH = os.path.join('log', "run.log")
TABL_PATH = os.path.join('log', "tabl.log")
TABL_CSV_PATH = os.path.join('log', "tabl.csv")
MOD_JSON_PATH = os.path.join('log', "modifying.json")
GEN_JSON_PATH = os.path.join('log', "generating.json")
SWAP_JSON_PATH = os.path.join('log', "swapping.json")

Result = Dict[str, Any]
_MOD_CACHE: Dict[str, Any] = {}
_GEN_CACHE: Dict[Tuple[str, str, str, str], Any] = {}
_SWAP_CACHE: Dict[str, dai.DeepfakePredictor] = {}
_INIT_TIMINGS: Dict[str, float] = {"Modifying": 0.0, "Generating": 0.0, "Swapping": 0.0}
_JSON_WRITERS: Dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверка одного изображения или всех изображений в каталоге datasets."
    )
    parser.add_argument("image", nargs="?", type=Path, help="Путь к одному изображению для анализа")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Если указан, скрипт обработает все изображения из этой папки (рекурсивно не ходим).",
    )

    mod_group = parser.add_argument_group("Modifying")
    mod_group.add_argument(
        "--modifying-model",
        type=Path,
        help=f"Путь к одиночной модели (.keras/.pkl). По умолчанию {DEFAULT_MODIFYING_MODEL} если существует.",
    )
    mod_group.add_argument(
        "--modifying-ensemble",
        type=Path,
        help=f"Путь к JSON-конфигу ансамбля детекторов модификаций. По умолчанию {DEFAULT_MODIFYING_ENSEMBLE} если существует.",
    )
    mod_group.add_argument(
        "--skip-modifying",
        action="store_true",
        help="Пропустить запуск детектора модификаций",
    )

    gen_group = parser.add_argument_group("Generating detector")
    gen_group.add_argument(
        "--generating-vit",
        type=Path,
        help=f"Путь к весам ViT (.pth). По умолчанию {DEFAULT_GENERATING_VIT} если существует.",
    )
    gen_group.add_argument(
        "--generating-convnext",
        type=Path,
        help=f"Путь к весам ConvNeXt (.pth). По умолчанию {DEFAULT_GENERATING_CONVNEXT} если существует.",
    )
    gen_group.add_argument(
        "--generating-classifier",
        type=Path,
        help=f"Путь к финальному классификатору (.pkl). По умолчанию {DEFAULT_GENERATING_CLASSIFIER} если существует.",
    )
    gen_group.add_argument(
        "--generating-device",
        default=None,
        help="Устройство torch (например, cuda или cpu). По умолчанию выбирается автоматически.",
    )
    gen_group.add_argument(
        "--skip-generating",
        action="store_true",
        help="Пропустить запуск детектора сгенерированных изображений",
    )

    swap_group = parser.add_argument_group("Swapping detector")
    swap_group.add_argument(
        "--swapping-models-dir",
        type=Path,
        default=DEFAULT_SWAPPING_DIR,
        help=f"Каталог с моделями модуля swapping (по умолчанию '{DEFAULT_SWAPPING_DIR}')",
    )
    swap_group.add_argument(
        "--skip-swapping",
        action="store_true",
        help="Пропустить запуск детектора подмены лица",
    )

    args = parser.parse_args()
    if args.dataset:
        if not os.path.isdir(args.dataset):
            parser.error(f"Каталог datasets не найден: {args.dataset}")
    elif args.image is None:
        parser.error("Нужно указать файл изображения или --dataset <папка>.")
    elif not args.image.is_file():
        parser.error(f"Файл изображения не найден: {args.image}")
    return args


def _resolve_generating_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    vit = args.generating_vit
    convnext = args.generating_convnext
    clf = args.generating_classifier

    if vit is None and DEFAULT_GENERATING_VIT:
        vit = DEFAULT_GENERATING_VIT
    if convnext is None and DEFAULT_GENERATING_CONVNEXT:
        convnext = DEFAULT_GENERATING_CONVNEXT
    if clf is None and DEFAULT_GENERATING_CLASSIFIER:
        clf = DEFAULT_GENERATING_CLASSIFIER
    return vit, convnext, clf


def _resolve_modifying_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    model = args.modifying_model
    ensemble = args.modifying_ensemble
    if model is None and DEFAULT_MODIFYING_MODEL:
        model = DEFAULT_MODIFYING_MODEL
    if ensemble is None and model is None and DEFAULT_MODIFYING_ENSEMBLE.exists():
        ensemble = DEFAULT_MODIFYING_ENSEMBLE
    return model, ensemble


def _prepare_ensemble_config(config_path: Path) -> tuple[Path, Path | None]:
    """
    Returns path to ensemble config adapted to current repo layout.
    If rewriting is needed, returns path to temp file plus path for cleanup.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception:
        return config_path, None

    model_paths = config.get("model_paths")
    if not isinstance(model_paths, list):
        return config_path, None

    updated = False
    resolved_paths: list[str] = []
    for path_str in model_paths:
        candidate = Path(path_str)
        if candidate.is_file():
            resolved_paths.append(str(candidate))
            continue

        alt_paths = [
            ROOT_DIR / path_str,
            ROOT_DIR / "modifying" / path_str,
            config_path.parent / path_str,
        ]
        resolved = None
        for alt in alt_paths:
            if alt.is_file():
                resolved = alt
                break
        if resolved is not None:
            updated = True
            resolved_paths.append(str(resolved))
        else:
            resolved_paths.append(path_str)

    if not updated:
        return config_path, None

    config["model_paths"] = resolved_paths
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    try:
        json.dump(config, tmp)
    finally:
        tmp.close()
    return Path(tmp.name), Path(tmp.name)


def _get_modifying_detector(
    model_path: Path | None, ensemble_path: Path | None
) -> Any:
    if ensemble_path:
        resolved_path, cleanup_path = _prepare_ensemble_config(ensemble_path)
        key = f"ensemble:{resolved_path}"
        if key not in _MOD_CACHE:
            start = time.perf_counter()
            try:
                _MOD_CACHE[key] = dai.mod_start_ensemble(dai.mod_ens_dict)
            finally:
                if cleanup_path and cleanup_path.exists():
                    cleanup_path.unlink()
            _INIT_TIMINGS["Modifying"] += time.perf_counter() - start
        return _MOD_CACHE[key]
    assert model_path is not None
    key = f"model:{model_path}"
    if key not in _MOD_CACHE:
        start = time.perf_counter()
        _MOD_CACHE[key] = dai.mod_start_model(model_path=str(model_path))
        _INIT_TIMINGS["Modifying"] += time.perf_counter() - start
    return _MOD_CACHE[key]


def _get_generating_detector(
    vit_path: Path, convnext_path: Path, clf_path: Path, device: str | None
) -> dai.GeneratedImageDetector:
    key = (str(vit_path), str(convnext_path), str(clf_path), device or "auto")
    if key not in _GEN_CACHE:
        start = time.perf_counter()
        _GEN_CACHE[key] = dai.GeneratedImageDetector(
            vit_model_path=key[0],
            convnext_model_path=key[1],
            dt_model=key[2],
            device=None if key[3] == "auto" else key[3],
        )
        _INIT_TIMINGS["Generating"] += time.perf_counter() - start
    return _GEN_CACHE[key]


def _get_swapping_predictor(models_dir: str) -> dai.DeepfakePredictor:
    key = models_dir
    print('key:', key)
    if key not in _SWAP_CACHE:
        start = time.perf_counter()
        _SWAP_CACHE[key] = dai.DeepfakePredictor(models_dir=key)
        _INIT_TIMINGS["Swapping"] += time.perf_counter() - start
    return _SWAP_CACHE[key]


def run_modifying(image_path: Path, args: argparse.Namespace) -> Result:
    if args.skip_modifying:
        return {"status": "skipped", "reason": "Запуск модуля modifying отключён флагом."}
    model_path, ensemble_path = _resolve_modifying_paths(args)
    if not model_path and not ensemble_path:
        return {
            "status": "skipped",
            "reason": "Не указаны пути --modifying-model или --modifying-ensemble.",
        }
    try:
        detector = _get_modifying_detector(model_path, ensemble_path)
        start_time = time.perf_counter()
        data = detector.run(str(image_path))
        data["elapsed"] = time.perf_counter() - start_time
        return {"status": "ok", "data": data}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Modifying detector error: {exc}"}


def run_generating(image_path: Path, args: argparse.Namespace) -> Result:
    if args.skip_generating:
        return {"status": "skipped", "reason": "Запуск модуля generating отключён флагом."}
    vit_path, convnext_path, clf_path = _resolve_generating_paths(args)
    if not all([vit_path, convnext_path, clf_path]):
        return {
            "status": "skipped",
            "reason": "Для запуска необходимы --generating-vit, --generating-convnext и --generating-classifier.",
        }
    try:
        detector = _get_generating_detector(vit_path, convnext_path, clf_path, args.generating_device)
        start = time.perf_counter()
        data = detector.get_final_score(str(image_path))
        data["elapsed"] = time.perf_counter() - start
        return {"status": "ok", "data": data}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Generating detector error: {exc}"}


def run_swapping(image_path: Path, args: argparse.Namespace) -> Result:
    if args.skip_swapping:
        return {"status": "skipped", "reason": "Запуск модуля swapping отключён флагом."}
    models_dir = args.swapping_models_dir
    print('run_swapping models_dir:', models_dir)
    if not os.path.isdir(models_dir):
        print('not os.path.isdir(models_dir)')
        return {
            "status": "skipped",
            "reason": f"Каталог с моделями для swapping не найден: {models_dir}",
        }
    try:
        predictor = _get_swapping_predictor(models_dir)
        start = time.perf_counter()
        data = predictor.predict(image_path=str(image_path))
        data["elapsed"] = time.perf_counter() - start
        return {"status": "ok", "data": data}
    except dai.SwappingError as exc:
        return {"status": "error", "message": f"Swapping detector error: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"Swapping detector unexpected error: {exc}"}


def print_section(title: str, result: Result, image_path: Path) -> None:
    print(f"\n=== {title} ===")
    status = result.get("status")
    if status == "ok":
        print(_format_data(title, result["data"]))
        _print_json(title, result, image_path)
    elif status == "skipped":
        print(f"Пропущено: {result.get('reason')}")
    else:
        print(result.get("message", "Неизвестная ошибка."))


def _run_all(image_path: Path, args: argparse.Namespace) -> Tuple[Tuple[str, Result], ...]:
    return (
        ("Modifying", run_modifying(image_path, args)),
        ("Generating", run_generating(image_path, args)),
        ("Swapping", run_swapping(image_path, args)),
    )


def _collect_images(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in exts and p.is_file()])


def _summarize_result(title: str, result: Result, include_extras: bool = True) -> str:
    status = result.get("status")
    if status == "ok":
        return _format_data(title, result["data"], include_extras=include_extras)
    if status == "skipped":
        return f"skip: {result.get('reason')}"
    return f"err: {result.get('message')}"


def _print_table(rows: List[List[str]], headers: Iterable[str], file: io.TextIOBase | None = None) -> None:
    headers_list = list(headers)
    table = [headers_list] + rows
    widths = [max(len(str(row[i])) for row in table) for i in range(len(headers_list))]

    def fmt(row: List[str]) -> str:
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(widths)))

    print(fmt(headers_list), file=file)
    print("-+-".join("-" * w for w in widths), file=file)
    for row in rows:
        print(fmt(row), file=file)


def _write_table_snapshot(rows: List[List[str]], headers: Iterable[str], file: io.TextIOBase) -> None:
    file.seek(0)
    file.truncate()
    _print_table(rows, headers, file=file)
    file.flush()


def _write_csv_snapshot(rows: List[List[str]], headers: Iterable[str], file: io.TextIOBase) -> None:
    file.seek(0)
    file.truncate()
    writer = csv.writer(file, delimiter=";")
    writer.writerow(list(headers))
    writer.writerows(rows)
    file.flush()


def _extract_decision_prob(title: str, result: Result) -> tuple[str, str]:
    status = result.get("status")
    if status == "ok":
        data = result["data"]
        if title == "Modifying":
            label = data.get("label", "unknown")
            is_fake = False
            if isinstance(label, str) and label.lower() in {"modification", "fake"}:
                is_fake = True
            if isinstance(label, (int, float)) and label == 1:
                is_fake = True
            prob = data.get("probability")
            if isinstance(prob, (int, float)):
                fake_prob = float(prob) if is_fake else max(0.0, min(1.0, 1.0 - float(prob)))
                return ("F" if is_fake else "R", f"{fake_prob:.4f}".replace(".", ","))
            return ("F" if is_fake else "R", "-")
        if title == "Generating":
            prob = data.get("prob_of_fake")
            label = data.get("is_fake")
            dec = "F" if label else "R"
            if isinstance(prob, (int, float)):
                return dec, f"{float(prob):.4f}".replace(".", ",")
            return dec, "-"
        if title == "Swapping":
            decision = data.get("final_decision")
            dec = "F" if isinstance(decision, str) and decision.lower() == "fake" else "R"
            final_prob = data.get("final_probability", {})
            fake_prob = None
            if isinstance(final_prob, dict):
                fake_prob = final_prob.get("fake_prob")
            if isinstance(fake_prob, (int, float)):
                return dec, f"{float(fake_prob):.4f}".replace(".", ",")
            return dec, "-"
    if status == "skipped":
        return ("S", "-")
    return ("E", "-")


def _extract_swap_extras(result: Result) -> list[str]:
    placeholders = ["-"] * 8
    if result.get("status") != "ok":
        return placeholders
    data = result.get("data", {})
    base = data.get("base_models_prob", {})
    effnet = data.get("specialized_effnet_prob", {})

    def get_fake_prob(src: dict, key: str) -> str:
        val = src.get(key)
        if isinstance(val, dict):
            fp = val.get("fake_prob")
            if isinstance(fp, (int, float)):
                return f"{fp:.4f}".replace(".", ",")
        return "-"

    bm_ef = get_fake_prob(base, "ef_rf")
    bm_fl = get_fake_prob(base, "fl_rf")
    bm_lbp = get_fake_prob(base, "lbp_cb")
    bm_tf = get_fake_prob(base, "tf_sf_cb")

    eff_seg = get_fake_prob(effnet, "segmind")
    eff_git = get_fake_prob(effnet, "github")
    eff_rgb = get_fake_prob(effnet, "rgb")
    eff_roop = get_fake_prob(effnet, "roop")

    return [bm_ef, bm_fl, bm_lbp, bm_tf, eff_seg, eff_git, eff_rgb, eff_roop]


def _format_data(title: str, data: dict[str, Any], include_extras: bool = True) -> str:
    if title == "Modifying":
        label = data.get("label", "unknown")
        is_fake = False
        if isinstance(label, str) and label.lower() in {"modification", "fake"}:
            is_fake = True
        if isinstance(label, (int, float)) and label == 1:
            is_fake = True

        prob = data.get("probability")
        if isinstance(prob, (int, float)):
            fake_prob = float(prob) if is_fake else max(0.0, min(1.0, 1.0 - float(prob)))
            decision = "F" if is_fake else "R"
            return f"dec={decision}({fake_prob:.4f})"

        decision = "F" if is_fake else "R"
        return f"dec={decision}"
    if title == "Generating":
        prob = data.get("prob_of_fake")
        label = data.get("is_fake")
        decision = "F" if label else "R"
        if isinstance(prob, (int, float)):
            return f"dec={decision}({float(prob):.4f})"
        return f"dec={decision}"
    if title == "Swapping":
        decision = data.get("final_decision")
        final_prob = data.get("final_probability", {})
        if decision:
            prob_value = None
            if isinstance(final_prob, dict):
                if isinstance(decision, str) and decision.lower() == "fake":
                    prob_value = final_prob.get("fake_prob")
                elif isinstance(decision, str) and decision.lower() == "real":
                    prob_value = final_prob.get("fake_prob")
            extras: list[str] = []
            if include_extras:
                # Base models fake probabilities
                base = data.get("base_models_prob")
                if isinstance(base, dict):
                    short_keys = {"ef_rf": "ef", "fl_rf": "fl", "lbp_cb": "lbp", "tf_sf_cb": "tf"}
                    parts = []
                    for key, short in short_keys.items():
                        val = base.get(key, {}) if isinstance(base.get(key), dict) else None
                        fake_val = val.get("fake_prob") if isinstance(val, dict) else None
                        if isinstance(fake_val, (int, float)):
                            parts.append(f"{short}={fake_val:.2f}")
                    if parts:
                        extras.append(f"bm[{','.join(parts)}]")
                # Specialized effnet fake probabilities
                effnet = data.get("specialized_effnet_prob")
                if isinstance(effnet, dict):
                    short_keys = {"segmind": "seg", "github": "git", "rgb": "rgb", "roop": "roop"}
                    parts = []
                    for key, short in short_keys.items():
                        val = effnet.get(key, {}) if isinstance(effnet.get(key), dict) else None
                        fake_val = val.get("fake_prob") if isinstance(val, dict) else None
                        if isinstance(fake_val, (int, float)):
                            parts.append(f"{short}={fake_val:.2f}")
                    if parts:
                        extras.append(f"eff[{','.join(parts)}]")

            if prob_value is not None:
                try:
                    dec_char = "F" if isinstance(decision, str) and decision.lower() == "fake" else "R"
                    return "dec={0}({1:.4f}){2}".format(
                        dec_char, float(prob_value), f",{','.join(extras)}" if extras else ""
                    )
                except (TypeError, ValueError):
                    pass
            if extras:
                dec_char = "F" if isinstance(decision, str) and decision.lower() == "fake" else "R"
                return f"dec={dec_char},{','.join(extras)}"
            dec_char = "F" if isinstance(decision, str) and decision.lower() == "fake" else "R"
            return f"dec={dec_char}"
    return json.dumps(data, ensure_ascii=False)


def _format_elapsed(result: Result) -> str:
    if result.get("status") != "ok":
        return "-"
    elapsed = result["data"].get("elapsed")
    if elapsed is None:
        return "-"
    return f"{elapsed:.3f}"


def _print_init_timings() -> None:
    print("Init timings (s):")
    for title, value in _INIT_TIMINGS.items():
        print(f"  {title[:4]:<4}: {value:.3f}")


def _format_hms(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _warm_up(args: argparse.Namespace) -> None:
    if not args.skip_modifying:
        model_path, ensemble_path = _resolve_modifying_paths(args)
        if model_path or ensemble_path:
            _get_modifying_detector(model_path, ensemble_path)
    if not args.skip_generating:
        vit_path, convnext_path, clf_path = _resolve_generating_paths(args)
        if all([vit_path, convnext_path, clf_path]):
            _get_generating_detector(vit_path, convnext_path, clf_path, args.generating_device)
    if not args.skip_swapping and os.path.isdir(args.swapping_models_dir):
        _get_swapping_predictor(args.swapping_models_dir)

def _print_json(title: str, result: Result, image_path: Path) -> None:
    if result.get("status") != "ok":
        return
    payload = {"module": title, "image": str(image_path), **result["data"]}
    print(json.dumps(payload, ensure_ascii=False))
    writer = _JSON_WRITERS.get(title)
    if writer:
        writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
        writer.flush()


class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> None:
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    tabl_file = open(TABL_PATH, "w", encoding="utf-8")
    tabl_csv_file = open(TABL_CSV_PATH, "w", encoding="utf-8", newline="")
    json_files = {
        "Modifying": open(MOD_JSON_PATH, "w", encoding="utf-8"),
        "Generating": open(GEN_JSON_PATH, "w", encoding="utf-8"),
        "Swapping": open(SWAP_JSON_PATH, "w", encoding="utf-8"),
    }
    _JSON_WRITERS.update(json_files)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    try:
        program_start = time.perf_counter()
        program_start_abs = time.time()
        args = parse_args()
        if args.dataset:
            images = _collect_images(args.dataset)
            if not images:
                print(f"Нет изображений в каталоге {args.dataset}")
                return
            _warm_up(args)
            rows_display: List[List[str]] = []
            rows_log: List[List[str]] = []
            rows_csv: List[List[str]] = []
            total = len(images)
            headers = ["Image", "Mod", "t_mod", "Gen", "t_gen", "Swap", "t_swap"]
            headers_csv = [
                "Image",
                "Mod_dec",
                "Mod_p",
                "t_mod",
                "Gen_dec",
                "Gen_p",
                "t_gen",
                "Swap_dec",
                "Swap_p",
                "Swap_bm_ef",
                "Swap_bm_fl",
                "Swap_bm_lbp",
                "Swap_bm_tf",
                "Swap_eff_seg",
                "Swap_eff_git",
                "Swap_eff_rgb",
                "Swap_eff_roop",
                "t_swap",
            ]
            processed_count = 0
            total_elapsed_images = 0.0
            for idx, image in enumerate(images, 1):
                image_start = time.perf_counter()
                rel_time = image_start - program_start
                abs_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                est_rel = None
                est_abs = None
                if processed_count > 0:
                    avg_time = total_elapsed_images / processed_count
                    remaining = total - idx + 1
                    est_rel = avg_time * remaining  # remaining time
                    est_abs_ts = program_start_abs + rel_time + est_rel
                    est_abs = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(est_abs_ts))
                print("-" * 120)
                if est_rel is not None and est_abs is not None:
                    print(
                        f"({idx}/{total}) {image.name} start: t_rel={_format_hms(rel_time)}, t_abs={abs_time}, "
                        f"est_end: t_left~{_format_hms(est_rel)}, t_abs~{est_abs}"
                    )
                else:
                    print(f"({idx}/{total}) {image.name} start: t_rel={_format_hms(rel_time)}, t_abs={abs_time}")
                sections = _run_all(image, args)
                for title, result in sections:
                    _print_json(title, result, image)
                row_display = [image.name]
                row_log = [str(image)]
                row_csv = [str(image)]
                for title, result in sections:
                    row_display.append(_summarize_result(title, result, include_extras=False))
                    row_display.append(_format_elapsed(result))
                    row_log.append(_summarize_result(title, result, include_extras=True))
                    row_log.append(_format_elapsed(result))
                    dec, prob = _extract_decision_prob(title, result)
                    if title == "Modifying":
                        row_csv.extend([dec, prob, _format_elapsed(result)])
                    elif title == "Generating":
                        row_csv.extend([dec, prob, _format_elapsed(result)])
                    elif title == "Swapping":
                        extras = _extract_swap_extras(result)
                        row_csv.extend([dec, prob, *extras, _format_elapsed(result)])
                rows_display.append(row_display)
                rows_log.append(row_log)
                rows_csv.append(row_csv)
                image_elapsed = time.perf_counter() - image_start
                total_elapsed_images += image_elapsed
                processed_count += 1
                _write_table_snapshot(rows_log, headers, tabl_file)
                _write_csv_snapshot(rows_csv, headers_csv, tabl_csv_file)
            _print_init_timings()
            _print_table(rows_display, headers)
            _print_table(rows_log, headers, file=log_file)
            _write_table_snapshot(rows_log, headers, tabl_file)
            _write_csv_snapshot(rows_csv, headers_csv, tabl_csv_file)
        else:
            sections = _run_all(args.image, args)
            print()
            for title, result in sections:
                print_section(title, result, args.image)
            print()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        log_file.flush()
        tabl_file.flush()
        tabl_csv_file.flush()
        for writer in json_files.values():
            writer.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        tabl_file.close()
        tabl_csv_file.close()
        for writer in json_files.values():
            writer.close()


if __name__ == "__main__":
    main()
