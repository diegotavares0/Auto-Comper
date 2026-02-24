"""
Auto-Comper v1.0 — Flask application with API routes and SSE progress.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
import traceback
import uuid

import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory, Response

from backend.config import CompRules, TunerConfig, PresetConfig
from backend.utils.audio_io import load_audio_file, save_audio
from backend.utils.progress import progress_manager
from backend.engine.pipeline import run_autocomp
from backend.tuner.pipeline import run_tuner
from backend.presets.pipeline import create_preset, apply_preset
from backend.presets import manager as preset_manager
from backend.presets import neural as neural_module

log = logging.getLogger("comper")

# Temp directories
OUTPUT_DIR = tempfile.mkdtemp(prefix="autocomper_out_")

# Store results from background tasks (task_id -> result dict)
_task_results = {}
_results_lock = threading.Lock()


def create_app() -> Flask:
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

    app = Flask(__name__, static_folder=frontend_path, static_url_path="")

    # --- Static file serving ---

    @app.route("/")
    def index():
        return send_from_directory(frontend_path, "index.html")

    @app.route("/css/<path:filename>")
    def css(filename):
        return send_from_directory(os.path.join(frontend_path, "css"), filename)

    @app.route("/js/<path:filename>")
    def js(filename):
        return send_from_directory(os.path.join(frontend_path, "js"), filename)

    # --- Output file serving ---

    @app.route("/api/output/<filename>")
    def output_file(filename):
        fpath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(fpath):
            return send_from_directory(OUTPUT_DIR, filename, mimetype="audio/wav")
        return jsonify({"error": "File not found"}), 404

    # --- SSE Progress ---

    @app.route("/api/progress/<task_id>")
    def progress_stream(task_id):
        q = progress_manager.subscribe(task_id)

        def generate():
            try:
                yield from progress_manager.stream(q)
            finally:
                progress_manager.unsubscribe(task_id, q)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- Task Result Fetch ---

    @app.route("/api/result/<task_id>")
    def task_result(task_id):
        with _results_lock:
            result = _task_results.get(task_id)

        if result is None:
            return jsonify({"status": "processing"}), 202

        if result.get("error"):
            return jsonify({"error": result["error"]}), 500

        response = {
            "filename": result["filename"],
            "report": result["report"],
            "task_id": task_id,
        }
        # Include original filename for A/B comparison (tuning)
        if result.get("original_filename"):
            response["original_filename"] = result["original_filename"]

        return jsonify(response)

    # --- Comping API ---

    @app.route("/api/comp", methods=["POST"])
    def comp():
        task_id = str(uuid.uuid4())[:8]
        progress_manager.start(task_id, "Recebendo arquivos...")

        try:
            files = request.files.getlist("files")
            if len(files) < 2:
                return jsonify({"error": "Minimo 2 takes necessarios"}), 400

            # Save uploaded files
            take_dir = tempfile.mkdtemp(prefix="takes_")
            for f in files:
                fpath = os.path.join(take_dir, f.filename)
                f.save(fpath)

            # Build rules from form data
            rules = CompRules()
            rules.segment_method = request.form.get("segment_method", "musical")
            rules.crossfade_ms = float(request.form.get("crossfade_ms", 50))
            rules.use_alignment = request.form.get("alignment", "xcorr") != "none"
            rules.switch_penalty = float(request.form.get("switch_penalty", 0.15))
            rules.min_improvement_to_switch = float(request.form.get("min_improvement", 0.08))
            rules.max_takes_in_comp = int(request.form.get("max_takes", 4))

            # Custom sections from Structure tab (manual block boundaries)
            custom_sec = request.form.get("custom_sections", "")
            if custom_sec:
                try:
                    rules.custom_sections = json.loads(custom_sec)
                except Exception:
                    rules.custom_sections = None

            # Structure-aware sections (from Structure detection tab)
            struct_sec = request.form.get("structure_sections", "")
            if struct_sec:
                try:
                    rules.structure_sections = json.loads(struct_sec)
                    # Auto-switch to structure mode when sections are provided
                    rules.segment_method = "structure"
                except Exception:
                    rules.structure_sections = None

            # Load takes
            sr = rules.sample_rate
            takes = []
            wav_files = sorted([
                f for f in os.listdir(take_dir)
                if f.lower().endswith(('.wav', '.flac', '.aiff', '.aif'))
            ])
            for fname in wav_files:
                audio = load_audio_file(os.path.join(take_dir, fname), sr)
                takes.append(audio)

            # Run comping in background thread (non-blocking)
            def run():
                try:
                    callback = progress_manager.make_callback(task_id)
                    comp_audio, report = run_autocomp(takes, sr, rules, callback)

                    # Save output
                    out_name = f"comp_{task_id}.wav"
                    out_path = os.path.join(OUTPUT_DIR, out_name)
                    save_audio(out_path, comp_audio, sr)

                    with _results_lock:
                        _task_results[task_id] = {
                            "filename": out_name,
                            "report": report,
                        }

                    progress_manager.complete(task_id, "Comp finalizado!")

                except Exception as e:
                    traceback.print_exc()
                    with _results_lock:
                        _task_results[task_id] = {"error": str(e)}
                    progress_manager.error(task_id, str(e))

                finally:
                    shutil.rmtree(take_dir, ignore_errors=True)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            # Return immediately with task_id — client subscribes to SSE
            return jsonify({"task_id": task_id})

        except Exception as e:
            progress_manager.error(task_id, str(e))
            traceback.print_exc()
            return jsonify({"error": str(e), "task_id": task_id}), 500

    # --- Tuning API ---

    @app.route("/api/tune", methods=["POST"])
    def tune():
        task_id = str(uuid.uuid4())[:8]
        progress_manager.start(task_id, "Recebendo arquivo...")

        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "Arquivo de audio necessario"}), 400

            # Save uploaded file
            tmp_dir = tempfile.mkdtemp(prefix="tuner_")
            fpath = os.path.join(tmp_dir, file.filename)
            file.save(fpath)

            # Build TunerConfig from form data
            config = TunerConfig()
            config.instrument_mode = request.form.get("instrument_mode", "voice")
            config.correction_amount = float(request.form.get("correction_amount", 80))
            config.retune_speed = float(request.form.get("retune_speed", 50))
            config.root_note = request.form.get("root_note", "auto")
            config.scale_type = request.form.get("scale_type", "major")
            config.preserve_vibrato = request.form.get(
                "preserve_vibrato", "true").lower() == "true"
            config.normalize_output = request.form.get(
                "normalize_output", "true").lower() == "true"

            # Load audio (mono, resampled)
            sr = config.sample_rate
            audio = load_audio_file(fpath, sr)

            # Run tuning in background thread (non-blocking)
            def run():
                try:
                    callback = progress_manager.make_callback(task_id)
                    tuned_audio, report = run_tuner(audio, sr, config, callback)

                    # Save both original and tuned for A/B comparison
                    orig_name = f"tune_orig_{task_id}.wav"
                    tuned_name = f"tune_{task_id}.wav"
                    save_audio(os.path.join(OUTPUT_DIR, orig_name), audio, sr)
                    save_audio(os.path.join(OUTPUT_DIR, tuned_name), tuned_audio, sr)

                    with _results_lock:
                        _task_results[task_id] = {
                            "filename": tuned_name,
                            "original_filename": orig_name,
                            "report": report,
                        }

                    progress_manager.complete(task_id, "Tuning finalizado!")

                except Exception as e:
                    traceback.print_exc()
                    with _results_lock:
                        _task_results[task_id] = {"error": str(e)}
                    progress_manager.error(task_id, str(e))

                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            return jsonify({"task_id": task_id})

        except Exception as e:
            progress_manager.error(task_id, str(e))
            traceback.print_exc()
            return jsonify({"error": str(e), "task_id": task_id}), 500

    # --- Presets API ---

    @app.route("/api/presets", methods=["GET"])
    def list_presets():
        presets = preset_manager.list_presets()
        return jsonify({
            "presets": presets,
            "neural_available": neural_module.is_available(),
        })

    @app.route("/api/presets", methods=["POST"])
    def create_preset_api():
        task_id = str(uuid.uuid4())[:8]
        progress_manager.start(task_id, "Recebendo referencia...")

        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "Arquivo de referencia necessario"}), 400

            name = request.form.get("name", "").strip()
            if not name:
                return jsonify({"error": "Nome do preset necessario"}), 400

            # Save uploaded file
            tmp_dir = tempfile.mkdtemp(prefix="preset_ref_")
            fpath = os.path.join(tmp_dir, file.filename)
            file.save(fpath)

            # Load audio
            sr = 48000
            audio = load_audio_file(fpath, sr)

            # Create preset in background
            def run():
                try:
                    callback = progress_manager.make_callback(task_id)
                    meta = create_preset(
                        audio, sr, name,
                        reference_audio_path=fpath,
                        progress_callback=callback,
                    )

                    with _results_lock:
                        _task_results[task_id] = {
                            "filename": "",
                            "report": meta,
                        }

                    progress_manager.complete(
                        task_id, f"Preset '{name}' criado!"
                    )

                except Exception as e:
                    traceback.print_exc()
                    with _results_lock:
                        _task_results[task_id] = {"error": str(e)}
                    progress_manager.error(task_id, str(e))

                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            return jsonify({"task_id": task_id})

        except Exception as e:
            progress_manager.error(task_id, str(e))
            traceback.print_exc()
            return jsonify({"error": str(e), "task_id": task_id}), 500

    @app.route("/api/presets/<preset_id>", methods=["DELETE"])
    def delete_preset_api(preset_id):
        if preset_manager.delete_preset(preset_id):
            return jsonify({"ok": True})
        return jsonify({"error": "Preset nao encontrado"}), 404

    @app.route("/api/apply-preset", methods=["POST"])
    def apply_preset_api():
        task_id = str(uuid.uuid4())[:8]
        progress_manager.start(task_id, "Recebendo arquivo...")

        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "Arquivo de audio necessario"}), 400

            preset_id = request.form.get("preset_id", "").strip()
            if not preset_id:
                return jsonify({"error": "Preset ID necessario"}), 400

            # Save uploaded file
            tmp_dir = tempfile.mkdtemp(prefix="preset_apply_")
            fpath = os.path.join(tmp_dir, file.filename)
            file.save(fpath)

            # Build config
            config = PresetConfig()
            config.preset_id = preset_id
            config.intensity = float(request.form.get("intensity", 80))
            config.use_neural = request.form.get(
                "use_neural", "false").lower() == "true"
            config.normalize_output = request.form.get(
                "normalize_output", "true").lower() == "true"

            # Load audio
            sr = config.sample_rate
            audio = load_audio_file(fpath, sr)

            # Apply preset in background
            def run():
                try:
                    callback = progress_manager.make_callback(task_id)
                    processed, report = apply_preset(
                        audio, sr, config,
                        progress_callback=callback,
                    )

                    # Save both original and processed for A/B
                    orig_name = f"tone_orig_{task_id}.wav"
                    proc_name = f"tone_{task_id}.wav"
                    save_audio(os.path.join(OUTPUT_DIR, orig_name), audio, sr)
                    save_audio(os.path.join(OUTPUT_DIR, proc_name), processed, sr)

                    with _results_lock:
                        _task_results[task_id] = {
                            "filename": proc_name,
                            "original_filename": orig_name,
                            "report": report,
                        }

                    progress_manager.complete(
                        task_id,
                        f"Tone finalizado! Preset: {report.get('preset_name', '')}"
                    )

                except Exception as e:
                    traceback.print_exc()
                    with _results_lock:
                        _task_results[task_id] = {"error": str(e)}
                    progress_manager.error(task_id, str(e))

                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            return jsonify({"task_id": task_id})

        except Exception as e:
            progress_manager.error(task_id, str(e))
            traceback.print_exc()
            return jsonify({"error": str(e), "task_id": task_id}), 500

    # --- Structure API ---

    @app.route("/api/structure/detect", methods=["POST"])
    def structure_detect():
        """Detect musical structure from a single audio file."""
        from backend.structure.analyzer import analyze_structure

        task_id = str(uuid.uuid4())[:8]
        progress_manager.start(task_id, "Recebendo arquivo...")

        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "Arquivo de audio necessario"}), 400

            # Save uploaded file
            tmp_dir = tempfile.mkdtemp(prefix="structure_")
            fpath = os.path.join(tmp_dir, file.filename)
            file.save(fpath)

            # Load audio
            sr = 48000
            audio = load_audio_file(fpath, sr)

            # Run structure detection in background thread
            def run():
                try:
                    callback = progress_manager.make_callback(task_id)
                    result = analyze_structure(audio, sr, progress_cb=callback)

                    with _results_lock:
                        _task_results[task_id] = {
                            "filename": "",
                            "report": result,
                        }

                    n_sec = result.get("n_sections", 0)
                    progress_manager.complete(
                        task_id,
                        f"Estrutura detectada: {n_sec} secoes!"
                    )

                except Exception as e:
                    traceback.print_exc()
                    with _results_lock:
                        _task_results[task_id] = {"error": str(e)}
                    progress_manager.error(task_id, str(e))

                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            return jsonify({"task_id": task_id})

        except Exception as e:
            progress_manager.error(task_id, str(e))
            traceback.print_exc()
            return jsonify({"error": str(e), "task_id": task_id}), 500

    # --- Trim API (placeholder for Sprint 2) ---

    @app.route("/api/trim", methods=["POST"])
    def trim():
        return jsonify({"error": "Auto-trim sera implementado no Sprint 2"}), 501

    # Cleanup on shutdown
    import atexit
    atexit.register(lambda: shutil.rmtree(OUTPUT_DIR, ignore_errors=True))

    return app
