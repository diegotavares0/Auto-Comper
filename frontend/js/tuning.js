/**
 * Produtora — Tuning tab logic.
 * Upload single file, configure pitch correction, A/B comparison.
 */

const Tuning = {
    wavesurferTuned: null,
    wavesurferOriginal: null,
    isPlaying: false,
    activePlayer: 'tuned',  // 'tuned' or 'original'
    currentSSE: null,
    file: null,

    init() {
        // DOM refs
        this.dropZone = Utils.$('#tune-drop-zone');
        this.fileInput = Utils.$('#tune-file-input');
        this.fileInfo = Utils.$('#tune-file-info');
        this.btnTune = Utils.$('#btn-tune');
        this.progressCard = Utils.$('#tune-progress-card');
        this.progressBar = Utils.$('#tune-progress-bar');
        this.progressPct = Utils.$('#tune-progress-pct');
        this.progressMsg = Utils.$('#tune-progress-msg');
        this.resultCard = Utils.$('#tune-result-card');
        this.btnPlay = Utils.$('#tune-btn-play');
        this.playerTime = Utils.$('#tune-player-time');
        this.btnDownload = Utils.$('#tune-btn-download');
        this.btnDownloadOrig = Utils.$('#tune-btn-download-orig');
        this.abToggle = Utils.$('#tune-ab-toggle');
        this.pitchCanvas = Utils.$('#tune-pitch-canvas');
        this.warningBanner = Utils.$('#tune-warning');

        if (!this.dropZone) return;  // Tab not yet in DOM

        // Slider bindings
        this.bindSlider('tune-cfg-correction', 'tune-val-correction', (v) => v + '%');
        this.bindSlider('tune-cfg-speed', 'tune-val-speed', (v) => v);

        // Drop zone events
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('over');
        });
        this.dropZone.addEventListener('dragleave', () => {
            this.dropZone.classList.remove('over');
        });
        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('over');
            if (e.dataTransfer.files.length > 0) {
                this.setFile(e.dataTransfer.files[0]);
            }
        });
        this.fileInput.addEventListener('change', () => {
            if (this.fileInput.files.length > 0) {
                this.setFile(this.fileInput.files[0]);
            }
        });

        // Tune button
        this.btnTune.addEventListener('click', () => this.runTune());

        // Play button
        this.btnPlay.addEventListener('click', () => this.togglePlay());

        // A/B toggle buttons
        if (this.abToggle) {
            this.abToggle.querySelectorAll('.ab-toggle-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    this.switchAB(btn.dataset.mode);
                });
            });
        }
    },

    bindSlider(inputId, displayId, formatter) {
        const input = Utils.$('#' + inputId);
        const display = Utils.$('#' + displayId);
        if (!input || !display) return;
        const update = () => { display.textContent = formatter(input.value); };
        input.addEventListener('input', update);
        update();
    },

    /**
     * Set the audio file for tuning.
     */
    setFile(file) {
        const ext = file.name.toLowerCase().split('.').pop();
        if (!['wav', 'flac', 'aiff', 'aif'].includes(ext)) {
            alert('Formato nao suportado. Use WAV, FLAC ou AIFF.');
            return;
        }

        this.file = file;
        this.fileInfo.innerHTML = '';

        const row = Utils.el('div', 'file-item');
        const name = Utils.el('span', 'file-name', file.name);
        const size = Utils.el('span', 'file-size', Utils.formatSize(file.size));
        const removeBtn = Utils.el('button', 'file-remove', '\u00d7');
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.clearFile();
        });

        row.append(name, size, removeBtn);
        this.fileInfo.appendChild(row);

        this.btnTune.disabled = false;
        Utils.hide(this.resultCard);
        Utils.hide(this.progressCard);
    },

    clearFile() {
        this.file = null;
        this.fileInfo.innerHTML = '';
        this.btnTune.disabled = true;
        this.fileInput.value = '';
    },

    /**
     * Gather config from form.
     */
    getConfig() {
        return {
            instrument_mode: Utils.$('#tune-cfg-instrument').value,
            correction_amount: Utils.$('#tune-cfg-correction').value,
            retune_speed: Utils.$('#tune-cfg-speed').value,
            root_note: Utils.$('#tune-cfg-root').value,
            scale_type: Utils.$('#tune-cfg-scale').value,
            preserve_vibrato: Utils.$('#tune-cfg-vibrato').checked ? 'true' : 'false',
            normalize_output: Utils.$('#tune-cfg-normalize').checked ? 'true' : 'false',
        };
    },

    /**
     * Run the tuning pipeline.
     */
    async runTune() {
        if (!this.file) return;

        // Reset UI
        this.btnTune.disabled = true;
        Utils.show(this.progressCard);
        Utils.hide(this.resultCard);
        Utils.hide(this.warningBanner);
        this.progressBar.style.width = '0%';
        this.progressPct.textContent = '0%';
        this.progressMsg.textContent = 'Enviando arquivo...';

        // Close previous SSE
        if (this.currentSSE) {
            this.currentSSE.close();
            this.currentSSE = null;
        }

        // Build FormData
        const fd = new FormData();
        fd.append('file', this.file);

        const config = this.getConfig();
        for (const [key, val] of Object.entries(config)) {
            fd.append(key, val);
        }

        try {
            const data = await API.post('/api/tune', fd);

            if (data.error) {
                this.showError(data.error);
                return;
            }

            if (data.task_id) {
                this.currentSSE = API.subscribeProgress(
                    data.task_id,
                    (pct, msg) => this.onProgress(pct, msg),
                    (msg) => this.onComplete(data.task_id, msg),
                    (msg) => this.showError(msg)
                );
            }

        } catch (err) {
            this.showError(err.message);
            this.btnTune.disabled = !this.file;
        }
    },

    onProgress(pct, msg) {
        this.progressBar.style.width = pct + '%';
        this.progressPct.textContent = pct + '%';
        this.progressMsg.textContent = msg;
    },

    async onComplete(taskId, msg) {
        this.progressBar.style.width = '100%';
        this.progressPct.textContent = '100%';
        this.progressMsg.textContent = msg || 'Buscando resultado...';

        try {
            const result = await API.fetchResult(taskId);
            this.showResult(result);
        } catch (err) {
            this.showError(err.message);
        } finally {
            this.btnTune.disabled = !this.file;
        }
    },

    showError(msg) {
        this.progressMsg.textContent = 'Erro: ' + msg;
        this.progressMsg.style.color = 'var(--red)';
        this.btnTune.disabled = !this.file;
        setTimeout(() => {
            this.progressMsg.style.color = '';
        }, 5000);
    },

    /**
     * Show tuning result with A/B comparison.
     */
    showResult(data) {
        Utils.show(this.resultCard);
        this.progressBar.style.width = '100%';
        this.progressPct.textContent = '100%';
        this.progressMsg.textContent = 'Tuning finalizado!';

        const tunedUrl = API.outputUrl(data.filename);
        const originalUrl = data.original_filename
            ? API.outputUrl(data.original_filename)
            : null;

        // Download links
        this.btnDownload.href = tunedUrl;
        this.btnDownload.download = data.filename;
        if (originalUrl && this.btnDownloadOrig) {
            this.btnDownloadOrig.href = originalUrl;
            this.btnDownloadOrig.download = data.original_filename;
            Utils.show(this.btnDownloadOrig);
        }

        // Init waveform players
        this.initWaveform('tuned', tunedUrl);
        if (originalUrl) {
            this.initWaveform('original', originalUrl);
        }

        // Set A/B toggle to tuned
        this.switchAB('tuned');

        // Render report
        if (data.report) {
            this.renderReport(data.report);
        }
    },

    /**
     * Initialize WaveSurfer for playback.
     */
    initWaveform(type, url) {
        const containerId = type === 'tuned'
            ? '#tune-waveform-tuned'
            : '#tune-waveform-original';
        const prop = type === 'tuned' ? 'wavesurferTuned' : 'wavesurferOriginal';
        const color = type === 'tuned' ? '#5cbf78' : '#6c8cff';
        const progressColor = type === 'tuned' ? '#78d898' : '#8cacff';

        if (this[prop]) {
            this[prop].destroy();
            this[prop] = null;
        }

        this[prop] = WaveSurfer.create({
            container: containerId,
            waveColor: color,
            progressColor: progressColor,
            cursorColor: color + '88',
            barWidth: 2,
            barGap: 1,
            barRadius: 1,
            height: 48,
            normalize: true,
            backend: 'WebAudio',
        });

        this[prop].load(url);

        this[prop].on('ready', () => {
            if (type === this.activePlayer) {
                const dur = this[prop].getDuration();
                this.playerTime.textContent = `0:00 / ${Utils.formatTime(dur)}`;
            }
        });

        this[prop].on('audioprocess', () => {
            if (type === this.activePlayer) {
                const cur = this[prop].getCurrentTime();
                const dur = this[prop].getDuration();
                this.playerTime.textContent =
                    `${Utils.formatTime(cur)} / ${Utils.formatTime(dur)}`;
            }
        });

        this[prop].on('finish', () => {
            this.isPlaying = false;
            this.updatePlayIcon();
        });

        this[prop].on('interaction', () => {
            if (type === this.activePlayer) {
                const cur = this[prop].getCurrentTime();
                const dur = this[prop].getDuration();
                this.playerTime.textContent =
                    `${Utils.formatTime(cur)} / ${Utils.formatTime(dur)}`;
            }
        });
    },

    /**
     * Switch A/B comparison.
     */
    switchAB(mode) {
        this.activePlayer = mode;
        const wasTuned = Utils.$('#tune-waveform-tuned');
        const wasOriginal = Utils.$('#tune-waveform-original');

        // Pause both
        if (this.wavesurferTuned) this.wavesurferTuned.pause();
        if (this.wavesurferOriginal) this.wavesurferOriginal.pause();
        this.isPlaying = false;
        this.updatePlayIcon();

        // Sync position
        const activeWs = mode === 'tuned' ? this.wavesurferTuned : this.wavesurferOriginal;
        const inactiveWs = mode === 'tuned' ? this.wavesurferOriginal : this.wavesurferTuned;

        if (inactiveWs && activeWs) {
            try {
                const pos = inactiveWs.getCurrentTime();
                const dur = activeWs.getDuration();
                if (dur > 0) {
                    activeWs.seekTo(Math.min(pos / dur, 1));
                }
            } catch (_) {}
        }

        // Show/hide waveforms
        if (wasTuned) wasTuned.style.display = mode === 'tuned' ? '' : 'none';
        if (wasOriginal) wasOriginal.style.display = mode === 'original' ? '' : 'none';

        // Update toggle buttons
        if (this.abToggle) {
            this.abToggle.querySelectorAll('.ab-toggle-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
        }

        // Update time display
        if (activeWs) {
            const cur = activeWs.getCurrentTime();
            const dur = activeWs.getDuration();
            this.playerTime.textContent =
                `${Utils.formatTime(cur)} / ${Utils.formatTime(dur)}`;
        }
    },

    togglePlay() {
        const ws = this.activePlayer === 'tuned'
            ? this.wavesurferTuned
            : this.wavesurferOriginal;

        if (!ws) return;
        ws.playPause();
        this.isPlaying = !this.isPlaying;
        this.updatePlayIcon();
    },

    updatePlayIcon() {
        const iconPlay = Utils.$('#tune-icon-play');
        const iconPause = Utils.$('#tune-icon-pause');
        if (!iconPlay || !iconPause) return;

        if (this.isPlaying) {
            iconPlay.style.display = 'none';
            iconPause.style.display = '';
        } else {
            iconPlay.style.display = '';
            iconPause.style.display = 'none';
        }
    },

    /**
     * Render the tuning report.
     */
    renderReport(report) {
        // Stats grid
        const statsEl = Utils.$('#tune-result-stats');
        if (statsEl) {
            statsEl.innerHTML = '';

            const stats = [
                { label: 'Duracao', value: Utils.formatTime(report.duration_s) },
                { label: 'Tom', value: report.effective_key || report.detected_key },
                { label: 'Confianca', value: Math.round(report.key_confidence * 100) + '%' },
                { label: 'Correcoes', value: report.corrections_applied },
                { label: 'Media', value: report.avg_correction_cents + ' cents' },
                { label: 'Maximo', value: report.max_correction_cents + ' cents' },
            ];

            for (const s of stats) {
                const box = Utils.el('div', 'stat-box');
                const val = Utils.el('div', 'stat-value', String(s.value));
                const lbl = Utils.el('div', 'stat-label', s.label);
                box.append(val, lbl);
                statsEl.appendChild(box);
            }
        }

        // Pitch analysis stats
        const pitchStatsEl = Utils.$('#tune-pitch-stats');
        if (pitchStatsEl && report.pitch_stats) {
            pitchStatsEl.innerHTML = '';
            const ps = report.pitch_stats;
            const items = [
                { label: 'Mediana Hz', value: ps.median_hz + ' Hz' },
                { label: 'Desvio', value: ps.std_cents + ' cents' },
                { label: 'Com voz', value: ps.voiced_pct + '%' },
                { label: 'Faixa', value: ps.range_semitones + ' st' },
            ];
            for (const item of items) {
                const box = Utils.el('div', 'stat-box');
                const val = Utils.el('div', 'stat-value', String(item.value));
                const lbl = Utils.el('div', 'stat-label', item.label);
                box.append(val, lbl);
                pitchStatsEl.appendChild(box);
            }
        }

        // Mixed signal warning
        if (report.mixed_signal_warning && this.warningBanner) {
            this.warningBanner.textContent =
                'Sinal misto detectado (voz + instrumento). ' +
                'A correcao foi aplicada de forma conservadora. ' +
                'Para melhores resultados, use takes separados.';
            Utils.show(this.warningBanner);
        }

        // Pitch visualization
        if (this.pitchCanvas) {
            this.renderPitchCurve(
                this.pitchCanvas,
                report.pitch_curve_original || [],
                report.pitch_curve_corrected || [],
                report.effective_key,
            );
        }
    },

    /**
     * Render pitch curve on canvas.
     */
    renderPitchCurve(canvas, original, corrected, keyLabel) {
        if (!original.length && !corrected.length) return;

        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        const w = rect.width;
        const h = 160;

        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        ctx.scale(dpr, dpr);

        // Background
        ctx.fillStyle = '#12121a';
        ctx.fillRect(0, 0, w, h);

        // Collect all Hz values to compute range
        const allHz = [
            ...original.map(p => p.hz),
            ...corrected.map(p => p.hz),
        ].filter(v => v > 0);

        if (!allHz.length) return;

        const minHz = Math.min(...allHz) * 0.9;
        const maxHz = Math.max(...allHz) * 1.1;
        const maxT = Math.max(
            original.length ? original[original.length - 1].t : 0,
            corrected.length ? corrected[corrected.length - 1].t : 0,
        );

        if (maxT <= 0) return;

        // Map functions (log scale for Hz)
        const logMin = Math.log2(minHz);
        const logMax = Math.log2(maxHz);
        const padY = 10;
        const plotH = h - padY * 2;

        const xMap = (t) => (t / maxT) * w;
        const yMap = (hz) => {
            const logHz = Math.log2(hz);
            return padY + plotH * (1 - (logHz - logMin) / (logMax - logMin));
        };

        // Draw reference lines for semitones in range
        ctx.strokeStyle = '#ffffff10';
        ctx.lineWidth = 0.5;
        const midiMin = Math.floor(12 * Math.log2(minHz / 440) + 69);
        const midiMax = Math.ceil(12 * Math.log2(maxHz / 440) + 69);

        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillStyle = '#ffffff30';

        for (let m = midiMin; m <= midiMax; m++) {
            const hz = 440 * Math.pow(2, (m - 69) / 12);
            if (hz < minHz || hz > maxHz) continue;
            const y = yMap(hz);

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();

            // Label every 3rd note (to avoid clutter)
            if (m % 3 === 0) {
                const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F',
                                   'F#', 'G', 'G#', 'A', 'A#', 'B'];
                const name = noteNames[m % 12] + Math.floor(m / 12 - 1);
                ctx.fillText(name, 2, y - 2);
            }
        }

        // Draw original pitch (dim blue dots)
        ctx.fillStyle = '#6c8cff50';
        for (const p of original) {
            const x = xMap(p.t);
            const y = yMap(p.hz);
            ctx.beginPath();
            ctx.arc(x, y, 1.5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Draw corrected pitch (bright green line)
        if (corrected.length > 1) {
            ctx.strokeStyle = '#5cbf78';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(xMap(corrected[0].t), yMap(corrected[0].hz));
            for (let i = 1; i < corrected.length; i++) {
                // Only draw line if points are close in time (< 0.3s gap)
                if (corrected[i].t - corrected[i - 1].t < 0.3) {
                    ctx.lineTo(xMap(corrected[i].t), yMap(corrected[i].hz));
                } else {
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(xMap(corrected[i].t), yMap(corrected[i].hz));
                }
            }
            ctx.stroke();
        }

        // Key label
        if (keyLabel) {
            ctx.fillStyle = '#ffffff60';
            ctx.font = '11px "DM Sans", sans-serif';
            ctx.fillText(keyLabel, w - ctx.measureText(keyLabel).width - 6, 16);
        }
    },
};
