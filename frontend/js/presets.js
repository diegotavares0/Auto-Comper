/**
 * Produtora — Tone presets tab logic.
 * Create presets from reference clips, apply to guitar takes, A/B comparison.
 */

const Presets = {
    wavesurferProcessed: null,
    wavesurferOriginal: null,
    isPlaying: false,
    activePlayer: 'processed',
    currentSSE: null,
    applyFile: null,
    refFile: null,
    presets: [],
    selectedPresetId: '',
    neuralAvailable: false,

    init() {
        // DOM refs — New Preset
        this.btnNew = Utils.$('#tone-btn-new');
        this.newPresetCard = Utils.$('#tone-new-preset-card');
        this.presetNameInput = Utils.$('#tone-preset-name');
        this.refDropZone = Utils.$('#tone-ref-drop-zone');
        this.refFileInput = Utils.$('#tone-ref-file-input');
        this.refFileInfo = Utils.$('#tone-ref-file-info');
        this.btnCreate = Utils.$('#btn-create-preset');
        this.btnCancel = Utils.$('#btn-cancel-preset');
        this.createProgressCard = Utils.$('#tone-create-progress-card');
        this.createProgressBar = Utils.$('#tone-create-progress-bar');
        this.createProgressPct = Utils.$('#tone-create-progress-pct');
        this.createProgressMsg = Utils.$('#tone-create-progress-msg');

        // DOM refs — Preset Library
        this.presetList = Utils.$('#tone-preset-list');

        // DOM refs — Apply Preset
        this.applyDropZone = Utils.$('#tone-apply-drop-zone');
        this.applyFileInput = Utils.$('#tone-apply-file-input');
        this.applyFileInfo = Utils.$('#tone-apply-file-info');
        this.btnApply = Utils.$('#btn-apply-tone');
        this.applyProgressCard = Utils.$('#tone-apply-progress-card');
        this.applyProgressBar = Utils.$('#tone-apply-progress-bar');
        this.applyProgressPct = Utils.$('#tone-apply-progress-pct');
        this.applyProgressMsg = Utils.$('#tone-apply-progress-msg');

        // DOM refs — Result
        this.resultCard = Utils.$('#tone-result-card');
        this.btnPlay = Utils.$('#tone-btn-play');
        this.playerTime = Utils.$('#tone-player-time');
        this.btnDownload = Utils.$('#tone-btn-download');
        this.btnDownloadOrig = Utils.$('#tone-btn-download-orig');
        this.abToggle = Utils.$('#tone-ab-toggle');
        this.spectralCanvas = Utils.$('#tone-spectral-canvas');

        if (!this.btnNew) return;

        // Slider binding
        this.bindSlider('tone-cfg-intensity', 'tone-val-intensity', (v) => v + '%');

        // ── New Preset events ──
        this.btnNew.addEventListener('click', () => this.showNewPresetForm());
        this.btnCancel.addEventListener('click', () => this.hideNewPresetForm());
        this.btnCreate.addEventListener('click', () => this.createPreset());
        this.presetNameInput.addEventListener('input', () => this.updateCreateButton());

        // Reference drop zone
        this.refDropZone.addEventListener('click', () => this.refFileInput.click());
        this.refDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.refDropZone.classList.add('over');
        });
        this.refDropZone.addEventListener('dragleave', () => {
            this.refDropZone.classList.remove('over');
        });
        this.refDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.refDropZone.classList.remove('over');
            if (e.dataTransfer.files.length > 0) this.setRefFile(e.dataTransfer.files[0]);
        });
        this.refFileInput.addEventListener('change', () => {
            if (this.refFileInput.files.length > 0) this.setRefFile(this.refFileInput.files[0]);
        });

        // ── Apply Preset events ──
        this.btnApply.addEventListener('click', () => this.applyPreset());
        this.applyDropZone.addEventListener('click', () => this.applyFileInput.click());
        this.applyDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.applyDropZone.classList.add('over');
        });
        this.applyDropZone.addEventListener('dragleave', () => {
            this.applyDropZone.classList.remove('over');
        });
        this.applyDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.applyDropZone.classList.remove('over');
            if (e.dataTransfer.files.length > 0) this.setApplyFile(e.dataTransfer.files[0]);
        });
        this.applyFileInput.addEventListener('change', () => {
            if (this.applyFileInput.files.length > 0) this.setApplyFile(this.applyFileInput.files[0]);
        });

        // Play button
        this.btnPlay.addEventListener('click', () => this.togglePlay());

        // A/B toggle
        if (this.abToggle) {
            this.abToggle.querySelectorAll('.ab-toggle-btn').forEach(btn => {
                btn.addEventListener('click', () => this.switchAB(btn.dataset.mode));
            });
        }

        // Preset select change
        const presetSelect = Utils.$('#tone-cfg-preset');
        if (presetSelect) {
            presetSelect.addEventListener('change', () => {
                this.selectedPresetId = presetSelect.value;
                this.updateApplyButton();
            });
        }

        // Load presets from server
        this.loadPresets();
    },

    bindSlider(inputId, displayId, formatter) {
        const input = Utils.$('#' + inputId);
        const display = Utils.$('#' + displayId);
        if (!input || !display) return;
        const update = () => { display.textContent = formatter(input.value); };
        input.addEventListener('input', update);
        update();
    },

    // ═══════════════════════════════════════════
    //  PRESET LIBRARY
    // ═══════════════════════════════════════════

    async loadPresets() {
        try {
            const resp = await fetch('/api/presets');
            const data = await resp.json();
            this.presets = data.presets || [];
            this.neuralAvailable = data.neural_available || false;
            this.renderPresetList();
            this.updatePresetSelect();
            this.updateNeuralOption();
        } catch (err) {
            console.error('Failed to load presets:', err);
        }
    },

    renderPresetList() {
        if (!this.presetList) return;
        this.presetList.innerHTML = '';

        if (this.presets.length === 0) {
            const empty = Utils.el('div', 'preset-empty',
                'Nenhum preset criado ainda. Clique em "+ Novo" para comecar.');
            this.presetList.appendChild(empty);
            return;
        }

        for (const preset of this.presets) {
            const card = Utils.el('div', 'preset-card');
            card.dataset.id = preset.id;

            const name = Utils.el('div', 'preset-card-name', preset.name);

            const meta = Utils.el('div', 'preset-card-meta');
            const timbre = preset.timbre || {};
            const brightness = timbre.brightness != null
                ? (timbre.brightness * 100).toFixed(0) + '% brilho'
                : '';
            const warmth = timbre.warmth != null
                ? (timbre.warmth * 100).toFixed(0) + '% calor'
                : '';
            const dur = preset.duration_s
                ? Utils.formatTime(preset.duration_s) + ' ref'
                : '';
            meta.textContent = [brightness, warmth, dur].filter(Boolean).join(' · ');

            const deleteBtn = Utils.el('button', 'preset-card-delete', '×');
            deleteBtn.title = 'Deletar preset';
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deletePreset(preset.id, preset.name);
            });

            card.append(name, meta, deleteBtn);

            // Click to select
            card.addEventListener('click', () => {
                this.selectPreset(preset.id);
            });

            this.presetList.appendChild(card);
        }
    },

    updatePresetSelect() {
        const select = Utils.$('#tone-cfg-preset');
        if (!select) return;

        const currentVal = select.value;
        select.innerHTML = '<option value="">Selecione um preset...</option>';

        for (const preset of this.presets) {
            const opt = document.createElement('option');
            opt.value = preset.id;
            opt.textContent = preset.name;
            select.appendChild(opt);
        }

        // Restore selection if still exists
        if (currentVal && this.presets.some(p => p.id === currentVal)) {
            select.value = currentVal;
            this.selectedPresetId = currentVal;
        }
    },

    updateNeuralOption() {
        const modeSelect = Utils.$('#tone-cfg-mode');
        if (!modeSelect) return;

        const neuralOpt = modeSelect.querySelector('option[value="neural"]');
        if (neuralOpt) {
            if (this.neuralAvailable) {
                neuralOpt.textContent = 'DSP + Neural';
                neuralOpt.disabled = false;
            } else {
                neuralOpt.textContent = 'DSP + Neural (instalar PyTorch)';
                neuralOpt.disabled = true;
            }
        }
    },

    selectPreset(id) {
        const select = Utils.$('#tone-cfg-preset');
        if (select) {
            select.value = id;
            this.selectedPresetId = id;
        }

        // Highlight card
        if (this.presetList) {
            this.presetList.querySelectorAll('.preset-card').forEach(card => {
                card.classList.toggle('selected', card.dataset.id === id);
            });
        }

        this.updateApplyButton();
    },

    async deletePreset(id, name) {
        if (!confirm(`Deletar preset "${name}"?`)) return;

        try {
            await fetch(`/api/presets/${id}`, { method: 'DELETE' });
            await this.loadPresets();
        } catch (err) {
            console.error('Failed to delete preset:', err);
        }
    },

    // ═══════════════════════════════════════════
    //  CREATE PRESET
    // ═══════════════════════════════════════════

    showNewPresetForm() {
        Utils.show(this.newPresetCard);
        this.presetNameInput.focus();
    },

    hideNewPresetForm() {
        Utils.hide(this.newPresetCard);
        Utils.hide(this.createProgressCard);
        this.refFile = null;
        this.refFileInfo.innerHTML = '';
        this.presetNameInput.value = '';
        this.btnCreate.disabled = true;
        this.refFileInput.value = '';
    },

    setRefFile(file) {
        const ext = file.name.toLowerCase().split('.').pop();
        if (!['wav', 'flac', 'aiff', 'aif'].includes(ext)) {
            alert('Formato nao suportado. Use WAV, FLAC ou AIFF.');
            return;
        }

        this.refFile = file;
        this.refFileInfo.innerHTML = '';

        const row = Utils.el('div', 'file-item');
        const name = Utils.el('span', 'file-name', file.name);
        const size = Utils.el('span', 'file-size', Utils.formatSize(file.size));
        const removeBtn = Utils.el('button', 'file-remove', '\u00d7');
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.refFile = null;
            this.refFileInfo.innerHTML = '';
            this.updateCreateButton();
        });

        row.append(name, size, removeBtn);
        this.refFileInfo.appendChild(row);
        this.updateCreateButton();
    },

    updateCreateButton() {
        const name = this.presetNameInput ? this.presetNameInput.value.trim() : '';
        this.btnCreate.disabled = !this.refFile || !name;
    },

    async createPreset() {
        if (!this.refFile || !this.presetNameInput.value.trim()) return;

        this.btnCreate.disabled = true;
        Utils.show(this.createProgressCard);
        this.createProgressBar.style.width = '0%';
        this.createProgressPct.textContent = '0%';
        this.createProgressMsg.textContent = 'Enviando referencia...';

        const fd = new FormData();
        fd.append('file', this.refFile);
        fd.append('name', this.presetNameInput.value.trim());

        try {
            const data = await API.post('/api/presets', fd);

            if (data.error) {
                this.createProgressMsg.textContent = 'Erro: ' + data.error;
                this.createProgressMsg.style.color = 'var(--red)';
                return;
            }

            if (data.task_id) {
                this.currentSSE = API.subscribeProgress(
                    data.task_id,
                    (pct, msg) => {
                        this.createProgressBar.style.width = pct + '%';
                        this.createProgressPct.textContent = pct + '%';
                        this.createProgressMsg.textContent = msg;
                    },
                    async () => {
                        this.createProgressBar.style.width = '100%';
                        this.createProgressPct.textContent = '100%';
                        this.createProgressMsg.textContent = 'Preset criado!';
                        this.hideNewPresetForm();
                        await this.loadPresets();
                    },
                    (msg) => {
                        this.createProgressMsg.textContent = 'Erro: ' + msg;
                        this.createProgressMsg.style.color = 'var(--red)';
                        this.btnCreate.disabled = false;
                    }
                );
            }

        } catch (err) {
            this.createProgressMsg.textContent = 'Erro: ' + err.message;
            this.btnCreate.disabled = false;
        }
    },

    // ═══════════════════════════════════════════
    //  APPLY PRESET
    // ═══════════════════════════════════════════

    setApplyFile(file) {
        const ext = file.name.toLowerCase().split('.').pop();
        if (!['wav', 'flac', 'aiff', 'aif'].includes(ext)) {
            alert('Formato nao suportado. Use WAV, FLAC ou AIFF.');
            return;
        }

        this.applyFile = file;
        this.applyFileInfo.innerHTML = '';

        const row = Utils.el('div', 'file-item');
        const name = Utils.el('span', 'file-name', file.name);
        const size = Utils.el('span', 'file-size', Utils.formatSize(file.size));
        const removeBtn = Utils.el('button', 'file-remove', '\u00d7');
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.applyFile = null;
            this.applyFileInfo.innerHTML = '';
            this.applyFileInput.value = '';
            this.updateApplyButton();
        });

        row.append(name, size, removeBtn);
        this.applyFileInfo.appendChild(row);
        Utils.hide(this.resultCard);
        this.updateApplyButton();
    },

    updateApplyButton() {
        this.btnApply.disabled = !this.applyFile || !this.selectedPresetId;
    },

    async applyPreset() {
        if (!this.applyFile || !this.selectedPresetId) return;

        this.btnApply.disabled = true;
        Utils.show(this.applyProgressCard);
        Utils.hide(this.resultCard);
        this.applyProgressBar.style.width = '0%';
        this.applyProgressPct.textContent = '0%';
        this.applyProgressMsg.textContent = 'Enviando arquivo...';

        if (this.currentSSE) {
            this.currentSSE.close();
            this.currentSSE = null;
        }

        const fd = new FormData();
        fd.append('file', this.applyFile);
        fd.append('preset_id', this.selectedPresetId);
        fd.append('intensity', Utils.$('#tone-cfg-intensity').value);
        fd.append('use_neural', Utils.$('#tone-cfg-mode').value === 'neural' ? 'true' : 'false');
        fd.append('normalize_output', Utils.$('#tone-cfg-normalize').checked ? 'true' : 'false');

        try {
            const data = await API.post('/api/apply-preset', fd);

            if (data.error) {
                this.showApplyError(data.error);
                return;
            }

            if (data.task_id) {
                this.currentSSE = API.subscribeProgress(
                    data.task_id,
                    (pct, msg) => this.onApplyProgress(pct, msg),
                    (msg) => this.onApplyComplete(data.task_id, msg),
                    (msg) => this.showApplyError(msg)
                );
            }

        } catch (err) {
            this.showApplyError(err.message);
        }
    },

    onApplyProgress(pct, msg) {
        this.applyProgressBar.style.width = pct + '%';
        this.applyProgressPct.textContent = pct + '%';
        this.applyProgressMsg.textContent = msg;
    },

    async onApplyComplete(taskId, msg) {
        this.applyProgressBar.style.width = '100%';
        this.applyProgressPct.textContent = '100%';
        this.applyProgressMsg.textContent = msg || 'Buscando resultado...';

        try {
            const result = await API.fetchResult(taskId);
            this.showResult(result);
        } catch (err) {
            this.showApplyError(err.message);
        } finally {
            this.btnApply.disabled = !(this.applyFile && this.selectedPresetId);
        }
    },

    showApplyError(msg) {
        this.applyProgressMsg.textContent = 'Erro: ' + msg;
        this.applyProgressMsg.style.color = 'var(--red)';
        this.btnApply.disabled = !(this.applyFile && this.selectedPresetId);
        setTimeout(() => { this.applyProgressMsg.style.color = ''; }, 5000);
    },

    // ═══════════════════════════════════════════
    //  RESULT DISPLAY
    // ═══════════════════════════════════════════

    showResult(data) {
        // Guard: backend must return a valid filename
        if (!data || !data.filename) {
            this.showApplyError('Resultado sem arquivo de saida. Tente novamente.');
            return;
        }

        Utils.show(this.resultCard);

        const processedUrl = API.outputUrl(data.filename);
        const originalUrl = data.original_filename
            ? API.outputUrl(data.original_filename)
            : null;

        // Download links
        this.btnDownload.href = processedUrl;
        this.btnDownload.download = data.filename;
        if (originalUrl && this.btnDownloadOrig) {
            this.btnDownloadOrig.href = originalUrl;
            this.btnDownloadOrig.download = data.original_filename;
            Utils.show(this.btnDownloadOrig);
        }

        // Init waveforms (with try/catch to prevent WaveSurfer crash)
        try {
            this.initWaveform('processed', processedUrl);
            if (originalUrl) this.initWaveform('original', originalUrl);
        } catch (err) {
            console.error('WaveSurfer init failed:', err);
        }

        this.switchAB('processed');

        if (data.report) this.renderReport(data.report);
    },

    initWaveform(type, url) {
        const containerId = type === 'processed'
            ? '#tone-waveform-processed'
            : '#tone-waveform-original';
        const prop = type === 'processed' ? 'wavesurferProcessed' : 'wavesurferOriginal';
        const color = type === 'processed' ? '#f0a050' : '#6c8cff';
        const progressColor = type === 'processed' ? '#f0c080' : '#8cacff';

        if (this[prop]) { this[prop].destroy(); this[prop] = null; }

        this[prop] = WaveSurfer.create({
            container: containerId,
            waveColor: color,
            progressColor: progressColor,
            cursorColor: color + '88',
            barWidth: 2, barGap: 1, barRadius: 1,
            height: 48, normalize: true, backend: 'WebAudio',
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
                this.playerTime.textContent = `${Utils.formatTime(cur)} / ${Utils.formatTime(dur)}`;
            }
        });

        this[prop].on('finish', () => {
            this.isPlaying = false;
            this.updatePlayIcon();
        });
    },

    switchAB(mode) {
        this.activePlayer = mode;
        const wasProcessed = Utils.$('#tone-waveform-processed');
        const wasOriginal = Utils.$('#tone-waveform-original');

        if (this.wavesurferProcessed) this.wavesurferProcessed.pause();
        if (this.wavesurferOriginal) this.wavesurferOriginal.pause();
        this.isPlaying = false;
        this.updatePlayIcon();

        // Sync position
        const activeWs = mode === 'processed' ? this.wavesurferProcessed : this.wavesurferOriginal;
        const inactiveWs = mode === 'processed' ? this.wavesurferOriginal : this.wavesurferProcessed;

        if (inactiveWs && activeWs) {
            try {
                const pos = inactiveWs.getCurrentTime();
                const dur = activeWs.getDuration();
                if (dur > 0) activeWs.seekTo(Math.min(pos / dur, 1));
            } catch (_) {}
        }

        if (wasProcessed) wasProcessed.style.display = mode === 'processed' ? '' : 'none';
        if (wasOriginal) wasOriginal.style.display = mode === 'original' ? '' : 'none';

        if (this.abToggle) {
            this.abToggle.querySelectorAll('.ab-toggle-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.mode === mode);
            });
        }

        if (activeWs) {
            const cur = activeWs.getCurrentTime();
            const dur = activeWs.getDuration();
            this.playerTime.textContent = `${Utils.formatTime(cur)} / ${Utils.formatTime(dur)}`;
        }
    },

    togglePlay() {
        const ws = this.activePlayer === 'processed'
            ? this.wavesurferProcessed
            : this.wavesurferOriginal;
        if (!ws) return;
        ws.playPause();
        this.isPlaying = !this.isPlaying;
        this.updatePlayIcon();
    },

    updatePlayIcon() {
        const iconPlay = Utils.$('#tone-icon-play');
        const iconPause = Utils.$('#tone-icon-pause');
        if (!iconPlay || !iconPause) return;
        iconPlay.style.display = this.isPlaying ? 'none' : '';
        iconPause.style.display = this.isPlaying ? '' : 'none';
    },

    renderReport(report) {
        const statsEl = Utils.$('#tone-result-stats');
        if (statsEl) {
            statsEl.innerHTML = '';
            const dsp = report.dsp || {};
            const gain = dsp.gain_curve_db || {};
            const stats = [
                { label: 'Preset', value: report.preset_name || '—' },
                { label: 'Duracao', value: Utils.formatTime(report.duration_s) },
                { label: 'Intensidade', value: report.intensity + '%' },
                { label: 'EQ Min', value: (gain.min || 0) + ' dB' },
                { label: 'EQ Max', value: '+' + (gain.max || 0) + ' dB' },
                { label: 'Neural', value: report.neural_used ? 'Sim' : 'Nao' },
            ];

            for (const s of stats) {
                const box = Utils.el('div', 'stat-box');
                const val = Utils.el('div', 'stat-value', String(s.value));
                const lbl = Utils.el('div', 'stat-label', s.label);
                box.append(val, lbl);
                statsEl.appendChild(box);
            }
        }

        // Spectral comparison visualization
        if (this.spectralCanvas && report.spectral_comparison) {
            this.renderSpectralComparison(
                this.spectralCanvas,
                report.spectral_comparison.bands || [],
            );
        }
    },

    renderSpectralComparison(canvas, bands) {
        if (!bands.length) return;

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

        // Draw EQ difference as bars
        const padX = 40;
        const padY = 20;
        const plotW = w - padX * 2;
        const plotH = h - padY * 2;
        const midY = padY + plotH / 2;

        // Zero line
        ctx.strokeStyle = '#ffffff20';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padX, midY);
        ctx.lineTo(padX + plotW, midY);
        ctx.stroke();

        // +/- reference lines
        ctx.strokeStyle = '#ffffff10';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillStyle = '#ffffff30';
        for (const db of [-12, -6, 6, 12]) {
            const y = midY - (db / 18) * (plotH / 2);
            ctx.beginPath();
            ctx.moveTo(padX, y);
            ctx.lineTo(padX + plotW, y);
            ctx.stroke();
            ctx.fillText(db > 0 ? '+' + db : String(db), 4, y + 3);
        }
        ctx.fillText('0 dB', 4, midY + 3);

        // Bars
        const barW = Math.max(plotW / bands.length - 2, 3);

        for (let i = 0; i < bands.length; i++) {
            const band = bands[i];
            const x = padX + (i / bands.length) * plotW;
            const diff = Math.max(-18, Math.min(18, band.diff_db));
            const barH = (diff / 18) * (plotH / 2);

            // Color: warm/orange for boost, blue for cut
            if (diff >= 0) {
                ctx.fillStyle = '#f0a05080';
            } else {
                ctx.fillStyle = '#6c8cff60';
            }

            ctx.fillRect(x, midY - Math.max(barH, 0), barW, Math.abs(barH));

            // Frequency labels (every 4th band)
            if (i % 4 === 0) {
                ctx.fillStyle = '#ffffff30';
                ctx.font = '8px "JetBrains Mono", monospace';
                const freq = band.freq_hz;
                const label = freq >= 1000 ? (freq / 1000).toFixed(1) + 'k' : Math.round(freq);
                ctx.fillText(label, x, h - 4);
            }
        }
    },
};
