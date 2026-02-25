/**
 * Produtora — Comping tab logic.
 */

const Comping = {
    wavesurfer: null,
    isPlaying: false,
    currentSSE: null,

    init() {
        this.btnComp = Utils.$('#btn-comp');
        this.progressCard = Utils.$('#progress-card');
        this.progressBar = Utils.$('#progress-bar');
        this.progressPct = Utils.$('#progress-pct');
        this.progressMsg = Utils.$('#progress-msg');
        this.resultCard = Utils.$('#result-card');
        this.btnPlay = Utils.$('#btn-play');
        this.playerTime = Utils.$('#player-time');
        this.btnDownload = Utils.$('#btn-download');

        // Config slider bindings
        this.bindSlider('cfg-crossfade', 'val-crossfade', (v) => v + 'ms');
        this.bindSlider('cfg-switch-penalty', 'val-switch-penalty', (v) => v);
        this.bindSlider('cfg-min-improvement', 'val-min-improvement', (v) => v);
        this.bindSlider('cfg-max-takes', 'val-max-takes', (v) => v);

        // Normalization sliders
        const normFmt = (v) => v == 0 ? 'Off' : v + '%';
        this.bindSlider('cfg-tempo-norm', 'val-tempo-norm', normFmt);
        this.bindSlider('cfg-pitch-center', 'val-pitch-center', normFmt);

        // Comp button
        this.btnComp.addEventListener('click', () => this.runComp());

        // Play button
        this.btnPlay.addEventListener('click', () => this.togglePlay());
    },

    /**
     * Bind a range input to display its value.
     */
    bindSlider(inputId, displayId, formatter) {
        const input = Utils.$('#' + inputId);
        const display = Utils.$('#' + displayId);
        if (!input || !display) return;

        const update = () => {
            display.textContent = formatter(input.value);
        };
        input.addEventListener('input', update);
        update();
    },

    /**
     * Update comp button state based on file count.
     */
    updateButton(files) {
        this.btnComp.disabled = files.length < 2;
    },

    /**
     * Gather config values from the form.
     */
    getConfig() {
        return {
            segment_method: Utils.$('#cfg-segment-method').value,
            alignment: Utils.$('#cfg-alignment').value,
            crossfade_ms: Utils.$('#cfg-crossfade').value,
            switch_penalty: Utils.$('#cfg-switch-penalty').value,
            min_improvement: Utils.$('#cfg-min-improvement').value,
            max_takes: Utils.$('#cfg-max-takes').value,
            auto_trim: Utils.$('#cfg-auto-trim').checked ? 'true' : 'false',
            tempo_normalize_intensity: Utils.$('#cfg-tempo-norm').value,
            pitch_center_intensity: Utils.$('#cfg-pitch-center').value,
        };
    },

    /**
     * Run the comping pipeline.
     */
    async runComp() {
        if (Upload.files.length < 2) return;

        // Reset UI
        this.btnComp.disabled = true;
        Utils.show(this.progressCard);
        Utils.hide(this.resultCard);
        this.progressBar.style.width = '0%';
        this.progressPct.textContent = '0%';
        this.progressMsg.textContent = 'Enviando arquivos...';

        // Close any previous SSE connection
        if (this.currentSSE) {
            this.currentSSE.close();
            this.currentSSE = null;
        }

        // Build FormData
        const fd = new FormData();
        for (const f of Upload.files) {
            fd.append('files', f);
        }

        const config = this.getConfig();
        for (const [key, val] of Object.entries(config)) {
            fd.append(key, val);
        }

        // Include confirmed structure sections if available
        const structSections = Structure.getConfirmedSections();
        if (structSections) {
            fd.append('structure_sections', JSON.stringify(structSections));
        }

        try {
            // POST to /api/comp — returns immediately with task_id
            const data = await API.post('/api/comp', fd);

            if (data.error) {
                this.showError(data.error);
                return;
            }

            // Subscribe to SSE progress stream
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
            this.btnComp.disabled = Upload.files.length < 2;
        }
    },

    /**
     * Handle progress update.
     */
    onProgress(pct, msg) {
        this.progressBar.style.width = pct + '%';
        this.progressPct.textContent = pct + '%';
        this.progressMsg.textContent = msg;
    },

    /**
     * Handle completion — fetch the result from the server.
     */
    async onComplete(taskId, msg) {
        this.progressBar.style.width = '100%';
        this.progressPct.textContent = '100%';
        this.progressMsg.textContent = msg || 'Buscando resultado...';

        try {
            // Fetch the result (audio file + report)
            const result = await API.fetchResult(taskId);
            this.showResult(result);
        } catch (err) {
            this.showError(err.message);
        } finally {
            this.btnComp.disabled = Upload.files.length < 2;
        }
    },

    /**
     * Show error message.
     */
    showError(msg) {
        this.progressMsg.textContent = 'Erro: ' + msg;
        this.progressMsg.style.color = 'var(--red)';
        this.btnComp.disabled = Upload.files.length < 2;
        setTimeout(() => {
            this.progressMsg.style.color = '';
        }, 5000);
    },

    /**
     * Display the comp result.
     */
    showResult(data) {
        Utils.show(this.resultCard);
        this.progressBar.style.width = '100%';
        this.progressPct.textContent = '100%';
        this.progressMsg.textContent = 'Comp finalizado!';

        const audioUrl = API.outputUrl(data.filename);

        // Download link
        this.btnDownload.href = audioUrl;
        this.btnDownload.download = data.filename;

        // Init waveform player
        this.initWaveform(audioUrl);

        // Render report
        if (data.report) {
            this.renderReport(data.report);
        }
    },

    /**
     * Initialize WaveSurfer for result playback.
     */
    initWaveform(url) {
        // Destroy previous instance
        if (this.wavesurfer) {
            this.wavesurfer.destroy();
            this.wavesurfer = null;
        }

        this.wavesurfer = WaveSurfer.create({
            container: '#player-waveform',
            waveColor: '#4a6ad4',
            progressColor: '#6c8cff',
            cursorColor: '#8cacff',
            barWidth: 2,
            barGap: 1,
            barRadius: 1,
            height: 48,
            normalize: true,
            backend: 'WebAudio',
        });

        this.wavesurfer.load(url);

        this.wavesurfer.on('ready', () => {
            const dur = this.wavesurfer.getDuration();
            this.playerTime.textContent = `0:00 / ${Utils.formatTime(dur)}`;
        });

        this.wavesurfer.on('audioprocess', () => {
            const cur = this.wavesurfer.getCurrentTime();
            const dur = this.wavesurfer.getDuration();
            this.playerTime.textContent = `${Utils.formatTime(cur)} / ${Utils.formatTime(dur)}`;
        });

        this.wavesurfer.on('finish', () => {
            this.isPlaying = false;
            this.updatePlayIcon();
        });

        this.wavesurfer.on('interaction', () => {
            const cur = this.wavesurfer.getCurrentTime();
            const dur = this.wavesurfer.getDuration();
            this.playerTime.textContent = `${Utils.formatTime(cur)} / ${Utils.formatTime(dur)}`;
        });
    },

    /**
     * Toggle play/pause.
     */
    togglePlay() {
        if (!this.wavesurfer) return;
        this.wavesurfer.playPause();
        this.isPlaying = !this.isPlaying;
        this.updatePlayIcon();
    },

    /**
     * Update play/pause icon visibility.
     */
    updatePlayIcon() {
        const iconPlay = Utils.$('#icon-play');
        const iconPause = Utils.$('#icon-pause');
        if (this.isPlaying) {
            iconPlay.style.display = 'none';
            iconPause.style.display = '';
        } else {
            iconPlay.style.display = '';
            iconPause.style.display = 'none';
        }
    },

    /**
     * Render the comp report (stats, usage bar, decisions).
     */
    renderReport(report) {
        // Stats
        const statsEl = Utils.$('#result-stats');
        statsEl.innerHTML = '';

        const stats = [
            { label: 'Duracao', value: Utils.formatTime(report.duration_s) },
            { label: 'Blocos', value: report.total_blocks },
            { label: 'Take base', value: 'Take ' + report.base_take },
            { label: 'Takes usados', value: report.takes_in_comp },
            { label: 'Trocas', value: report.take_switches },
            { label: 'Score medio', value: report.avg_score },
        ];

        for (const s of stats) {
            const box = Utils.el('div', 'stat-box');
            const val = Utils.el('div', 'stat-value', String(s.value));
            const lbl = Utils.el('div', 'stat-label', s.label);
            box.append(val, lbl);
            statsEl.appendChild(box);
        }

        // Normalization stats
        const normEl = Utils.$('#result-norm');
        normEl.innerHTML = '';
        if (report.normalization && report.normalization.enabled) {
            this.renderNormStats(normEl, report.normalization);
            normEl.style.display = '';
        } else {
            normEl.style.display = 'none';
        }

        // Usage bar
        const usageEl = Utils.$('#result-usage');
        usageEl.innerHTML = '';

        if (report.take_usage_pct && Object.keys(report.take_usage_pct).length > 0) {
            const label = Utils.el('div', 'usage-bar-label', 'Distribuicao de takes');
            const bar = Utils.el('div', 'usage-bar');
            const legend = Utils.el('div', 'usage-legend');

            // Sort by take number
            const entries = Object.entries(report.take_usage_pct)
                .sort(([a], [b]) => Number(a) - Number(b));

            for (const [take, pct] of entries) {
                // Bar segment
                const seg = Utils.el('div', 'usage-segment');
                seg.style.flex = pct;
                seg.style.background = Utils.takeColor(Number(take) - 1);
                seg.textContent = pct >= 5 ? `T${take}` : '';
                bar.appendChild(seg);

                // Legend
                const item = Utils.el('div', 'usage-legend-item');
                const dot = Utils.el('span', 'usage-legend-dot');
                dot.style.background = Utils.takeColor(Number(take) - 1);
                const text = Utils.el('span', '', `Take ${take}: ${pct}%`);
                item.append(dot, text);
                legend.appendChild(item);
            }

            usageEl.append(label, bar, legend);
        }

        // Decisions
        const decisionsEl = Utils.$('#result-decisions');
        decisionsEl.innerHTML = '';

        if (report.decisions && report.decisions.length > 0) {
            // Header
            const header = Utils.el('div', 'decision-row header');
            header.innerHTML = '<span>Bloco</span><span>Take</span><span>Score</span><span>Tempo</span><span></span>';
            decisionsEl.appendChild(header);

            for (const d of report.decisions) {
                const row = Utils.el('div', 'decision-row');

                const blk = Utils.el('span', '', `#${d.block + 1}`);

                const takeBadge = Utils.el('span', 'take-badge', `Take ${d.take}`);
                takeBadge.style.background = Utils.takeColor(d.take - 1);

                const score = Utils.el('span', 'score', d.score.toFixed(3));

                const time = Utils.el('span', '', Utils.formatTime(d.start_s) + ' - ' + Utils.formatTime(d.end_s));
                time.style.fontSize = '0.85em';
                time.style.color = 'var(--text-muted)';

                const switchBadge = Utils.el('span', 'switch-badge', d.switched ? 'SWITCH' : '');

                row.append(blk, takeBadge, score, time, switchBadge);
                decisionsEl.appendChild(row);
            }
        }
    },

    /**
     * Render normalization stats section in the comp report.
     */
    renderNormStats(container, norm) {
        const title = Utils.el('div', 'config-section-title', 'Normalizacao aplicada');
        container.appendChild(title);

        // Summary line
        const parts = [];
        if (norm.target_bpm) parts.push(`Tempo alvo: ${norm.target_bpm} BPM (${norm.tempo_intensity}%)`);
        if (norm.target_hz)  parts.push(`Pitch alvo: ${norm.target_hz} Hz (${norm.pitch_intensity}%)`);
        if (norm.tempo_spread_bpm !== undefined) parts.push(`Spread: ${norm.tempo_spread_bpm} BPM`);
        if (norm.pitch_spread_cents !== undefined) parts.push(`Spread: ${norm.pitch_spread_cents} cents`);

        if (parts.length) {
            const summary = Utils.el('div', 'norm-summary', parts.join(' · '));
            container.appendChild(summary);
        }

        // Per-take table
        if (norm.per_take && norm.per_take.length > 0) {
            const details = document.createElement('details');
            details.className = 'decisions-details';
            const sum = document.createElement('summary');
            sum.textContent = 'Correcoes por take';
            details.appendChild(sum);

            const table = Utils.el('div', 'norm-table');

            // Header
            const header = Utils.el('div', 'norm-row header');
            header.innerHTML = '<span>Take</span><span>BPM original</span><span>Correcao BPM</span><span>Shift (cents)</span>';
            table.appendChild(header);

            for (const pt of norm.per_take) {
                const row = Utils.el('div', 'norm-row');

                const take = Utils.el('span', 'take-badge', `Take ${pt.take}`);
                take.style.background = Utils.takeColor(pt.take - 1);

                const bpm = Utils.el('span', '', pt.original_bpm ? `${pt.original_bpm}` : '-');
                const correction = Utils.el('span', '',
                    pt.bpm_correction !== undefined ? (pt.bpm_correction > 0 ? `+${pt.bpm_correction}` : `${pt.bpm_correction}`) : '-');
                const cents = Utils.el('span', '',
                    pt.pitch_shift_cents !== undefined ? (pt.pitch_shift_cents > 0 ? `+${pt.pitch_shift_cents}` : `${pt.pitch_shift_cents}`) : '-');

                // Color code significant corrections
                if (pt.bpm_correction && Math.abs(pt.bpm_correction) > 2) {
                    correction.style.color = 'var(--orange)';
                }
                if (pt.pitch_shift_cents && Math.abs(pt.pitch_shift_cents) > 15) {
                    cents.style.color = 'var(--orange)';
                }

                row.append(take, bpm, correction, cents);
                table.appendChild(row);
            }

            details.appendChild(table);
            container.appendChild(details);
        }
    },
};
