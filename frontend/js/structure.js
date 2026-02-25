/**
 * Produtora — Structure tab logic.
 *
 * Detect musical sections (Intro, Verso, Refrao, Ponte, Outro) from a
 * reference take, let the user edit labels, and confirm for use in comping.
 */

const Structure = {
    file: null,
    sections: null,        // detected sections (editable)
    confirmed: false,      // whether the user confirmed the structure
    currentSSE: null,
    ssmData: null,         // SSM thumbnail for visualization
    noveltyData: null,     // novelty curve for visualization

    // Section group colors (matching --purple palette)
    groupColors: [
        '#a87cff', '#6c8cff', '#5cbf78', '#e0a040', '#5cc8d8',
        '#e05858', '#ff8c6c', '#d880c0', '#8cd860', '#c0c060',
    ],

    init() {
        this.dropZone = Utils.$('#struct-drop-zone');
        this.fileInput = Utils.$('#struct-file-input');
        this.fileInfo = Utils.$('#struct-file-info');
        this.btnDetect = Utils.$('#btn-struct-detect');
        this.progressCard = Utils.$('#struct-progress-card');
        this.progressBar = Utils.$('#struct-progress-bar');
        this.progressPct = Utils.$('#struct-progress-pct');
        this.progressMsg = Utils.$('#struct-progress-msg');
        this.resultCard = Utils.$('#struct-result-card');
        this.summaryEl = Utils.$('#struct-summary');
        this.timelineEl = Utils.$('#struct-timeline');
        this.sectionsEl = Utils.$('#struct-sections');
        this.btnConfirm = Utils.$('#btn-struct-confirm');
        this.btnReset = Utils.$('#btn-struct-reset');
        this.confirmedBadge = Utils.$('#struct-confirmed-badge');

        // Drop zone events
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.onFileSelect(e));

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
            const files = e.dataTransfer.files;
            if (files.length > 0) this.setFile(files[0]);
        });

        // Buttons
        this.btnDetect.addEventListener('click', () => this.runDetect());
        this.btnConfirm.addEventListener('click', () => this.confirmStructure());
        this.btnReset.addEventListener('click', () => this.reset());
    },

    /**
     * Handle file selection from input.
     */
    onFileSelect(e) {
        if (e.target.files.length > 0) {
            this.setFile(e.target.files[0]);
        }
    },

    /**
     * Set the reference file.
     */
    setFile(file) {
        const valid = /\.(wav|flac|aiff?|aif)$/i;
        if (!valid.test(file.name)) return;

        this.file = file;
        this.fileInfo.innerHTML = '';

        const row = Utils.el('div', 'file-row');
        const icon = Utils.el('span', 'file-icon', '\u266B');
        const name = Utils.el('span', 'file-name', file.name);
        const size = Utils.el('span', 'file-size', Utils.formatSize(file.size));
        const remove = Utils.el('button', 'file-remove', '\u00D7');
        remove.addEventListener('click', () => {
            this.file = null;
            this.fileInfo.innerHTML = '';
            this.btnDetect.disabled = true;
        });

        row.append(icon, name, size, remove);
        this.fileInfo.appendChild(row);
        this.btnDetect.disabled = false;

        // Clear previous results
        if (this.sections) {
            Utils.hide(this.resultCard);
            this.sections = null;
            this.confirmed = false;
            Utils.hide(this.confirmedBadge);
        }
    },

    /**
     * Run structure detection.
     */
    async runDetect() {
        if (!this.file) return;

        // Reset UI
        this.btnDetect.disabled = true;
        Utils.show(this.progressCard);
        Utils.hide(this.resultCard);
        Utils.hide(this.confirmedBadge);
        this.confirmed = false;
        this.progressBar.style.width = '0%';
        this.progressPct.textContent = '0%';
        this.progressMsg.textContent = 'Enviando arquivo...';

        if (this.currentSSE) {
            this.currentSSE.close();
            this.currentSSE = null;
        }

        const fd = new FormData();
        fd.append('file', this.file);

        try {
            const data = await API.post('/api/structure/detect', fd);

            if (data.error) {
                this.showError(data.error);
                return;
            }

            if (data.task_id) {
                this.currentSSE = API.subscribeProgress(
                    data.task_id,
                    (pct, msg) => this.onProgress(pct, msg),
                    (msg) => this.onComplete(data.task_id, msg),
                    (msg) => this.showError(msg),
                );
            }
        } catch (err) {
            this.showError(err.message);
            this.btnDetect.disabled = false;
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
            this.showResult(result.report);
        } catch (err) {
            this.showError(err.message);
        } finally {
            this.btnDetect.disabled = false;
        }
    },

    showError(msg) {
        this.progressMsg.textContent = 'Erro: ' + msg;
        this.progressMsg.style.color = 'var(--red)';
        this.btnDetect.disabled = false;
        setTimeout(() => {
            this.progressMsg.style.color = '';
        }, 5000);
    },

    /**
     * Display detected structure.
     */
    showResult(report) {
        if (!report || !report.sections || report.sections.length === 0) {
            this.showError('Nenhuma secao detectada');
            return;
        }

        this.sections = report.sections;
        this.ssmData = report.ssm_thumbnail || null;
        this.noveltyData = {
            curve: report.novelty_curve || [],
            times: report.novelty_times || [],
        };

        Utils.show(this.resultCard);
        this.progressMsg.textContent = 'Estrutura detectada!';

        // Summary
        this.summaryEl.textContent =
            `${report.n_sections} secoes, ${report.n_groups} grupos, ` +
            `${Utils.formatTime(report.duration_s)}`;

        // Render timeline + section list
        this.renderTimeline();
        this.renderSections();

        // Draw SSM heatmap
        if (this.ssmData) {
            this.drawSSM();
        }

        // Draw novelty curve
        if (this.noveltyData.curve.length > 0) {
            this.drawNovelty();
        }
    },

    /**
     * Render the colored section timeline bar.
     */
    renderTimeline() {
        this.timelineEl.innerHTML = '';

        if (!this.sections || this.sections.length === 0) return;

        const totalDur = this.sections[this.sections.length - 1].end_s;

        for (const sec of this.sections) {
            const dur = sec.end_s - sec.start_s;
            const pct = (dur / totalDur) * 100;

            const seg = Utils.el('div', 'struct-timeline-segment');
            seg.style.flex = pct;
            seg.style.background = this.groupColors[sec.group % this.groupColors.length];
            seg.textContent = pct > 6 ? sec.label : '';
            seg.title = `${sec.label} (${Utils.formatTime(sec.start_s)} - ${Utils.formatTime(sec.end_s)})`;

            this.timelineEl.appendChild(seg);
        }
    },

    /**
     * Render editable section list.
     */
    renderSections() {
        this.sectionsEl.innerHTML = '';

        if (!this.sections) return;

        for (let i = 0; i < this.sections.length; i++) {
            const sec = this.sections[i];

            const row = Utils.el('div', 'struct-section-row');

            // Color dot
            const dot = Utils.el('div', 'struct-section-dot');
            dot.style.background = this.groupColors[sec.group % this.groupColors.length];

            // Editable label
            const label = document.createElement('input');
            label.type = 'text';
            label.className = 'struct-section-label';
            label.value = sec.label;
            label.dataset.idx = i;
            label.addEventListener('change', (e) => {
                const idx = parseInt(e.target.dataset.idx);
                this.sections[idx].label = e.target.value;
                this.renderTimeline(); // update timeline text
                this.confirmed = false;
                Utils.hide(this.confirmedBadge);
            });

            // Time range
            const time = Utils.el('span', 'struct-section-time',
                `${Utils.formatTime(sec.start_s)} - ${Utils.formatTime(sec.end_s)}`);

            // Group badge
            const group = Utils.el('span', 'struct-section-group', sec.name);

            row.append(dot, label, time, group);
            this.sectionsEl.appendChild(row);
        }
    },

    /**
     * Confirm structure for use in comping.
     */
    confirmStructure() {
        if (!this.sections) return;

        this.confirmed = true;
        Utils.show(this.confirmedBadge);

        // Update the confirm button text
        this.btnConfirm.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"/>
            </svg>
            Confirmado!
        `;
        this.btnConfirm.disabled = true;

        // Re-enable after a moment
        setTimeout(() => {
            this.btnConfirm.disabled = false;
            this.btnConfirm.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
                Confirmar e Usar no Comp
            `;
        }, 2000);
    },

    /**
     * Get confirmed sections for the Comp pipeline.
     * Returns null if not confirmed.
     */
    getConfirmedSections() {
        if (!this.confirmed || !this.sections) return null;
        return this.sections;
    },

    /**
     * Reset the structure tab.
     */
    reset() {
        this.sections = null;
        this.confirmed = false;
        this.ssmData = null;
        this.noveltyData = null;

        Utils.hide(this.resultCard);
        Utils.hide(this.confirmedBadge);
        Utils.hide(this.progressCard);

        this.timelineEl.innerHTML = '';
        this.sectionsEl.innerHTML = '';
    },

    /**
     * Draw SSM heatmap on canvas.
     */
    drawSSM() {
        const canvas = Utils.$('#struct-ssm-canvas');
        if (!canvas || !this.ssmData) return;

        const size = this.ssmData.length;
        canvas.width = size;
        canvas.height = size;
        canvas.style.imageRendering = 'pixelated';

        const ctx = canvas.getContext('2d');
        const img = ctx.createImageData(size, size);

        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const val = this.ssmData[y][x];
                const idx = (y * size + x) * 4;

                // Purple-tinted colormap
                const intensity = Math.floor(val * 255);
                img.data[idx]     = Math.floor(intensity * 0.66); // R
                img.data[idx + 1] = Math.floor(intensity * 0.49); // G
                img.data[idx + 2] = intensity;                     // B
                img.data[idx + 3] = 255;                           // A
            }
        }

        ctx.putImageData(img, 0, 0);
    },

    /**
     * Draw novelty curve on canvas with section boundary markers.
     */
    drawNovelty() {
        const canvas = Utils.$('#struct-novelty-canvas');
        if (!canvas || !this.noveltyData.curve.length) return;

        const W = 680;
        const H = 120;
        canvas.width = W;
        canvas.height = H;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, W, H);

        const curve = this.noveltyData.curve;
        const times = this.noveltyData.times;
        const n = curve.length;
        const maxVal = Math.max(...curve, 0.001);
        const totalDur = times[times.length - 1] || 1;

        // Draw section boundaries as vertical lines
        if (this.sections) {
            ctx.strokeStyle = 'rgba(168, 124, 255, 0.3)';
            ctx.lineWidth = 1;
            for (const sec of this.sections) {
                const x = (sec.start_s / totalDur) * W;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, H);
                ctx.stroke();
            }
        }

        // Draw novelty curve
        ctx.beginPath();
        ctx.strokeStyle = '#a87cff';
        ctx.lineWidth = 1.5;

        for (let i = 0; i < n; i++) {
            const x = (i / (n - 1)) * W;
            const y = H - (curve[i] / maxVal) * (H - 10) - 5;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Fill under curve
        ctx.lineTo(W, H);
        ctx.lineTo(0, H);
        ctx.closePath();
        ctx.fillStyle = 'rgba(168, 124, 255, 0.08)';
        ctx.fill();
    },
};
