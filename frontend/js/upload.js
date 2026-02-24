/**
 * Auto-Comper v1.0 — Drag-and-drop file upload component.
 */

const Upload = {
    files: [],

    init() {
        this.dropZone = Utils.$('#drop-zone');
        this.fileInput = Utils.$('#file-input');
        this.fileList = Utils.$('#file-list');

        // Click to browse
        this.dropZone.addEventListener('click', () => this.fileInput.click());

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            this.addFiles(e.target.files);
            e.target.value = '';  // Reset so same files can be re-added
        });

        // Drag events
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
            this.addFiles(e.dataTransfer.files);
        });
    },

    /**
     * Add files from FileList, filtering for audio formats.
     */
    addFiles(fileList) {
        const validExts = ['.wav', '.flac', '.aiff', '.aif'];
        let added = 0;

        for (const f of fileList) {
            const ext = f.name.toLowerCase().substring(f.name.lastIndexOf('.'));
            if (!validExts.includes(ext)) continue;

            // Skip duplicates
            if (this.files.some(existing => existing.name === f.name && existing.size === f.size)) {
                continue;
            }

            this.files.push(f);
            added++;
        }

        if (added > 0) {
            this.render();
            this.onChangeCallback();
        }
    },

    /**
     * Remove file by index.
     */
    removeFile(idx) {
        this.files.splice(idx, 1);
        this.render();
        this.onChangeCallback();
    },

    /**
     * Clear all files.
     */
    clear() {
        this.files = [];
        this.render();
        this.onChangeCallback();
    },

    /**
     * Render the file list.
     */
    render() {
        this.fileList.innerHTML = '';

        this.files.forEach((f, idx) => {
            const row = Utils.el('div', 'file-row');

            // Icon
            const icon = Utils.el('span', 'file-icon');
            icon.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>';

            // Name
            const name = Utils.el('span', 'file-name', f.name);

            // Size
            const size = Utils.el('span', 'file-size', Utils.formatSize(f.size));

            // Remove button
            const remove = Utils.el('button', 'file-remove', '\u00d7');
            remove.title = 'Remover';
            remove.addEventListener('click', (e) => {
                e.stopPropagation();
                this.removeFile(idx);
            });

            row.append(icon, name, size, remove);
            this.fileList.appendChild(row);
        });
    },

    /**
     * Register a callback for when files change.
     */
    onChange(cb) {
        this._onChange = cb;
    },

    onChangeCallback() {
        if (this._onChange) this._onChange(this.files);
    },
};
