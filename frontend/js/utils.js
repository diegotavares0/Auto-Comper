/**
 * Auto-Comper v1.0 — DOM helpers, formatters, colors.
 */

const Utils = {
    /**
     * Format seconds as m:ss.
     */
    formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    },

    /**
     * Format file size in human-readable form.
     */
    formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    },

    /**
     * Get a distinct color for a take index.
     */
    takeColor(idx) {
        const colors = [
            '#6c8cff', // blue (accent)
            '#5cbf78', // green
            '#e0a040', // orange
            '#a87cff', // purple
            '#5cc8d8', // cyan
            '#e05858', // red
            '#ff8c6c', // coral
            '#8cd860', // lime
            '#d880c0', // pink
            '#c0c060', // yellow-green
            '#60a0d8', // sky
            '#d89060', // tan
            '#7c60d8', // indigo
            '#60d8a0', // mint
            '#d86080', // rose
            '#a0d860', // chartreuse
        ];
        return colors[idx % colors.length];
    },

    /**
     * Shorthand for querySelector.
     */
    $(sel) {
        return document.querySelector(sel);
    },

    /**
     * Shorthand for querySelectorAll.
     */
    $$(sel) {
        return document.querySelectorAll(sel);
    },

    /**
     * Create an element with classes and optional text.
     */
    el(tag, className, text) {
        const e = document.createElement(tag);
        if (className) e.className = className;
        if (text !== undefined) e.textContent = text;
        return e;
    },

    /**
     * Show/hide element.
     */
    show(el) {
        if (typeof el === 'string') el = document.querySelector(el);
        if (el) el.style.display = '';
    },

    hide(el) {
        if (typeof el === 'string') el = document.querySelector(el);
        if (el) el.style.display = 'none';
    },

    /**
     * Set inner HTML safely (only for trusted content).
     */
    setHTML(el, html) {
        if (typeof el === 'string') el = document.querySelector(el);
        if (el) el.innerHTML = html;
    },
};
