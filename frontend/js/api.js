/**
 * Produtora — API communication + SSE progress.
 */

const API = {
    /**
     * POST FormData to endpoint, return JSON response.
     */
    async post(url, formData) {
        const resp = await fetch(url, {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.error || `HTTP ${resp.status}`);
        }
        return data;
    },

    /**
     * Fetch comp result by task_id (polls until ready).
     */
    async fetchResult(taskId, maxRetries = 30) {
        for (let i = 0; i < maxRetries; i++) {
            const resp = await fetch(`/api/result/${taskId}`);

            if (resp.status === 202) {
                // Still processing — wait and retry
                await new Promise(r => setTimeout(r, 1000));
                continue;
            }

            const data = await resp.json();
            if (!resp.ok) {
                throw new Error(data.error || `HTTP ${resp.status}`);
            }
            return data;
        }
        throw new Error('Timeout aguardando resultado');
    },

    /**
     * Subscribe to SSE progress stream.
     * @param {string} taskId
     * @param {function} onProgress - (pct, msg) callback
     * @param {function} onComplete - (msg) callback
     * @param {function} onError - (msg) callback
     * @returns {EventSource} - call .close() to stop
     */
    subscribeProgress(taskId, onProgress, onComplete, onError) {
        const es = new EventSource(`/api/progress/${taskId}`);

        es.addEventListener('progress', (e) => {
            try {
                const d = JSON.parse(e.data);
                if (onProgress) onProgress(d.pct, d.msg);
            } catch (_) {}
        });

        es.addEventListener('complete', (e) => {
            try {
                const d = JSON.parse(e.data);
                if (onComplete) onComplete(d.msg);
            } catch (_) {}
            es.close();
        });

        es.addEventListener('error_event', (e) => {
            try {
                const d = JSON.parse(e.data);
                if (onError) onError(d.msg);
            } catch (_) {}
            es.close();
        });

        es.onerror = () => {
            // SSE connection closed (normal after complete/error)
            es.close();
        };

        return es;
    },

    /**
     * Get output file URL.
     */
    outputUrl(filename) {
        return `/api/output/${encodeURIComponent(filename)}`;
    },
};
