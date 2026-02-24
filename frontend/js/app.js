/**
 * Auto-Comper v1.0 — App init, tab routing, global state.
 */

const App = {
    currentTab: 'comp',

    init() {
        // Init components
        Upload.init();
        Comping.init();
        Tuning.init();
        Presets.init();
        Structure.init();

        // Wire upload → comping button state
        Upload.onChange((files) => {
            Comping.updateButton(files);
        });

        // Tab navigation
        const tabs = Utils.$$('.nav-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                this.switchTab(tab.dataset.tab);
            });
        });

        console.log('Auto-Comper v1.0 inicializado');
    },

    /**
     * Switch active tab.
     */
    switchTab(tabName) {
        this.currentTab = tabName;

        // Update nav buttons
        Utils.$$('.nav-tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tabName);
        });

        // Update tab content
        Utils.$$('.tab-content').forEach(c => {
            c.classList.toggle('active', c.id === 'tab-' + tabName);
        });
    },
};

// Boot
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
