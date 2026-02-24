#!/usr/bin/env python3
"""
Auto-Comper v1.0 — Diego & Claude 2026
Run: python run.py -> http://localhost:8085
"""

import logging
import webbrowser
import threading

logging.basicConfig(level=logging.INFO, format="%(message)s")

from backend.app import create_app

PORT = 8085


def main():
    print(f"""
======================================================
  AUTO-COMPER v1.0
  Comping + Estrutura + Tuning
------------------------------------------------------
  Abra: http://localhost:{PORT}
  Ctrl+C para parar
======================================================
""")
    app = create_app()
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
