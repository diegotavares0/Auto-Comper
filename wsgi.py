"""
Produtora — Production WSGI entry point.
Used by gunicorn in Docker/Railway deployment.
"""

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from backend.app import create_app

app = create_app()

# For local testing with: python wsgi.py
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8085))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
