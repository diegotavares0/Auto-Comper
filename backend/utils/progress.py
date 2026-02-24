"""
Thread-safe progress manager with SSE streaming.
"""

import json
import queue
import threading
from typing import Dict, List


class ProgressManager:
    """Manages progress state for multiple concurrent tasks with SSE support."""

    def __init__(self):
        self._tasks: Dict[str, dict] = {}
        self._listeners: Dict[str, List[queue.Queue]] = {}
        self._lock = threading.Lock()

    def start(self, task_id: str, message: str = "Iniciando..."):
        with self._lock:
            self._tasks[task_id] = {
                "percent": 0,
                "message": message,
                "busy": True,
                "event_type": "progress",
            }

    def update(self, task_id: str, percent: int, message: str):
        with self._lock:
            self._tasks[task_id] = {
                "percent": percent,
                "message": message,
                "busy": True,
                "event_type": "progress",
            }
            state = self._tasks[task_id].copy()

        # Notify SSE listeners
        for q in self._listeners.get(task_id, []):
            try:
                q.put_nowait(state)
            except queue.Full:
                pass

    def complete(self, task_id: str, message: str = "Concluido!"):
        with self._lock:
            self._tasks[task_id] = {
                "percent": 100,
                "message": message,
                "busy": False,
                "event_type": "complete",
            }
            state = self._tasks[task_id].copy()

        for q in self._listeners.get(task_id, []):
            try:
                q.put_nowait(state)
            except queue.Full:
                pass

    def error(self, task_id: str, message: str):
        with self._lock:
            self._tasks[task_id] = {
                "percent": 0,
                "message": f"Erro: {message}",
                "busy": False,
                "event_type": "error_event",
            }
            state = self._tasks[task_id].copy()

        for q in self._listeners.get(task_id, []):
            try:
                q.put_nowait(state)
            except queue.Full:
                pass

    def get_state(self, task_id: str) -> dict:
        with self._lock:
            return self._tasks.get(task_id, {
                "percent": 0,
                "message": "Pronto",
                "busy": False,
            }).copy()

    def subscribe(self, task_id: str) -> queue.Queue:
        q = queue.Queue(maxsize=50)
        with self._lock:
            if task_id not in self._listeners:
                self._listeners[task_id] = []
            self._listeners[task_id].append(q)
        return q

    def unsubscribe(self, task_id: str, q: queue.Queue):
        with self._lock:
            if task_id in self._listeners:
                try:
                    self._listeners[task_id].remove(q)
                except ValueError:
                    pass

    def stream(self, q: queue.Queue):
        """Generator for SSE events with named event types."""
        while True:
            try:
                state = q.get(timeout=30)

                # Extract event type and build SSE payload
                event_type = state.pop("event_type", "progress")
                payload = {
                    "pct": state["percent"],
                    "msg": state["message"],
                }

                # Named SSE event: "event: <type>\ndata: {...}\n\n"
                yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

                # Only break on complete/error (busy=False), NOT on progress at 100%
                if not state.get("busy", True):
                    break

            except queue.Empty:
                # Keep-alive to prevent connection timeout
                yield ": keepalive\n\n"

    def make_callback(self, task_id: str):
        """Create a progress callback function for a task."""
        def callback(percent: int, message: str):
            self.update(task_id, percent, message)
        return callback


# Global singleton
progress_manager = ProgressManager()
