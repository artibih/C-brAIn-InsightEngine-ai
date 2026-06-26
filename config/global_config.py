import yaml
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "rag_config.yaml")

CONFIG = {}

def load_configs():
    global CONFIG, QUESTIONS
    try:
        with open(CONFIG_PATH, "r") as f:
            CONFIG.clear()
            CONFIG.update(yaml.safe_load(f) or {})
        
        print("[ConfigWatcher] Reloaded configuration files.")
    except Exception as e:
        print(f"[ConfigWatcher] Failed to reload config: {e}")

load_configs()

class ConfigFileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == CONFIG_PATH:
            print(f"[ConfigWatcher] Detected change in {event.src_path}")
            load_configs()

def start_watcher():
    observer = Observer()
    handler = ConfigFileChangeHandler()
    observer.schedule(handler, path=os.path.dirname(CONFIG_PATH), recursive=False)
    thread = threading.Thread(target=observer.start, daemon=True)
    thread.start()

start_watcher()

