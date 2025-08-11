import json
import time
import os
import re
import mss
import pytesseract
from PIL import Image
import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import font as tkFont
from tkinter import messagebox
from thefuzz import fuzz
import threading
import queue

# --- Part 1: Centralized Configuration ---
class Config:
    """Holds all static configuration for the application."""
    DEBUG_MODE = False
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    BG_COLOR = "#2E2E2E"
    TITLE_BAR_COLOR = "#3c3c3c"
    FG_COLOR = "#FFFFFF"
    TITLE_COLOR = "#E6FF01"
    FONT_FAMILY = "Segoe UI"

    TEMPLATE_IMG_PATH = "scan.png"
    EVENT_FILES = ["scraped_character_events.json", "scraped_events_data.json"]

    MATCH_THRESHOLD = 85
    TEMPLATE_CONFIDENCE_THRESHOLD = 0.7

# --- Part 2: Main Application Class ---
class EventFinderApp(tk.Tk):
    """
    A lightweight application to find and display event information by scanning the screen.
    """
    def __init__(self):
        super().__init__()

        # --- MODIFIED: Threading and Queue for Parallelization ---
        self.results_queue = queue.Queue()
        self.is_running = threading.Event()
        self.is_running.set() # This event will be used to signal the thread to stop

        # --- Core Application State ---
        self.events_data = []
        self.template_gray = None
        self.template_width = 0
        self.template_height = 0
        self.last_successful_query = "" # Now represents the currently displayed item
        self.after_id = None
        self.dynamic_widgets = []

        # --- Initialization ---
        self._load_resources()
        self._setup_ui()

        # --- MODIFIED: Start the background worker thread ---
        self.worker_thread = threading.Thread(target=self._worker_scan_loop, daemon=True)
        self.worker_thread.start()

        # --- MODIFIED: Start the UI result processing loop ---
        self._process_results_loop()

    # --- Setup and Resource Loading (Unchanged) ---
    def _resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def _load_resources(self):
        template_path = self._resource_path(Config.TEMPLATE_IMG_PATH)
        try:
            self.template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if self.template_gray is None:
                raise IOError(f"OpenCV could not read the template image: {template_path}")
            self.template_height, self.template_width = self.template_gray.shape
            print(f"Template '{os.path.basename(template_path)}' loaded.")
        except Exception as e:
            messagebox.showerror("Fatal Error", f"Could not load template image.\n\nError: {e}")
            self.destroy()
            sys.exit(1)

        print("Loading event data...")
        for filename in Config.EVENT_FILES:
            path = self._resource_path(filename)
            display_name = os.path.basename(path)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        item['source_file'] = display_name
                        for event in item.get("events", []):
                            event["normalized_title"] = self._normalize_text(event.get("title", ""))
                    self.events_data.extend(data)
                    print(f"  - Loaded {display_name}")
            except Exception as e:
                print(f"  - Warning: Could not load or process {display_name}. Error: {e}")

        if not self.events_data:
            messagebox.showerror("Fatal Error", "Could not load any event data files.")
            self.destroy()
            sys.exit(1)
        print("Data loading complete.")

    def _setup_ui(self):
        self.FONT_NORMAL = tkFont.Font(family=Config.FONT_FAMILY, size=10)
        self.FONT_BOLD = tkFont.Font(family=Config.FONT_FAMILY, size=11, weight="bold")
        self.FONT_TITLE = tkFont.Font(family=Config.FONT_FAMILY, size=12, weight="bold")
        self.FONT_OR = tkFont.Font(family=Config.FONT_FAMILY, size=11, weight="bold",slant="italic")
        
        self.title("Event Finder")
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        self.config(bg=Config.BG_COLOR)
        window_width = 375
        x_pos = self.winfo_screenwidth() - window_width - window_width//3
        self.geometry(f"{window_width}x250+{x_pos}+100")
        title_bar = tk.Frame(self, bg=Config.TITLE_BAR_COLOR, relief='raised', bd=0, height=50)
        title_bar.pack(side=tk.TOP, fill=tk.X)
        title_bar.pack_propagate(False)
        self.header_label = tk.Label(title_bar, text="Scanning...", font=self.FONT_TITLE, bg=Config.TITLE_BAR_COLOR, fg=Config.TITLE_COLOR)
        self.header_label.pack(side=tk.LEFT, padx=10)
        close_button = tk.Label(title_bar, text="✕", bg=Config.TITLE_BAR_COLOR, fg="white", font=self.FONT_BOLD, cursor="hand2")
        close_button.pack(side=tk.RIGHT, padx=10)
        title_bar.bind("<Button-1>", self.start_drag)
        title_bar.bind("<B1-Motion>", self.do_drag)
        self.header_label.bind("<Button-1>", self.start_drag)
        self.header_label.bind("<B1-Motion>", self.do_drag)
        close_button.bind("<Button-1>", self.quit_app)
        self.content_frame = tk.Frame(self, bg=Config.BG_COLOR)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        footer_frame = tk.Frame(self, bg=Config.BG_COLOR)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.perf_label = tk.Label(footer_frame, text="Performance: -", font=self.FONT_NORMAL, bg=Config.BG_COLOR, fg="#888888")
        self.perf_label.pack()
        self._update_display(None, None, None, None)

    # --- NEW: Worker thread loop ---
    def _worker_scan_loop(self):
        """
        Runs in a separate thread to perform all heavy processing (scan, OCR, search)
        without blocking the UI.
        """
        while self.is_running.is_set():
            loop_start_time = time.perf_counter()

            # Perform the entire scan and search operation
            scan_result, timings = self._scan_for_text()
            
            result_package = {"timings": timings, "scan_status": scan_result["status"], "debug_image": scan_result.get("debug_image")}

            if scan_result["status"] == "success":
                lines = [line.strip() for line in scan_result["text"].splitlines() if line.strip()]
                query = lines[-1] if lines else ""
                result_package["query"] = query
                
                if query:
                    event, holder, source, search_latency = self._find_event(query)
                    timings["search"] = search_latency
                    result_package.update({"event": event, "holder": holder, "source": source})
                else:
                    timings["search"] = 0
                    result_package.update({"event": None, "holder": None, "source": None})
            else:
                timings["search"] = 0
                result_package.update({"query": None, "event": None, "holder": None, "source": None})

            # Put the finished result into the thread-safe queue for the main thread
            self.results_queue.put(result_package)

            # Control the scan rate to aim for a cycle time of ~150ms (0.15s)
            loop_duration = time.perf_counter() - loop_start_time
            sleep_time = max(0, 0.15 - loop_duration)
            time.sleep(sleep_time)

    # --- NEW: UI update loop ---
    def _process_results_loop(self):
        """
        Runs in the main UI thread. Checks the queue for results from the worker
        and updates the UI accordingly.
        """
        try:
            # Get a result from the queue without blocking
            result = self.results_queue.get_nowait()

            # Unpack the result
            query = result["query"]
            found_event = result["event"]
            holder = result["holder"]
            source = result["source"]
            scan_status = result["scan_status"]

            # Use a simple key to track what is currently on screen
            # This prevents redrawing the same information repeatedly
            current_display_key = f"{found_event['title'] if found_event else 'None'}"

            if self.last_successful_query != current_display_key:
                self.last_successful_query = current_display_key
                self._update_display(query, found_event, holder, source)

            # Show debug window if enabled and an image is available
            if Config.DEBUG_MODE and result.get("debug_image") is not None:
                self._show_debug_window(result["debug_image"])

            # Update performance stats
            self._update_performance_display(result["timings"])

        except queue.Empty:
            # The queue was empty, which is normal. Do nothing.
            pass
        except Exception as e:
            print(f"Error while processing queue result: {e}")

        # Schedule the next check
        self.after_id = self.after(150, self._process_results_loop)

    # --- Screen Scanning and OCR Logic (Unchanged) ---
    def _scan_for_text(self):
        timings = {"capture": 0, "template": 0, "ocr": 0}
        best_match = {"max_val": 0, "max_loc": None, "screenshot_np": None}
        with mss.mss() as sct:
            start_time = time.perf_counter()
            for monitor in sct.monitors[1:]:
                sct_img = sct.grab(monitor)
                screenshot_np = np.array(sct_img)
                screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_BGRA2GRAY)
                res = cv2.matchTemplate(screenshot_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_match["max_val"]:
                    best_match.update({"max_val": max_val, "max_loc": max_loc, "screenshot_np": screenshot_np})
            timings["capture"] = timings["template"] = (time.perf_counter() - start_time) * 1000
        debug_image = cv2.cvtColor(best_match["screenshot_np"], cv2.COLOR_RGBA2BGR) if Config.DEBUG_MODE and best_match["screenshot_np"] is not None else None
        if best_match["max_val"] < Config.TEMPLATE_CONFIDENCE_THRESHOLD:
            if Config.DEBUG_MODE and debug_image is not None and best_match["max_val"] > 0.1:
                confidence_text = f"Confidence: {best_match['max_val']:.2f} (Below Threshold)"
                text_position = (best_match['max_loc'][0], best_match['max_loc'][1] - 10)
                cv2.putText(debug_image, confidence_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.rectangle(debug_image, best_match['max_loc'], (best_match['max_loc'][0] + self.template_width, best_match['max_loc'][1] + self.template_height), (0, 165, 255), 2)
            return {"status": "template_not_found", "text": "", "debug_image": debug_image}, timings
        top_left, screenshot_np = best_match["max_loc"], best_match["screenshot_np"]
        roi_x, roi_y = top_left[0] + int(top_left[0] // 45), top_left[1] + int(self.template_height // 2.5 )
        roi_w, roi_h = self.template_width + int(self.template_width // 2.25), self.template_height // 2
        roi_img_np = screenshot_np[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        if Config.DEBUG_MODE and debug_image is not None:
            cv2.rectangle(debug_image, top_left, (top_left[0] + self.template_width, top_left[1] + self.template_height), (0, 255, 0), 2)
            cv2.rectangle(debug_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
            confidence_text = f"Confidence: {best_match['max_val']:.2f}"
            text_position = (top_left[0], top_left[1] - 10)
            cv2.putText(debug_image, confidence_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5)
        roi_gray = cv2.cvtColor(roi_img_np, cv2.COLOR_BGRA2GRAY)
        _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        start_ocr = time.perf_counter()
        try:
            text = pytesseract.image_to_string(Image.fromarray(roi_binary), lang='eng', config='--psm 6')
            timings["ocr"] = (time.perf_counter() - start_ocr) * 1000
            print(f"[DEBUG] OCR Raw Output: {repr(text)}")
            return {"status": "success", "text": text.strip(), "debug_image": debug_image}, timings
        except pytesseract.TesseractError as e:
            print(f"Pytesseract Error: {e}")
            return {"status": "ocr_error", "text": "", "debug_image": debug_image}, timings

    # --- Data Handling and Search Logic (Unchanged) ---
    def _normalize_text(self, text: str) -> str:
        if not text: return ""
        return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()

    def _find_event(self, query):
        start_time = time.perf_counter()
        normalized_query = self._normalize_text(query)
        print(f"[DEBUG] Normalized query for matching: '{normalized_query}'")
        if not normalized_query:
            return None, None, None, 0
        best_match = {"score": 0, "event": None, "holder": None, "source": None}
        for holder in self.events_data:
            for event in holder.get("events", []):
                if (score := fuzz.token_set_ratio(normalized_query, event.get("normalized_title", ""))) > best_match["score"]:
                    best_match.update({
                        "score": score, "event": event,
                        "holder": holder.get("event_holder_name") or holder.get("character_name") or "Unknown",
                        "source": holder.get("source_file")
                    })
        latency_ms = (time.perf_counter() - start_time) * 1000
        if best_match["score"] >= Config.MATCH_THRESHOLD:
            print(f"[DEBUG] Match Found! Score: {best_match['score']}% for event '{best_match['event']['title']}'")
            return best_match["event"], best_match["holder"], best_match["source"], latency_ms
        print(f"[DEBUG] No match found above threshold. Best score was {best_match['score']}%")
        return None, None, None, latency_ms

    # --- UI Update and Management (Unchanged) ---
    def _update_display(self, query, found_event, holder_name, source_file):
        for widget in self.dynamic_widgets:
            widget.destroy()
        self.dynamic_widgets = []
        if found_event:
            self.header_label.config(text=f"Event: {found_event['title']}")
            for option in found_event.get('options', []):
                choice_label = tk.Label(self.content_frame, text=f"▶ {option.get('choice', 'N/A')}", font=self.FONT_BOLD, bg=Config.BG_COLOR, fg="#FFFB00", wraplength=400, justify="left")
                choice_label.pack(anchor="w", pady=(8, 2))
                self.dynamic_widgets.append(choice_label)
                for effect in option.get('effects', []):
                    is_or = (effect[:2] == 'or') or ("Randomly either" in effect)
                    effect_label = tk.Label(self.content_frame,
                                            text=f"  • {effect}",
                                            font=self.FONT_OR if is_or else self.FONT_NORMAL,
                                            bg=Config.BG_COLOR,
                                            fg="#FF7300" if is_or else Config.FG_COLOR,
                                            wraplength=380,
                                            justify="center" if is_or else "left")
                    effect_label.pack(anchor="w", padx=10)
                    self.dynamic_widgets.append(effect_label)
        else:
            self.header_label.config(text="Waiting...")
            msg = f"No match for: '{query}'" if query else "Waiting for event on screen..."
            no_event_label = tk.Label(self.content_frame, text=msg, font=self.FONT_NORMAL, bg=Config.BG_COLOR, fg=Config.FG_COLOR, wraplength=400, justify="center")
            no_event_label.pack(pady=20)
            self.dynamic_widgets.append(no_event_label)
        self.update_idletasks()
        self.geometry(f'{self.winfo_width()}x{self.winfo_reqheight()}')

    def _update_performance_display(self, timings):
        perf_text = (f"Capture: {timings['capture']:.0f}ms | "
                     f"Template: {timings['template']:.0f}ms | "
                     f"OCR: {timings['ocr']:.0f}ms | "
                     f"Search: {timings['search']:.0f}ms")
        self.perf_label.config(text=perf_text)

    def _show_debug_window(self, image_np):
        h, w, _ = image_np.shape
        display_w = min(1280, w)
        display_h = int(h * (display_w / w))
        resized_image = cv2.resize(image_np, (display_w, display_h))
        cv2.imshow("Debug Feed", resized_image)
        cv2.waitKey(1)

    def start_drag(self, event):
        self._offset_x, self._offset_y = event.x, event.y

    def do_drag(self, event):
        self.geometry(f'+{self.winfo_pointerx() - self._offset_x}+{self.winfo_pointery() - self._offset_y}')

    # --- MODIFIED: quit_app to handle the thread ---
    def quit_app(self, event=None):
        print("Closing application...")
        self.is_running.clear() # Signal the worker thread to stop
        if self.after_id: self.after_cancel(self.after_id)
        if Config.DEBUG_MODE: cv2.destroyAllWindows()
        self.destroy()

# --- Main Execution Block ---
def main():
    if not os.path.exists(Config.TESSERACT_PATH):
        messagebox.showerror("Tesseract Not Found", f"Tesseract not found at:\n{Config.TESSERACT_PATH}\nPlease edit the Config class in the script.")
        return
    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
    try:
        app = EventFinderApp()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Unhandled Exception", f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()