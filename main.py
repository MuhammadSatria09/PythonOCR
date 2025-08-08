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


# from rapidocr import RapidOCR

# engine = RapidOCR()
# --- Part 1: Centralized Configuration ---
# A dedicated class to hold all static configuration variables.
# This makes it easier to change settings without digging through the code.
class Config:
    """Holds all static configuration for the application."""
    DEBUG_MODE = True
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # UI Theming
    BG_COLOR = "#2E2E2E"
    TITLE_BAR_COLOR = "#3c3c3c"
    FG_COLOR = "#FFFFFF"
    TITLE_COLOR = "#E6FF01"
    FONT_FAMILY = "Segoe UI"

    # File Paths (relative to the script)
    TEMPLATE_IMG_PATH = "scan.png"
    EVENT_FILES = ["scraped_character_events.json", "scraped_events_data.json"]

    # Logic Thresholds
    MATCH_THRESHOLD = 85 # Minimum score for a fuzzy match to be considered valid
    TEMPLATE_CONFIDENCE_THRESHOLD = 0.6 # Minimum confidence for template matching

# --- Part 2: Main Application Class ---
# A single, consolidated class for the application. It handles the UI,
# data management, and screen scanning logic in one place for simplicity.
class EventFinderApp(tk.Tk):
    """
    A lightweight application to find and display event information by scanning the screen.
    """
    def __init__(self):
        super().__init__()

        # --- Core Application State ---
        self.events_data = []
        self.template_gray = None
        self.template_width = 0
        self.template_height = 0
        self.last_successful_query = ""
        self.after_id = None
        self.dynamic_widgets = []

        # --- Initialization ---
        self._load_resources()
        self._setup_ui()

        # Start the main application loop
        self.periodic_scan()

    # --- Setup and Resource Loading ---

    def _resource_path(self, relative_path):
        """Gets the absolute path to a resource, for PyInstaller compatibility."""
        try:
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def _load_resources(self):
        """Loads the template image and all event data."""
        # Load Template Image
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

        # Load Event Data
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
        """Configures the window and creates all UI widgets."""
        # Fonts
        self.FONT_NORMAL = tkFont.Font(family=Config.FONT_FAMILY, size=10)
        self.FONT_BOLD = tkFont.Font(family=Config.FONT_FAMILY, size=11, weight="bold")
        self.FONT_TITLE = tkFont.Font(family=Config.FONT_FAMILY, size=12, weight="bold")
        self.FONT_OR = tkFont.Font(family=Config.FONT_FAMILY, size=11, weight="bold",slant="italic")

        # Window Setup
        self.title("Event Finder")
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        self.config(bg=Config.BG_COLOR)
        window_width = 450
        x_pos = self.winfo_screenwidth() - window_width - window_width//3
        self.geometry(f"{window_width}x250+{x_pos}+100")

        # Title Bar
        title_bar = tk.Frame(self, bg=Config.TITLE_BAR_COLOR, relief='raised', bd=0, height=30)
        title_bar.pack(side=tk.TOP, fill=tk.X)
        title_bar.pack_propagate(False)
        self.header_label = tk.Label(title_bar, text="Scanning...", font=self.FONT_TITLE, bg=Config.TITLE_BAR_COLOR, fg=Config.TITLE_COLOR)
        self.header_label.pack(side=tk.LEFT, padx=10)
        close_button = tk.Label(title_bar, text="✕", bg=Config.TITLE_BAR_COLOR, fg="white", font=self.FONT_BOLD, cursor="hand2")
        close_button.pack(side=tk.RIGHT, padx=10)

        # Bindings for dragging and closing
        title_bar.bind("<Button-1>", self.start_drag)
        title_bar.bind("<B1-Motion>", self.do_drag)
        self.header_label.bind("<Button-1>", self.start_drag)
        self.header_label.bind("<B1-Motion>", self.do_drag)
        close_button.bind("<Button-1>", self.quit_app)

        # Content and Footer
        self.content_frame = tk.Frame(self, bg=Config.BG_COLOR)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        footer_frame = tk.Frame(self, bg=Config.BG_COLOR)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.perf_label = tk.Label(footer_frame, text="Performance: -", font=self.FONT_NORMAL, bg=Config.BG_COLOR, fg="#888888")
        self.perf_label.pack()

        self._update_display(None, None, None, None)

    # --- Core Application Loop ---

    def periodic_scan(self):
        """The main loop that orchestrates scanning and UI updates."""
        try:
            scan_result, timings = self._scan_for_text()

            if Config.DEBUG_MODE and scan_result.get("debug_image") is not None:
                self._show_debug_window(scan_result["debug_image"])

            search_latency = 0
            if scan_result["status"] == "success":
                lines = [line.strip() for line in scan_result["text"].splitlines() if line.strip()]
                current_query = lines[-1] if lines else ""
                
                # --- ADDED FOR DEBUGGING ---
                # Print the query that is about to be used for the search.
                if current_query:
                    print(f"[DEBUG] Using query: '{current_query}'")

                if current_query and current_query != self.last_successful_query:
                    self.last_successful_query = current_query
                    event, holder, source, search_latency = self._find_event(current_query)
                    self._update_display(current_query, event, holder, source)
                elif not current_query and self.last_successful_query:
                    self.last_successful_query = ""
                    self._update_display(None, None, None, None)

            elif scan_result["status"] == "template_not_found" and self.last_successful_query:
                self.last_successful_query = ""
                self._update_display(None, None, None, None)

            timings["search"] = search_latency
            self._update_performance_display(timings)

        except Exception as e:
            print(f"An error occurred in the scan loop: {e}")
            self.header_label.config(text="Error. Retrying...")

        self.after_id = self.after(150, self.periodic_scan)



        # --- Screen Scanning and OCR Logic ---

    def _scan_for_text(self):
        """Scans monitors for the template, performs OCR, and returns the text."""
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
            # --- ADDED FOR DEBUGGING ---
            # If a template is visible but below the threshold, still show it in debug mode
            if Config.DEBUG_MODE and debug_image is not None and best_match["max_val"] > 0.1: # Show if confidence is at least 10%
                 confidence_text = f"Confidence: {best_match['max_val']:.2f} (Below Threshold)"
                 text_position = (best_match['max_loc'][0], best_match['max_loc'][1] - 10)
                 cv2.putText(debug_image, confidence_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange text for low confidence
                 cv2.rectangle(debug_image, best_match['max_loc'], (best_match['max_loc'][0] + self.template_width, best_match['max_loc'][1] + self.template_height), (0, 165, 255), 2)
            
            return {"status": "template_not_found", "text": "", "debug_image": debug_image}, timings

        top_left, screenshot_np = best_match["max_loc"], best_match["screenshot_np"]
        roi_x, roi_y = top_left[0], top_left[1] + int(self.template_height // 2.5 )
        roi_w, roi_h = self.template_width + self.template_width // 2, int(self.template_height // 2)
        roi_img_np = screenshot_np[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        if Config.DEBUG_MODE and debug_image is not None:
            # Draw rectangles for template match (green) and OCR region (red)
            cv2.rectangle(debug_image, top_left, (top_left[0] + self.template_width, top_left[1] + self.template_height), (0, 255, 0), 2)
            cv2.rectangle(debug_image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
            
            # --- THIS IS THE ADDED LINE ---
            # Prepare and draw the confidence text on the debug image
            confidence_text = f"Confidence: {best_match['max_val']:.2f}"
            text_position = (top_left[0], top_left[1] - 10) # Position text 10 pixels above the green box
            cv2.putText(debug_image, confidence_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5) # Yellow text

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
    # --- Data Handling and Search Logic ---

    def _normalize_text(self, text: str) -> str:
        """Cleans and standardizes text for reliable matching."""
        if not text: return ""
        return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', '', text.lower())).strip()

    def _find_event(self, query):
        """Finds the best matching event for a given query."""
        start_time = time.perf_counter()
        normalized_query = self._normalize_text(query)
        
        # --- ADDED FOR DEBUGGING ---
        # Print the normalized query to see what the search algorithm is actually using.
        print(f"[DEBUG] Normalized query for matching: '{normalized_query}'")

        if not normalized_query:
            return None, None, None, 0

        best_match = {"score": 0, "event": None, "holder": None, "source": None}
        for holder in self.events_data:
            for event in holder.get("events", []):
                if (score := fuzz.token_set_ratio(normalized_query, event.get("normalized_title", ""))) > best_match["score"]:
                    best_match.update({
                        "score": score,
                        "event": event,
                        "holder": holder.get("event_holder_name") or holder.get("character_name") or "Unknown",
                        "source": holder.get("source_file")
                    })
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        if best_match["score"] >= Config.MATCH_THRESHOLD:
            # --- ADDED FOR DEBUGGING ---
            print(f"[DEBUG] Match Found! Score: {best_match['score']}% for event '{best_match['event']['title']}'")
            return best_match["event"], best_match["holder"], best_match["source"], latency_ms
        
        # --- ADDED FOR DEBUGGING ---
        print(f"[DEBUG] No match found above threshold. Best score was {best_match['score']}%")
        return None, None, None, latency_ms

    # --- UI Update and Management ---

    def _update_display(self, query, found_event, holder_name, source_file):
        """Dynamically updates the UI with event information."""
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
                    is_or = (effect == 'or')
                    effect_label = tk.Label(self.content_frame, text="OR" if is_or else f"  • {effect}", font=self.FONT_OR if is_or else self.FONT_NORMAL, bg=Config.BG_COLOR, fg="#FF7300" if is_or else Config.FG_COLOR, wraplength=380, justify="left")
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

    def quit_app(self, event=None):
        print("Closing application...")
        if self.after_id: self.after_cancel(self.after_id)
        if Config.DEBUG_MODE: cv2.destroyAllWindows()
        self.destroy()

# --- Main Execution Block ---
def main():
    """Main function to initialize and run the application."""
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