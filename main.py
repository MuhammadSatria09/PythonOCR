import json
import time
import os
import re
import mss
import mss.tools
import pytesseract
from PIL import Image
import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import font as tkFont
from tkinter import messagebox

# Make sure to install it first: pip install thefuzz python-Levenshtein
from thefuzz import fuzz

# --- Helper function to find data files ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- CONFIGURATION ---
DEBUG_MODE = False # Set to False to hide the debug window
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TEMPLATE_PATH = resource_path("scan.png")
FILES_TO_LOAD = [
    resource_path("scraped_character_events.json"),
    resource_path("scraped_events_data.json")
]
REF_ROI_OFFSET_X = 0

# --- OPTIMIZATION: Load template and get dimensions ONCE at startup ---
try:
    TEMPLATE_GRAY = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if TEMPLATE_GRAY is None:
        raise IOError("OpenCV could not read the template image file.")
    
    # We still need the original color image to get the reference height for the OCR crop
    template_image_for_size = cv2.imread(TEMPLATE_PATH)
    if template_image_for_size is None:
         raise IOError("OpenCV could not read the template image file for sizing.")

    REF_ROI_HEIGHT, REF_ROI_WIDTH, _ = template_image_for_size.shape
    REF_ROI_OFFSET_Y = REF_ROI_HEIGHT // 3
    print(f"Template loaded. ROI dimensions set to: Width={REF_ROI_WIDTH}, Height={REF_ROI_HEIGHT}")

except Exception as e:
    print(f"FATAL ERROR: Could not load template image '{os.path.basename(TEMPLATE_PATH)}'.")
    print(f"Reason: {e}")
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Fatal Error", f"Could not load '{os.path.basename(TEMPLATE_PATH)}'. The application cannot start.\n\nError: {e}")
    root.destroy()
    sys.exit()


# --- OCR AND SCREEN CAPTURE FUNCTION (OPTIMIZED) ---
def capture_and_ocr_screen_with_template(loaded_template_gray, ref_roi_offset_x, ref_roi_offset_y, ref_roi_width, ref_roi_height):
    timings = {"capture": 0, "template_matching": 0, "ocr": 0}
    debug_image = None
    
    start_capture = time.perf_counter()
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
    timings["capture"] = (time.perf_counter() - start_capture) * 1000

    screenshot_np = np.array(sct_img)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    screenshot_gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    
    start_match = time.perf_counter()
    # --- OPTIMIZATION: Use the pre-loaded template image ---
    res = cv2.matchTemplate(screenshot_gray, loaded_template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    timings["template_matching"] = (time.perf_counter() - start_match) * 1000

    threshold = 0.5
    if max_val >= threshold:
        top_left = max_loc
        x1, y1 = top_left[0] + ref_roi_offset_x, top_left[1] + ref_roi_offset_y
        x2, y2 = x1 + ref_roi_width, y1 + ref_roi_height // 2

        if x2 <= x1 or y2 <= y1:
            return {"text": "", "status": "error_invalid_roi", "timings": timings, "debug_image": None}

        if DEBUG_MODE:
            bottom_right = (top_left[0] + loaded_template_gray.shape[1], top_left[1] + loaded_template_gray.shape[0])
            cv2.rectangle(screenshot_bgr, top_left, bottom_right, (0, 255, 0), 2)
            cv2.rectangle(screenshot_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
            score_text = f"Match: {max_val:.2f}"
            cv2.putText(screenshot_bgr, score_text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            debug_image = screenshot_bgr

        roi_img_np = screenshot_np[y1:y2, x1:x2]
        start_ocr = time.perf_counter()
        text = pytesseract.image_to_string(Image.fromarray(roi_img_np), lang='eng')
        timings["ocr"] = (time.perf_counter() - start_ocr) * 1000
        return {"text": text.strip(), "status": "success", "timings": timings, "debug_image": debug_image}
    else:
        if DEBUG_MODE:
            debug_image = screenshot_bgr
        return {"text": "", "status": "template_not_found", "timings": timings, "debug_image": debug_image}

# --- JSON DATA HANDLING (OPTIMIZED) ---
def load_events(filenames):
    combined_data = []
    print("Loading and pre-processing event data...")
    for filename in filenames:
        display_name = os.path.basename(filename)
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    item['source_file'] = display_name
                    # --- OPTIMIZATION: Pre-process and cache normalized titles ---
                    for event in item.get("events", []):
                        title = event.get("title", "")
                        event["normalized_title"] = normalize_title(title)
                combined_data.extend(data)
                print(f"  - Successfully loaded and processed {display_name}")
        except FileNotFoundError:
            print(f"  - Warning: '{display_name}' not found. Skipping.")
        except json.JSONDecodeError:
            print(f"  - Warning: '{display_name}' is not a valid JSON. Skipping.")
    
    if not combined_data:
        print("\nError: Could not load any event data. Exiting.")
        return None
    
    print("Data loading complete.")
    return combined_data

# --- Title normalization ---
def normalize_title(title: str) -> str:
    if not title: return ""
    normalized = title.lower()
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

# --- Event finding logic (OPTIMIZED) ---
def find_event(events_data, search_query):
    start_search = time.perf_counter()
    normalized_query = normalize_title(search_query)
    if not normalized_query: return None, None, None, 0

    best_match = {"score": 0, "event": None, "holder": None, "source": None}
    MATCH_THRESHOLD = 85
    for holder in events_data:
        for event in holder.get("events", []):
            # --- OPTIMIZATION: Use the pre-processed title ---
            normalized_json_title = event.get("normalized_title") # No calculation needed here
            if not normalized_json_title: continue
            
            score = fuzz.token_set_ratio(normalized_query, normalized_json_title)
            if score > best_match["score"]:
                best_match["score"] = score
                best_match["event"] = event
                best_match["holder"] = holder.get("event_holder_name") or holder.get("character_name") or "Unknown Holder"
                best_match["source"] = holder.get("source_file")
    
    latency_ms = (time.perf_counter() - start_search) * 1000
    if best_match["score"] >= MATCH_THRESHOLD:
        print(f"Found match '{best_match['event']['title']}' with score {best_match['score']}% for query '{search_query}'")
        return best_match["event"], best_match["holder"], best_match["source"], latency_ms
    else:
        return None, None, None, latency_ms

# --- GUI APPLICATION CLASS (OPTIMIZED) ---
class App(tk.Tk):
    def __init__(self, events_data):
        super().__init__()
        self.events_data = events_data
        self.last_successful_query = ""
        self.after_id = None
        self.title("Event Finder")

        # --- Window Setup ---
        screen_width = self.winfo_screenwidth()
        x_position = int(screen_width - screen_width / 3)
        y_position = 100
        self.geometry(f"450x200+{x_position}+{y_position}")
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        
        # --- Theming and Fonts ---
        BG_COLOR = "#2E2E2E"
        TITLE_BAR_COLOR = "#3c3c3c"
        FG_COLOR = "#FFFFFF"
        TITLE_COLOR = "#E6FF01"
        FONT_NORMAL = tkFont.Font(family="Segoe UI", size=10)
        FONT_BOLD = tkFont.Font(family="Segoe UI", size=11, weight="bold")
        FONT_TITLE = tkFont.Font(family="Segoe UI", size=12, weight="bold")
        
        # --- Title Bar Frame ---
        title_bar = tk.Frame(self, bg=TITLE_BAR_COLOR, relief='raised', bd=0)
        title_bar.pack(side=tk.TOP, fill=tk.X)

        close_button = tk.Label(title_bar, text=" X ", bg="#C70039", fg="white", font=FONT_BOLD, cursor="hand2")
        close_button.pack(side=tk.RIGHT, padx=4, pady=4)
        close_button.bind("<Button-1>", self.quit_app)

        self.header_label = tk.Label(title_bar, text="Scanning for event...", font=FONT_TITLE, bg=TITLE_BAR_COLOR, fg=TITLE_COLOR, wraplength=380, justify="left")
        self.header_label.pack(side=tk.LEFT, padx=10, pady=4)

        # Drag-to-move functionality
        self._offset_x = 0
        self._offset_y = 0
        title_bar.bind('<Button-1>', self.start_drag)
        title_bar.bind('<B1-Motion>', self.do_drag)
        self.header_label.bind('<Button-1>', self.start_drag)
        self.header_label.bind('<B1-Motion>', self.do_drag)

        # --- Main Content Container ---
        content_container = tk.Frame(self, bg=BG_COLOR, bd=2, relief="groove")
        content_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        main_frame = tk.Frame(content_container, bg=BG_COLOR, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        footer_frame = tk.Frame(main_frame, bg=BG_COLOR)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        self.content_frame = tk.Frame(main_frame, bg=BG_COLOR)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.perf_label = tk.Label(footer_frame, text="Performance: -", font=FONT_NORMAL, bg=BG_COLOR, fg="#888888")
        self.perf_label.pack()

        # --- OPTIMIZATION: Create a pool of widgets to be reused ---
        self.option_widgets = []
        MAX_OPTIONS_DISPLAY = 5  # Max choices to show
        MAX_EFFECTS_PER_OPTION = 4 # Max effects per choice
        for _ in range(MAX_OPTIONS_DISPLAY):
            choice_label = tk.Label(self.content_frame, font=tkFont.Font(family="Segoe UI", size=10, weight="bold"), bg=BG_COLOR, fg="#FFC107", wraplength=400, justify="left")
            effects_labels = []
            for _ in range(MAX_EFFECTS_PER_OPTION):
                effect_label = tk.Label(self.content_frame, font=tkFont.Font(family="Segoe UI", size=10), bg=BG_COLOR, fg=FG_COLOR, wraplength=380, justify="left")
                effects_labels.append(effect_label)
            self.option_widgets.append({"choice": choice_label, "effects": effects_labels})
        
        self.no_event_label = tk.Label(self.content_frame, text="", font=FONT_NORMAL, bg=BG_COLOR, fg=FG_COLOR, wraplength=400, justify="left")
        
        self.update_idletasks()
        self.geometry('')
        self.periodic_scan()

    def start_drag(self, event):
        self._offset_x = event.x
        self._offset_y = event.y

    def do_drag(self, event):
        x = self.winfo_pointerx() - self._offset_x
        y = self.winfo_pointery() - self._offset_y
        self.geometry(f'+{x}+{y}')

    def quit_app(self, event=None):
        if self.after_id: self.after_cancel(self.after_id)
        if DEBUG_MODE: cv2.destroyAllWindows()
        self.destroy()

    def update_display(self, query, found_event, holder_name, source_file, timings):
        # --- OPTIMIZATION: Hide all reusable widgets first ---
        self.no_event_label.pack_forget()
        for group in self.option_widgets:
            group["choice"].pack_forget()
            for label in group["effects"]:
                label.pack_forget()

        if found_event:
            event_title = found_event['title']
            display_header = f"Trainee Event: {event_title}" if source_file == "scraped_character_events.json" else f"{holder_name}: {event_title}"
            self.header_label.config(text=display_header)

            # Re-use existing labels to show the new content
            options = found_event.get('options', [])
            for i, option in enumerate(options):
                if i >= len(self.option_widgets): break # Stop if we have more options than labels
                
                widget_group = self.option_widgets[i]
                choice_label = widget_group["choice"]
                choice_text = f"▶ {option.get('choice', 'N/A')}"
                choice_label.config(text=choice_text)
                choice_label.pack(anchor="w", pady=(5, 2))
                
                effects = option.get('effects', [])
                for j, effect in enumerate(effects):
                    if j >= len(widget_group["effects"]): break
                    effect_label = widget_group["effects"][j]
                    effect_label.config(text=f"  • {effect}")
                    effect_label.pack(anchor="w", padx=5)
        else:
            self.header_label.config(text="Searching...")
            self.no_event_label.config(text=f"No event match for: '{query}'")
            self.no_event_label.pack(anchor="w")

        perf_text = (f"OCR: {timings['ocr']:.0f}ms | Template: {timings['template_matching']:.0f}ms | Search: {timings['search']:.0f}ms")
        self.perf_label.config(text=perf_text)
        self.update_idletasks()
        self.geometry('')

    def periodic_scan(self):
        # --- OPTIMIZATION: Pass the pre-loaded template to the function ---
        ocr_result = capture_and_ocr_screen_with_template(
            TEMPLATE_GRAY, REF_ROI_OFFSET_X, REF_ROI_OFFSET_Y, REF_ROI_WIDTH, REF_ROI_HEIGHT
        )
        
        if DEBUG_MODE and ocr_result["debug_image"] is not None:
            h, w, _ = ocr_result["debug_image"].shape
            display_w = 1280
            display_h = int(h * (display_w / w))
            resized_debug_image = cv2.resize(ocr_result["debug_image"], (display_w, display_h))
            cv2.imshow("Debug Feed", resized_debug_image)
            cv2.waitKey(1)

        cycle_timings = ocr_result["timings"]
        cycle_timings["search"] = 0
        if ocr_result["status"] not in ["success", "template_not_found"]:
            self.header_label.config(text=f"Error: {ocr_result['status']}")
            self.after_id = self.after(1000, self.periodic_scan)
            return

        ocr_text = ocr_result["text"]
        current_query = ""
        if ocr_text:
            lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
            if lines: current_query = lines[-1]

        if current_query and current_query != self.last_successful_query:
            self.last_successful_query = current_query
            found_event, holder_name, source_file, search_latency = find_event(self.events_data, current_query)
            cycle_timings["search"] = search_latency
            self.update_display(current_query, found_event, holder_name, source_file, cycle_timings)
        elif not current_query and self.last_successful_query:
            # This logic block handles when the event disappears from the screen
            self.last_successful_query = ""
            self.header_label.config(text="Scanning... (Template not found)")
            
            # Hide all widgets
            self.no_event_label.pack_forget()
            for group in self.option_widgets:
                group["choice"].pack_forget()
                for label in group["effects"]:
                    label.pack_forget()

            self.update_idletasks()
            self.geometry('')

        self.after_id = self.after(100, self.periodic_scan)

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # --- OPTIMIZATION: Load and process data before starting the GUI ---
    events_data = load_events(FILES_TO_LOAD)
    if events_data:
        print("\n--- Starting Application ---")
        app = App(events_data)
        app.mainloop()
    else:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Startup Error",
            "Critical error: Could not load event data files. The application cannot continue."
        )
        root.destroy()