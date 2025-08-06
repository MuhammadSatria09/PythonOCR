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
import sys ### NEW: Required for the path helper

# --- NEW: Helper function to find data files in a bundled app ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- CONFIGURATION (USER MUST SET THIS UP) ---

# 1. TESSERACT PATH: 
# IMPORTANT: Uncomment and set this path. This is CRITICAL for the compiled .exe.
# The user of your .exe will need Tesseract installed in this exact location.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. TEMPLATE IMAGE: Use the resource_path helper function
TEMPLATE_PATH = resource_path("scan.png") 

# 3. JSON DATA FILES: Use the resource_path helper function
FILES_TO_LOAD = [
    resource_path("scraped_character_events.json"), 
    resource_path("scraped_events_data.json")
]

# 4. ROI and RESOLUTION:
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080
REF_ROI_OFFSET_X = 0
REF_ROI_OFFSET_Y = 0
REF_ROI_WIDTH = 393
REF_ROI_HEIGHT = 86


# --- OCR AND SCREEN CAPTURE FUNCTIONS (No changes needed here) ---

def capture_and_ocr_screen_with_template(template_path, ref_roi_offset_x, ref_roi_offset_y, ref_roi_width, ref_roi_height):
    """
    Captures a screenshot, finds a template, and performs OCR on a defined ROI.
    Returns a dictionary containing the found text and performance timings for each step.
    """
    timings = {"capture": 0, "template_matching": 0, "ocr": 0}
    
    if not os.path.exists(template_path):
        return {"text": "", "status": f"error_no_template_file at {template_path}", "timings": timings}

    # --- 1. Screen Capture ---
    start_capture = time.perf_counter()
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
    timings["capture"] = (time.perf_counter() - start_capture) * 1000

    current_width, current_height = monitor["width"], monitor["height"]
    scale_x = current_width / REFERENCE_WIDTH
    scale_y = current_height / REFERENCE_HEIGHT

    img_pil = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
    screenshot_np = np.array(img_pil)
    screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return {"text": "", "status": f"error_template_load_fail at {template_path}", "timings": timings}

    template_scaled_width = max(1, int(template.shape[1] * scale_x))
    template_scaled_height = max(1, int(template.shape[0] * scale_y))
    template_scaled = cv2.resize(template, (template_scaled_width, template_scaled_height), interpolation=cv2.INTER_AREA)

    # --- 2. Template Matching ---
    start_match = time.perf_counter()
    res = cv2.matchTemplate(screenshot_gray, template_scaled, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    timings["template_matching"] = (time.perf_counter() - start_match) * 1000

    threshold = 0.7
    if max_val >= threshold:
        top_left = max_loc
        roi_offset_x_scaled = int(ref_roi_offset_x * scale_x)
        roi_offset_y_scaled = int(ref_roi_offset_y * scale_y)
        roi_width_scaled = int(ref_roi_width * scale_x)
        roi_height_scaled = int(ref_roi_height * scale_y)

        x1, y1 = top_left[0] + roi_offset_x_scaled, top_left[1] + roi_offset_y_scaled
        x2, y2 = x1 + roi_width_scaled, y1 + roi_height_scaled
        
        if x2 <= x1 or y2 <= y1:
            return {"text": "", "status": "error_invalid_roi", "timings": timings}

        roi_img_np = screenshot_np[y1:y2, x1:x2]

        # --- 3. OCR Processing ---
        start_ocr = time.perf_counter()
        text = pytesseract.image_to_string(Image.fromarray(roi_img_np), lang='eng')
        timings["ocr"] = (time.perf_counter() - start_ocr) * 1000

        return {"text": text.strip(), "status": "success", "timings": timings}
    else:
        return {"text": "", "status": "template_not_found", "timings": timings}

# --- JSON DATA HANDLING AND DISPLAY FUNCTIONS (No changes needed here) ---

def load_events(filenames):
    """
    Loads event data from a list of JSON files and "tags" each
    entry with its source filename.
    """
    combined_data = []
    print("Loading event data...")
    for filename in filenames:
        try:
            # Get the base name for display purposes
            display_name = os.path.basename(filename)
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add the source file to each top-level object
                for item in data:
                    item['source_file'] = display_name
                combined_data.extend(data)
                print(f"  - Successfully loaded {display_name}")
        except FileNotFoundError:
            print(f"  - Warning: '{os.path.basename(filename)}' not found. Skipping.")
        except json.JSONDecodeError:
            print(f"  - Warning: '{os.path.basename(filename)}' is not a valid JSON. Skipping.")
    
    if not combined_data:
        print("\nError: Could not load any event data. Exiting.")
        return None
    print("Data loading complete.")
    return combined_data

def find_event(events_data, search_query):
    """
    Searches for an event and returns the event details, holder name,
    the original source file, and search latency.
    """
    start_search = time.perf_counter()
    normalized_query = normalize_title(search_query)
    
    if not normalized_query:
        return None, None, None, 0

    found_event, holder_name, source_file = None, None, None
    for holder in events_data:
        for event in holder.get("events", []):
            json_title = event.get("title", "")
            normalized_json_title = normalize_title(json_title)
            
            if normalized_query == normalized_json_title:
                holder_name = holder.get("event_holder_name") or holder.get("character_name") or "Unknown Holder"
                found_event = event
                source_file = holder.get("source_file") # Get the source file tag
                break
        if found_event:
            break
            
    latency_ms = (time.perf_counter() - start_search) * 1000
    return found_event, holder_name, source_file, latency_ms

def display_results(query, found_event, holder_name, source_file, timings):
    """
    Clears the console and displays search results and performance metrics
    in the new requested format.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    
    if found_event:
        if source_file == "scraped_character_events.json":
            display_header = "Trainee Event"
        else:
            display_header = f"Event found for: {holder_name}"
        
        print(f"{display_header}")
        print(f"Title: {found_event['title']}")
        print("="*60)

        for option in found_event.get('options', []):
            print(f"\n--- Choice: {option.get('choice', 'N/A')} ---")
            for effect in option.get('effects', []):
                print(f"  - {effect}")
    else:
        print(f"Searching for text from screen: '{query}'")
        print("-" * 30)
        print(f"\nNo event details found matching '{query}'.")

    performance_line = (
        f"Screen Capture: {timings['capture']:.2f} ms | "
        f"Template Matching: {timings['template_matching']:.2f} ms | \n"
        f"OCR Processing: {timings['ocr']:.2f} ms | "
        f"JSON Search: {timings['search']:.2f} ms |"
    )
    print("\n" + "="*60)
    print(performance_line)
    print("\n--- Continuously scanning... (Press Ctrl+C to exit) ---")
    
def normalize_title(title: str) -> str:
    """
    Normalizes a title for robust comparison by making it lowercase and
    removing all characters that are not letters.
    """
    if not title:
        return ""
    normalized = title.lower()
    normalized = re.sub(r'[^a-z\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

# --- MAIN EXECUTION BLOCK (No changes needed here) ---

if __name__ == '__main__':
    events_data = load_events(FILES_TO_LOAD)
    if not events_data:
        # ### NEW: Add a pause so the user can see the error in the console ###
        input("Press Enter to exit.") 
        exit()

    last_successful_query = ""
    print("\n--- Starting Automatic Scan ---")
    print("Please ensure the target window/game is visible.")
    print("Press Ctrl+C to exit at any time.")
    time.sleep(1)

    while True:
        try:
            total_cycle_start = time.perf_counter()
            ocr_result = capture_and_ocr_screen_with_template(
                TEMPLATE_PATH, REF_ROI_OFFSET_X, REF_ROI_OFFSET_Y, REF_ROI_WIDTH, REF_ROI_HEIGHT
            )
            
            cycle_timings = ocr_result["timings"]
            cycle_timings["search"] = 0 

            ocr_text = ocr_result["text"]
            current_query = ""
            
            # ### NEW: Added a check for the error status for better debugging ###
            if ocr_result["status"] != "success" and ocr_result["status"] != "template_not_found":
                 print(f"An error occurred during capture: {ocr_result['status']}")

            if ocr_text:
                lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
                if lines:
                    current_query = lines[-1]

            if current_query and current_query != last_successful_query:
                print(f"New text detected: '{current_query}'. Searching...")
                last_successful_query = current_query
                found_event, holder_name, source_file, search_latency = find_event(events_data, current_query)
                cycle_timings["search"] = search_latency
                display_results(current_query, found_event, holder_name, source_file, cycle_timings)

            time.sleep(1)

        except KeyboardInterrupt:
            print("\nScript stopped by user. Exiting.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Restarting scan in 5 seconds...")
            time.sleep(1)