import json
import time
import mss
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os


REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

TEMPLATE_PATH = "scan.png"
REF_ROI_OFFSET_X = 0
REF_ROI_OFFSET_Y = 0
REF_ROI_WIDTH = 393
REF_ROI_HEIGHT = 86

def capture_and_ocr_screen_with_template(template_path, ref_roi_offset_x, ref_roi_offset_y, ref_roi_width, ref_roi_height, tesseract_path=None):
    if tesseract_path:
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            print(f"Error: Tesseract executable not found at '{tesseract_path}'")
            return ""

    if not os.path.exists(template_path):
        print(f"Error: Template image not found at '{template_path}'.")
        return ""

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        current_width = monitor["width"]
        current_height = monitor["height"]

        scale_x = current_width / REFERENCE_WIDTH
        scale_y = current_height / REFERENCE_HEIGHT

        sct_img = sct.grab(monitor)
        img_pil = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        screenshot_np = np.array(img_pil)
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template image from '{template_path}'. Check path and image format.")
        return ""

    template_scaled_width = int(template.shape[1] * scale_x)
    template_scaled_height = int(template.shape[0] * scale_y)

    template_scaled_width = max(1, template_scaled_width)
    template_scaled_height = max(1, template_scaled_height)

    template_scaled = cv2.resize(template, (template_scaled_width, template_scaled_height), interpolation=cv2.INTER_AREA)

    w, h = template_scaled.shape[::-1]

    res = cv2.matchTemplate(screenshot_gray, template_scaled, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    threshold = 0.7 # Adjust this threshold based on your template and desired accuracy
    if max_val >= threshold:
        top_left = max_loc

        roi_offset_x_scaled = int(ref_roi_offset_x * scale_x)
        roi_offset_y_scaled = int(ref_roi_offset_y * scale_y)
        roi_width_scaled = int(ref_roi_width * scale_x)
        roi_height_scaled = int(ref_roi_height * scale_y)

        x1 = top_left[0] + roi_offset_x_scaled
        y1 = top_left[1] + roi_offset_y_scaled
        x2 = x1 + roi_width_scaled
        y2 = y1 + roi_height_scaled

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(monitor["width"], x2)
        y2 = min(monitor["height"], y2)

        if x2 <= x1 or y2 <= y1:
            return ""

        roi_img_np = screenshot_np[y1:y2, x1:x2]
        roi_img_pil = Image.fromarray(cv2.cvtColor(roi_img_np, cv2.COLOR_RGB2BGR))

        # roi_img_pil.save("debug_roi.png")
        try:
            text = pytesseract.image_to_string(roi_img_pil, lang='eng')
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract OCR not found. Please install it and add to PATH or specify path in tesseract_path")
            return ""
        return text
    else:
        return ""

def search_event_title(event_title_to_find, data_file='gametora_support_data.json'):

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
        return None, 0
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{data_file}'.")
        return None, 0

    start_time = time.perf_counter()

    found_events = []
    for card in data:
        for event in card.get('events', []):
            if event_title_to_find.lower() in event.get('event_title', '').lower():
                found_events.append({
                    "card_name": card.get('name'),
                    "card_url": card.get('url'),
                    "event_title": event.get('event_title'),
                    "options": event.get('options')
                })

    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000 # Convert to milliseconds

    return found_events, latency

if __name__ == '__main__':
    os.system('cls')
    print("--- Event Title Search Script ---")
    # print("Position the target window. Scanning every second...")
    
    # Set to None to use system PATH, or specify path like r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_PATH = None

    while True:
        time.sleep(1)  # Scan every second
        ocr_text = capture_and_ocr_screen_with_template(
            TEMPLATE_PATH,
            REF_ROI_OFFSET_X,
            REF_ROI_OFFSET_Y,
            REF_ROI_WIDTH,
            REF_ROI_HEIGHT,
            TESSERACT_PATH
        ).strip()
        
        if not ocr_text:
            continue
            
        # Extract just the event title (last non-empty line)
        lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        if lines:
            search_query = lines[-1]
            # Remove trailing numbers if present (e.g. "Lovely Training Weather 5")
            if search_query and search_query[-1].isdigit():
                search_query = search_query[:-1].strip()
            break

    results, search_latency = search_event_title(search_query)

    if results is not None:
        if results:
            print(f"\nFound {len(results)} matching events:")
            for result in results:
                print(f"\nCard: {result['card_name']}")
                print(f"Event Title: {result['event_title']}")
                print("Options:")
                for option in result['options']:
                    print(f"  - {option['option']}: {option['result'].replace('\n', ' | ')}")
            print(f"\nSearch completed in {search_latency:.4f} ms.")
        else:
            print(f"\nNo events found matching '{search_query}'.")
            print(f"Search completed in {search_latency:.4f} ms.")
