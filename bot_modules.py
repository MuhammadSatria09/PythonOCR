import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab, Image
import re
import pyautogui
import time
import os

# ==============================================================================
# --- UTILITY AND VISION FUNCTIONS (SHARED) ---
# ==============================================================================

def capture_screen(region=None):
    """Capture screen content within specified region."""
    screen = ImageGrab.grab(bbox=region)
    return cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

def load_template(template_path):
    """Load a template image from a file path."""
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"Template image not found at {template_path}")
    return template

# ==============================================================================
# --- FARMING BOT MODULES (from FarmTess) ---
# ==============================================================================

def find_items_on_screen(screen, template, threshold=0.8):
    """Find all instances of a template on the screen."""
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    rectangles = []
    for pt in zip(*loc[::-1]):
        rectangles.append([pt[0], pt[1], w, h])
    if len(rectangles) > 0:
        rectangles, _ = cv2.groupRectangles(rectangles, 1, 0.1)
    return rectangles

# In bot_modules.py
import cv2
import pytesseract

def extract_item_text_tesseract(screen, rectangles, scale_factor=2.0, char_whitelist=None, preprocessing_cfg=None):
    """
    Extract text from specified rectangles using Tesseract, with a configurable preprocessing pipeline.
    """
    if not hasattr(rectangles, 'any') or not rectangles.any(): 
        return []

    if preprocessing_cfg is None:
        preprocessing_cfg = {} # Default to empty dict if not provided

    # --- Tesseract Configuration ---
    base_config = r'--oem 3 --psm 6'
    if char_whitelist and char_whitelist.strip():
        tesseract_config = f"{base_config} -c tessedit_char_whitelist='{char_whitelist}'"
    else:
        tesseract_config = base_config

    # --- Preprocessing Configuration ---
    interpolation_map = {
        "CUBIC": cv2.INTER_CUBIC,
        "LINEAR": cv2.INTER_LINEAR,
        "LANCZOS4": cv2.INTER_LANCZOS4
    }
    interpolation_method = interpolation_map.get(preprocessing_cfg.get("resizing_interpolation", "CUBIC"), cv2.INTER_CUBIC)

    extracted_texts = []
    for i, rect in enumerate(rectangles):
        x, y, w, h = rect
        padding = 5
        x_pad, y_pad = max(0, x - padding), max(0, y - padding)
        w_pad, h_pad = min(screen.shape[1] - x_pad, w + 2 * padding), min(screen.shape[0] - y_pad, h + 2 * padding)
        item_crop = screen[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        
        if item_crop.shape[0] > 0 and item_crop.shape[1] > 0:
            # 1. Upscaling
            item_crop_scaled = cv2.resize(item_crop, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=interpolation_method)
            
            # 2. Grayscaling
            gray = cv2.cvtColor(item_crop_scaled, cv2.COLOR_BGR2GRAY)

            # 3. Binarization (Thresholding)
            thresh_method_name = preprocessing_cfg.get("thresholding_method", "OTSU")
            use_inversion = preprocessing_cfg.get("threshold_inversion", True)
            
            threshold_type = cv2.THRESH_BINARY_INV if use_inversion else cv2.THRESH_BINARY

            if thresh_method_name == "ADAPTIVE":
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, 11, 2)
            else: # Default to OTSU
                threshold_type |= cv2.THRESH_OTSU
                _, processed = cv2.threshold(gray, 0, 255, threshold_type)

            # 4. Save a debug image of the final processed result if enabled
            if preprocessing_cfg.get("save_debug_image", False):
                # Ensure the debug folder exists
                if not os.path.exists('debug_ocr'):
                    os.makedirs('debug_ocr')
                cv2.imwrite(f'debug_ocr/processed_image_{i}.png', processed)

            # 5. OCR
            text = pytesseract.image_to_string(processed, config=tesseract_config)
            extracted_texts.append(text)
        else:
            extracted_texts.append('')
            
    return extracted_texts
def correct_ocr_errors(text, corrections):
    """Correct common OCR errors based on a provided dictionary."""
    text = text.replace('\n', ' ').replace('\f', '').strip()
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

def parse_distance(text):
    """Parse distance (e.g., '11m') from text."""
    match = re.search(r'(\d+)\s*m', text)
    return int(match.group(1)) if match else 999

# In bot_modules.py

def find_and_attack_closest(items_data, config, region_offset=(0, 0)):
    """
    Finds the closest target, performs attack actions, and returns True if an
    attack was made, otherwise False.
    """
    if not items_data:
        return False # No data, no attack
    
    closest_item = min(items_data, key=lambda item: item['distance'], default=None)

    if closest_item and closest_item['distance'] != 999:
        print(f">>> FARM: Closest target: '{closest_item['text']}' at {closest_item['distance']}m. Attacking...")
        x, y, w, h = closest_item['rect']
        offset_x, offset_y = region_offset
        absolute_click_x = offset_x + x + (w // 2)
        absolute_click_y = offset_y + y + (h // 2)
        
        pyautogui.moveTo(absolute_click_x, absolute_click_y, duration=0.1)
        pyautogui.click()
        time.sleep(0.2)
        pyautogui.press(config['attack_key'])
        
        time.sleep(0.5)
        pyautogui.press(config['targeting_key'])
        
        return True # Attack was successful
    else:
        print(">>> FARM: No valid targets with distance found.")
        return False # No attack was made
# ==============================================================================
# --- MANA CHECKER MODULES (from CheckMana) ---
# ==============================================================================

def find_image_on_screen(screen, template, threshold=0.8):
    """A simplified template matching function that returns the location of the best match if above threshold."""
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        return max_loc
    return None