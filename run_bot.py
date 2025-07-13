import json
import time
import threading
import sys
import os

# Third-party libraries
import pyautogui
import pytesseract
from pynput import keyboard

# Local modules
import bot_modules as bot

# --- GLOBAL CONTROL FLAGS ---
# Use threading.Event for thread-safe pause/resume and stop signals
paused_event = threading.Event()
stop_event = threading.Event()
action_lock = threading.Event() # NEW: For preventing actions during mana recovery

# --- CONFIGURATION LOADING ---
def load_config(config_path='config.json'):
    """Loads the configuration from a JSON file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode '{config_path}'. Please check for syntax errors.")
            sys.exit(1)

# --- KEYBOARD LISTENER FOR PAUSE/RESUME ---
def on_press(key):
    """Callback function for keyboard listener."""
    global PAUSE_KEY
    try:
        key_name = key.name
    except AttributeError:
        key_name = key.char

    if key_name == PAUSE_KEY:
        if paused_event.is_set():
            print("\n[CONTROL] Script Resumed.", flush=True)
            paused_event.clear()
        else:
            print("\n[CONTROL] Script Paused. Press {} again to resume.".format(PAUSE_KEY.upper()), flush=True)
            paused_event.set()

# --- BOT LOGIC THREADS ---

# In run_bot.py

def farming_bot_thread(config):
    """The main loop for the farming bot logic with cooldown management."""
    print("[FARMING BOT] Thread started.")
    cfg = config['farming_bot']
    
    try:
        monster_template = bot.load_template(cfg['monster_template_path'])
    except ValueError as e:
        print(f"[FARMING BOT] FATAL ERROR: {e}. Thread is stopping.")
        return

    cycle_count = 0
    while not stop_event.is_set():
        try:
            # --- Standard state checks ---
            if paused_event.is_set():
                time.sleep(1)
                continue
            
            if action_lock.is_set():
                print("[FARMING BOT] Paused. Waiting for mana recovery to finish...", end='\r')
                time.sleep(1)
                continue

            # --- Main logic with processing timer ---
            cycle_count += 1
            print(f"\n--- FARMING BOT Cycle {cycle_count}: Searching for target ---")
            
            # Start timing the processing part of the cycle
            process_start_time = time.time()

            region_tuple = tuple(cfg['screen_region']) if cfg['screen_region'] else None
            screen = bot.capture_screen(region=region_tuple)
            items = bot.find_items_on_screen(screen, monster_template, cfg['match_threshold'])
            
            attacked = False
            if len(items) > 0:
                # --- THIS IS THE LINE TO CHANGE ---
                raw_texts = bot.extract_item_text_tesseract(
                    screen, 
                    items, 
                    scale_factor=cfg['ocr_scale_factor'], 
                    char_whitelist=cfg.get('tesseract_char_whitelist'),
                    preprocessing_cfg=cfg.get('preprocessing') # Pass the entire preprocessing block
                )
                # ------------------------------------

                items_data = []
                for i, rect in enumerate(items):
                    corrected_text = bot.correct_ocr_errors(raw_texts[i], cfg['ocr_corrections'])
                    distance = bot.parse_distance(corrected_text)
                    items_data.append({'rect': rect, 'text': corrected_text, 'distance': distance})
                
                region_offset = (region_tuple[0], region_tuple[1]) if region_tuple else (0, 0)
                # find_and_attack_closest now returns True if it attacked
                attacked = bot.find_and_attack_closest(items_data, cfg, region_offset=region_offset)
            
            # --- Cooldown logic ---
            if attacked:
                # Stop the timer and calculate how long processing took
                process_duration = time.time() - process_start_time
                cooldown = cfg.get('skill_cooldown_seconds', 5.0) # Default to 5s if not in config
                
                # Calculate the remaining time we need to wait
                remaining_wait_time = cooldown - process_duration
                
                print(f"[FARMING BOT] Attack complete. Processing took {process_duration:.2f}s.")
                
                if remaining_wait_time > 0:
                    print(f"[FARMING BOT] Waiting for remaining cooldown: {remaining_wait_time:.2f}s...")
                    time.sleep(remaining_wait_time)
                else:
                    print("[FARMING BOT] Processing took longer than cooldown. Continuing immediately.")
            else:
                # This happens if no items were found or no valid target was attackable
                print("[FARMING BOT] No target attacked. Pressing targeting key to find new targets.")
                pyautogui.press(cfg['targeting_key'])
                time.sleep(1.5) # A standard pause when no targets are found

        except Exception as e:
            print(f"[FARMING BOT] An error occurred: {e}. Restarting cycle after a pause.")
            time.sleep(5)

    print("[FARMING BOT] Thread stopped.")


def mana_checker_thread(config):
    """The main loop for the mana checking logic."""
    print("[MANA CHECKER] Thread started.")
    cfg = config['mana_checker']
    
    try:
        mana_template = bot.load_template(cfg['mana_template_path'])
    except ValueError as e:
        print(f"[MANA CHECKER] FATAL ERROR: {e}. Thread is stopping.")
        return

    while not stop_event.is_set():
        try:
            # Block if paused
            if paused_event.is_set():
                time.sleep(1)
                continue

            screen = bot.capture_screen()
            match_location = bot.find_image_on_screen(screen, mana_template, cfg['confidence_threshold'])
            
            # Only trigger if the "out of mana" is found AND we are not already in a recovery sequence
            if match_location and not action_lock.is_set():
                print(f"\n[MANA CHECKER] Out of mana detected! Locking actions and starting recovery...")
                
                # Lock all other bot actions
                action_lock.set() 
                
                # --- Recovery sequence ---
                time.sleep(cfg['delay_after_detection_seconds'])
                pyautogui.press(cfg['mana_pot_key'])
                
                print(f"[MANA CHECKER] Used '{cfg['mana_pot_key']}'. Waiting for second skill...")
                time.sleep(cfg['delay_after_pot_seconds'])
                pyautogui.press(cfg['second_recovery_key'])
                time.sleep(0.5)
                pyautogui.press(cfg['second_recovery_key'])
                time.sleep(0.5)
                
                print(f"[MANA CHECKER] Used '{cfg['second_recovery_key']}'. Recovery complete. Releasing action lock.")
                
                # Release the lock so other bots can resume
                action_lock.clear()

                # A final small pause to let the game state settle
                time.sleep(1.0) 
            
            elif not action_lock.is_set():
                # Only print "Mana OK" if we are not in a recovery sequence
                print("[MANA CHECKER] Mana OK.", end='\r')
            
            time.sleep(cfg['check_interval_seconds'])
        
        except Exception as e:
            print(f"[MANA CHECKER] An error occurred: {e}. Restarting cycle after a pause.")
            time.sleep(5)

    print("[MANA CHECKER] Thread stopped.")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    config = load_config()

    # Apply global settings
    g_cfg = config['global_settings']
    try:
        if not os.path.exists(g_cfg['tesseract_path']):
             print(f"ERROR: Tesseract executable not found at '{g_cfg['tesseract_path']}'.")
             print("Please update the path in config.json.")
             sys.exit(1)
        pytesseract.pytesseract.tesseract_cmd = g_cfg['tesseract_path']
        pyautogui.PAUSE = g_cfg['pyautogui_pause']
        pyautogui.FAILSAFE = True
        PAUSE_KEY = g_cfg['pause_resume_key']
    except Exception as e:
        print(f"Error applying global settings: {e}")
        sys.exit(1)
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("="*50)
    print("Bot starting in 3 seconds...")
    print(f"Press '{PAUSE_KEY.upper()}' to pause or resume the script.")
    print("Press CTRL+C in this console to stop the bot completely.")
    print("="*50)
    time.sleep(3)
    
    paused_event.clear()

    threads = []
    if g_cfg.get('enable_farming_bot', False):
        farm_thread = threading.Thread(target=farming_bot_thread, args=(config,), daemon=True)
        threads.append(farm_thread)
        farm_thread.start()
        
    if g_cfg.get('enable_mana_checker', False):
        mana_thread = threading.Thread(target=mana_checker_thread, args=(config,), daemon=True)
        threads.append(mana_thread)
        mana_thread.start()

    if not threads:
        print("Both bots are disabled in config.json. Nothing to run. Exiting.")
        listener.stop()
        sys.exit(0)

    try:
        while True:
            if not any(t.is_alive() for t in threads):
                print("[CONTROL] All bot threads have stopped. Exiting.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[CONTROL] CTRL+C detected. Shutting down all threads...")
    finally:
        stop_event.set()
        if paused_event.is_set():
            paused_event.clear() 
        listener.stop()
        print("[CONTROL] Shutdown complete.")