
# Python OCR & Automation Bot

This is a sophisticated Python-based automation bot designed to perform tasks in a game by visually interpreting the screen. It operates using a multi-threaded approach to handle different tasks simultaneously.

## Features

-   **Multi-threaded Operation**: The farming logic and the mana-checking logic run in parallel, allowing the bot to react to status changes while actively looking for targets.
-   **Centralized Configuration**: All settings, from keybinds to file paths, are managed in a single `config.json` file for easy customization without touching the code.
-   **Intelligent Cooldown Management**: The bot's attack cycle includes a smart timer that accounts for the processing time (finding and targeting an enemy), ensuring it respects the game's skill cooldowns efficiently.
-   **Cooperative Tasking**: An "action lock" system ensures the farming bot will automatically pause its actions while the mana bot is performing a recovery sequence, preventing conflicts.
-   **Real-time OCR**: Utilizes Tesseract OCR to read text from the screen, such as enemy names and distances, to make intelligent targeting decisions.
-   **Global Pause/Resume**: You can pause and resume all bot activity at any time by pressing a single hotkey (default: `F10`).
-   **Error Resilience**: Both core modules are wrapped in error handlers to prevent the entire script from crashing due to a single detection failure.
-   **Modular Codebase**: The project is cleanly separated into a runner script, a module library, and a configuration file, making it easy to maintain and extend.

---

## ⚠️ Important Disclaimer

**Use at your own risk.** Using bots, scripts, or any form of automation can be against the Terms of Service of many online games and may result in penalties, including temporary or permanent account suspension. The author of this script is not responsible for any actions taken against your account. You are solely responsible for understanding and complying with the rules of the game you are playing.

---

## Requirements & Installation

Follow these steps to set up the bot on your system.

### 1. Python
Ensure you have **Python 3.8 or newer** installed. You can download it from the [official Python website](https://www.python.org/downloads/).

### 2. Tesseract OCR Engine
This is a critical dependency for reading text from the screen.
-   Download and install the 64-bit installer from this direct link: **[tesseract-ocr-w64-setup-v5.5.0.20241111.exe](https://sourceforge.net/projects/tesseract-ocr.mirror/files/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe/download)**.
-   During installation, **make sure to add Tesseract to your system's PATH**, or you will need to specify the full path to `tesseract.exe` in the `config.json` file.

### 3. Python Libraries (Dependencies)
These are the Python packages required to run the script. Open your terminal or command prompt and run the following command:

```bash
pip install opencv-python pytesseract numpy Pillow pyautogui pynput
```

---

## File Structure

Your project folder should be organized as follows for the script to work correctly.

```
/your-bot-folder
├── Templates/
│   ├── Monster.png
│   └── OutOfMana.png
├── run_bot.py          # The main script to execute the bot
├── bot_modules.py      # Contains all core bot functions
├── config.json         # All user-configurable settings
└── README.md           # This documentation file
```

**Note:** The `.png` files inside the `Templates` folder are examples. You should create your own by taking screenshots from your game for the best accuracy.

---

## Configuration

All customization is done in the `config.json` file.

-   **`tesseract_path`**: **(CRITICAL)** If Tesseract is not in your system's PATH, you must provide the full path to the executable here. Example for Windows: `"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"`. Note the double backslashes `\\`.
-   **Template Paths**: Make sure the paths for your `monster_template_path` and `mana_template_path` include the `Templates/` folder prefix (e.g., `"Templates/Monster.png"`). **I Recommend to take replace this with your screenshot**
-   **`pause_resume_key`**: The keyboard key to pause and resume the bot (e.g., `"f10"`, `"p"`).
-   **`attack_key`**, **`targeting_key`**, **`mana_pot_key`**, **`second_recovery_key`**: Set these to match the keybinds you use in-game.
-   **`skill_cooldown_seconds`**: Set this to the total cooldown time of your main attack skill in seconds (e.g., `6.0`). The bot will automatically subtract its processing time from this value.
-   **`screen_region`**: Set to `null` to capture the full screen, or specify a region `[x, y, width, height]` to limit the search area and improve performance.

---

## How to Run the Bot

1.  **Install all requirements** as described above.
2.  **Create a folder named `Templates`** in your project directory.
3.  **Prepare Template Images**:
    -   Take a screenshot of a target's nameplate in the game and save it as `Monster.png` inside the `Templates` folder.
    -   Take a screenshot of the "out of mana" icon/message and save it as `OutOfMana.png` inside the `Templates` folder.
4.  **Update `config.json`**:
    -   Verify the `tesseract_path` is correct.
    -   **Crucially, update the image paths to include the `Templates/` directory.** See the example below.
    -   Set all keybinds and cooldowns to match your in-game setup.
5.  **Open a terminal** or command prompt.
6.  **Navigate** to the project directory where `run_bot.py` is located.
7.  **Run the script** with the following command:
    ```bash
    python run_bot.py
    ```
8.  The bot will start after a 3-second delay. You can now switch to your game window.

### In-Game Controls
-   **Pause/Resume**: Press the key defined in `pause_resume_key` (default: **F10**).
-   **Stop**: Switch back to the terminal window and press **Ctrl+C**. This will shut down all bot threads gracefully.

### In-Game Requirement
- **Toram Online** : Magic Skill :
    - MP Charge Lv 10
    - Maximizer Lv 10
    - Magic : Storm Lv 10
