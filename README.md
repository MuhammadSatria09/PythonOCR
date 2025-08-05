
# BijiKuda

A career support card event Helper for Uma Musume Pretty Derby (Steam).

---

## Requirements & Installation

To set up the helper on your system.

### 1. Python
Ensure you have **Python 3.8 or newer** installed. You can download it from the [official Python website](https://www.python.org/downloads/).

### 2. Tesseract OCR Engine
This is a critical dependency for reading text from the screen.
-   Download and install the 64-bit installer for Tesseract OCR from **[Tesseract Github](https://github.com/UB-Mannheim/tesseract/wiki)**.
-   During installation, **make sure to add Tesseract to your system's PATH**, or you will need to specify the full path to `tesseract.exe` in line `134` on `main.py`.

### 3. Python Libraries (Dependencies)
These are the Python packages required to run the script. Open your terminal or command prompt and run the following command:

```bash
pip install -r .\requirements.txt
```
