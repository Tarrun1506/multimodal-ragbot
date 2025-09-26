# Install Tesseract OCR for Windows

Your RAG bot can work without Tesseract (using EasyOCR only), but installing Tesseract provides better OCR accuracy as a fallback option.

## Quick Installation (Windows)

### Option 1: Download Installer (Recommended)
1. **Download**: Go to https://github.com/UB-Mannheim/tesseract/wiki
2. **Install**: Run the installer (choose default location: `C:\Program Files\Tesseract-OCR`)
3. **Add to PATH**: 
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click "Environment Variables"
   - Under "System Variables", find "Path", click "Edit"
   - Click "New" and add: `C:\Program Files\Tesseract-OCR`
   - Click "OK" on all dialogs
4. **Restart**: Restart your terminal/IDE

### Option 2: Using Chocolatey
```bash
choco install tesseract
```

### Option 3: Using Scoop
```bash
scoop install tesseract
```

## Verify Installation

Open a new terminal and run:
```bash
tesseract --version
```

You should see something like:
```
tesseract 5.3.0
```

## Test Your RAG Bot

After installing Tesseract, restart your backend:
```bash
python app.py
```

You should see:
```
✅ EasyOCR initialized successfully
🚀 StarRAG Bot API starting...
```

Now upload an image - both EasyOCR and Tesseract will be available for text extraction!

## Note

**Your RAG bot works without Tesseract!** EasyOCR is the primary OCR engine and works great on its own. Tesseract is just a backup option for better accuracy.
