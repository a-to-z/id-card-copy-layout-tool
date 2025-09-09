# 🆔 ID Card Maker - Professional Edition

A professional bulk ID card printing solution with advanced features for precise card layout and cutting guides.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🎯 **Smart Card Detection**
- Automatic ID card boundary detection using computer vision
- Multiple detection algorithms with fallback methods
- Manual crop editor with auto-detection assistance

### 🎨 **Professional Layout**
- Optimized card placement on A4 pages (portrait/landscape)
- Configurable card dimensions and spacing
- Smart orientation selection for maximum cards per page

### ✂️ **Precision Cutting Guides**
- Configurable crop marks positioned away from card edges
- Professional cutting guide marks for accurate trimming
- Adjustable mark offset and size

### 🔄 **Dual-Sided Support**
- Front and back card printing
- Automatic back rotation for duplex printing compatibility
- Size matching between front and back images

### 🖼️ **Advanced Image Processing**
- Support for multiple formats (JPEG, PNG, TIFF, WebP, BMP)
- Automatic contrast and sharpness enhancement
- Border detection and removal
- Perspective correction for skewed cards

### 🎛️ **User-Friendly Interface**
- Modern, intuitive GUI with real-time previews
- Progress tracking with status updates
- Comprehensive error handling and user feedback
- Professional styling with icons and organized sections

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

#### Option 1: Download Prebuilt Executable
1. Go to the [Releases](../../releases) page
2. Download the appropriate file for your operating system:
   - **Windows**: `id-card-maker-windows.exe`
   - **macOS**: `id-card-maker-macos.zip`
   - **Linux**: `id-card-maker-linux.tar.gz`
3. Extract (if needed) and run the application

#### Option 2: Run from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/id-card-copy-layout-tool.git
cd id-card-copy-layout-tool

# Install dependencies
pip install -r requirements.txt

# Run the application
python id_card_maker.py
```

### Usage

1. **Load Images**: Click "Browse" to select front (and optionally back) card images
2. **Adjust Crop**: Use the crop editor to precisely select card boundaries
3. **Configure Settings**: Set card dimensions, spacing, and layout options
4. **Generate PDFs**: Click "Generate PDF Files" to create print-ready documents

## 🔧 Building from Source

### Local Build
```bash
# Install build dependencies
pip install -r requirements.txt
pip install pyinstaller

# Run the build script
python build.py
```

### GitHub Actions (Automatic)
- Push to `release` branch to trigger builds for all platforms
- Create a tag starting with `v` (e.g., `v1.0.0`) to create a GitHub release

## ⚙️ Configuration

### Card Settings
- **Dimensions**: Standard ID card (85.6 × 54.0 mm) or custom sizes
- **Spacing**: Adjustable padding between cards
- **Margins**: Configurable page margins

### Layout Options
- **Aspect Ratio**: Enforce standard card proportions
- **Orientation**: Automatic portrait/landscape selection
- **Crop Marks**: Professional cutting guides

### Print Settings
- **Duplex Support**: Back rotation for proper duplex printing
- **Quality**: High-resolution output suitable for professional printing

## 📁 Project Structure

```
id-card-copy-layout-tool/
├── id_card_maker.py          # Main application
├── requirements.txt          # Python dependencies
├── build.py                  # Local build script
├── build.spec               # PyInstaller configuration
├── README.md                # This file
├── .gitignore              # Git ignore rules
├── temp/                   # Temporary files (auto-created)
└── .github/
    └── workflows/
        └── release.yml     # GitHub Actions workflow
```

## 🐛 Troubleshooting

### Common Issues

**Card Detection Not Working**
- Ensure good contrast between card and background
- Use the manual crop editor for precise selection
- Try different lighting conditions when photographing cards

**PDF Generation Fails**
- Check that output directory has write permissions
- Ensure sufficient disk space
- Verify all required dependencies are installed

**Application Won't Start**
- Check Python version (3.8+ required)
- Install missing dependencies: `pip install -r requirements.txt`
- On Linux, ensure tkinter is available: `sudo apt-get install python3-tk`

### Performance Tips

- Use moderately sized images (1-3 MB) for faster processing
- Close other applications if memory is limited
- For bulk processing, consider processing cards in smaller batches

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) for modern GUI styling
- Image processing powered by [OpenCV](https://opencv.org/) and [Pillow](https://pillow.readthedocs.io/)
- PDF generation using [ReportLab](https://www.reportlab.com/)

---

**Made with ❤️ for professional ID card printing**
