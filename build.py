#!/usr/bin/env python3
"""
Build script for ID Card Maker
Creates standalone executables for different platforms using PyInstaller
"""

import os
import sys
import shutil
import subprocess
import platform

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"🔨 {description or cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description or 'Command'} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description or 'Command'} failed with exit code {e.returncode}")
        sys.exit(1)

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning previous build artifacts...")
    dirs_to_clean = ['build', 'dist', '__pycache__']
    files_to_clean = ['*.spec~']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Removed {dir_name}/")
    
    # Clean .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))
    
    print("✅ Cleanup completed")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    run_command("pip install -r requirements.txt", "Installing project dependencies")
    run_command("pip install pyinstaller", "Installing PyInstaller")

def create_icon():
    """Create a simple icon if none exists"""
    if not os.path.exists('icon.ico'):
        print("🎨 Creating default icon...")
        try:
            from PIL import Image, ImageDraw
            
            # Create a simple icon
            size = (64, 64)
            img = Image.new('RGBA', size, (70, 130, 180, 255))  # Steel blue background
            draw = ImageDraw.Draw(img)
            
            # Draw a simple card-like rectangle
            margin = 8
            draw.rectangle([margin, margin, size[0]-margin, size[1]-margin], 
                         fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=2)
            
            # Add some text-like lines
            for i in range(3):
                y = margin + 15 + i * 8
                draw.rectangle([margin + 5, y, size[0] - margin - 5, y + 2], 
                             fill=(100, 100, 100, 255))
            
            img.save('icon.ico', format='ICO', sizes=[(32, 32), (64, 64)])
            print("✅ Default icon created")
        except ImportError:
            print("⚠️  Could not create icon (PIL not available)")
    else:
        print("ℹ️  Icon already exists")

def build_executable():
    """Build the executable using PyInstaller"""
    system = platform.system()
    print(f"🚀 Building executable for {system}...")
    
    if system == "Windows":
        create_icon()
        cmd = """pyinstaller --clean --onefile --windowed --name "ID-Card-Maker" --icon=icon.ico --add-data "README.md;." --hidden-import=PIL._tkinter_finder --hidden-import=ttkbootstrap --collect-all ttkbootstrap id_card_maker.py"""
    elif system == "Darwin":  # macOS
        cmd = """pyinstaller --clean --onefile --windowed --name "ID-Card-Maker" --add-data "README.md:." --hidden-import=PIL._tkinter_finder --hidden-import=ttkbootstrap --collect-all ttkbootstrap id_card_maker.py"""
    else:  # Linux
        cmd = """pyinstaller --clean --onefile --name "id-card-maker" --add-data "README.md:." --hidden-import=PIL._tkinter_finder --hidden-import=ttkbootstrap --collect-all ttkbootstrap id_card_maker.py"""
    
    run_command(cmd, f"Building {system} executable")

def create_package():
    """Create a distributable package"""
    system = platform.system()
    print(f"📦 Creating package for {system}...")
    
    if not os.path.exists('dist'):
        print("❌ No dist folder found. Build may have failed.")
        return
    
    package_dir = 'package'
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)
    
    # Copy executable
    if system == "Windows":
        exe_name = "ID-Card-Maker.exe"
        if os.path.exists(f'dist/{exe_name}'):
            shutil.copy(f'dist/{exe_name}', package_dir)
        else:
            print(f"❌ Executable not found: dist/{exe_name}")
            return
    else:
        exe_name = "ID-Card-Maker" if system == "Darwin" else "id-card-maker"
        if os.path.exists(f'dist/{exe_name}'):
            shutil.copy(f'dist/{exe_name}', package_dir)
            # Make executable
            os.chmod(f'{package_dir}/{exe_name}', 0o755)
        else:
            print(f"❌ Executable not found: dist/{exe_name}")
            return
    
    # Copy documentation
    if os.path.exists('README.md'):
        shutil.copy('README.md', package_dir)
    if os.path.exists('LICENSE'):
        shutil.copy('LICENSE', package_dir)
    
    print(f"✅ Package created in {package_dir}/")
    print("\n📋 Package contents:")
    for item in os.listdir(package_dir):
        size = os.path.getsize(os.path.join(package_dir, item))
        print(f"   {item} ({size:,} bytes)")

def main():
    """Main build process"""
    print("🏗️  ID Card Maker Build Script")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('id_card_maker.py'):
        print("❌ id_card_maker.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    try:
        clean_build()
        install_dependencies()
        build_executable()
        create_package()
        
        print("\n🎉 Build completed successfully!")
        print(f"📦 Executable package is ready in the 'package/' directory")
        
        if platform.system() == "Windows":
            print("💡 You can now distribute ID-Card-Maker.exe")
        elif platform.system() == "Darwin":
            print("💡 You can now distribute ID-Card-Maker (macOS app)")
        else:
            print("💡 You can now distribute id-card-maker (Linux executable)")
            
    except KeyboardInterrupt:
        print("\n❌ Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()