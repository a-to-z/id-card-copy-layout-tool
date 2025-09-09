@echo off
REM Helper script to create a new release on Windows

if "%1"=="" (
    echo Usage: %0 ^<version^>
    echo Example: %0 1.0.0
    echo.
    echo This will:
    echo   1. Create and push a git tag v^<version^>
    echo   2. Trigger GitHub Actions to build and release
    exit /b 1
)

set VERSION=%1
set TAG=v%VERSION%

echo 🏷️  Creating release %TAG%...

REM Check if tag already exists
git rev-parse %TAG% >nul 2>&1
if %errorlevel% equ 0 (
    echo ❌ Tag %TAG% already exists!
    exit /b 1
)

REM Get current branch
for /f %%i in ('git branch --show-current') do set BRANCH=%%i
if not "%BRANCH%"=="main" if not "%BRANCH%"=="release" (
    echo ⚠️  Warning: You're on branch '%BRANCH%'. Consider switching to 'main' or 'release'.
    set /p REPLY=Continue anyway? (y/N): 
    if /i not "%REPLY%"=="y" exit /b 1
)

REM Create and push the tag
echo 📝 Creating tag %TAG%...
git tag -a "%TAG%" -m "Release %TAG%"

echo 🚀 Pushing tag to GitHub...
git push origin "%TAG%"

echo ✅ Done! GitHub Actions will now:
echo    - Build executables for Windows, macOS, and Linux
echo    - Create a GitHub release with downloadable files
echo    - Monitor progress in the Actions tab of your GitHub repo

echo.
echo 🔗 Release will be available in your GitHub repo's Releases section