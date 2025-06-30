@echo off
setlocal

:: ---------------- CONFIG ----------------
set ENV_NAME=env
set YOLO_FILE_ID=1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD
set YOLO_DEST=proj\weights\best.pt
set DPR_REPO=https://github.com/KaiyangZhou/deep-person-reid.git

:: ----------- Clone deep-person-reid -----------
if not exist "deep-person-reid" (
    echo Cloning deep-person-reid repo...
    git clone %DPR_REPO%
) else (
    echo deep-person-reid already exists. Skipping clone.
)

:: ----------- Create virtual environment -----------
if not exist "%ENV_NAME%" (
    echo Creating Python virtual environment '%ENV_NAME%'...
    python -m venv %ENV_NAME%
) else (
    echo Virtual environment '%ENV_NAME%' already exists.
)

:: ----------- Activate environment -----------
call %ENV_NAME%\Scripts\activate

:: ----------- Upgrade pip -----------
echo Upgrading pip...
python -m pip install --upgrade pip

:: ----------- Install root requirements -----------
if exist "requirements.txt" (
    echo Installing root requirements.txt...
    pip install -r requirements.txt
) else (
    echo No root requirements.txt found. Installing base packages...
    pip install torch torchvision opencv-python ultralytics scipy gdown
)

:: ----------- Install deep-person-reid requirements -----------
if exist "deep-person-reid\requirements.txt" (
    echo Installing deep-person-reid requirements...
    pip install -r deep-person-reid\requirements.txt
) else (
    echo No requirements.txt found in deep-person-reid. Skipping...
)

:: ----------- Install deep-person-reid as editable package -----------
cd deep-person-reid
echo Installing deep-person-reid in editable mode...
pip install -e .
cd ..

:: ----------- Download YOLO model -----------
if not exist "proj\weights" (
    mkdir proj\weights
)

where gdown >nul 2>nul
if errorlevel 1 (
    echo Installing gdown...
    pip install gdown
)

echo Downloading YOLO model to proj\weights\best.pt ...
gdown --id %YOLO_FILE_ID% -O %YOLO_DEST%

:: ----------- Done -----------
echo ----------------------------------------
echo ✅ Environment setup complete!
echo To activate manually: call %ENV_NAME%\Scripts\activate
pause
