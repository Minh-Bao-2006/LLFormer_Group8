@echo off
chcp 65001 >nul
echo ===================================================
echo     DANG TAT MAY CHU LLFORMER WEB (PORT 8000)...
echo ===================================================

:: Tim va tat tien trinh dang chay o port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Tim thay tien trinh PID: %%a tren cong 8000. Dang tat...
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo ===================================================
echo     MAY CHU DA DUOC TAT HOAN TOAN!
echo ===================================================
timeout /t 3 >nul

