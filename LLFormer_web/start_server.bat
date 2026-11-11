@echo off
chcp 65001 >nul
title LLFormer Low-Light Enhancement Web App

echo.
echo ══════════════════════════════════════════════════════════
echo   LLFormer — Low-Light Image Enhancement Web App
echo ══════════════════════════════════════════════════════════
echo.
echo   Truy cap tai: http://localhost:8000
echo   API docs:     http://localhost:8000/docs
echo.
echo   Nhan Ctrl+C de dung server.
echo ══════════════════════════════════════════════════════════
echo.

cd /d "%~dp0"
python web_app\backend\server.py

pause

