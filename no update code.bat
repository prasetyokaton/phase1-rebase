@echo off
cd /d "%~dp0"


echo Membangun ulang Docker (tanpa cache)...
"C:\Program Files\Git\bin\bash.exe" -c "docker compose build --no-cache"

echo Menjalankan Docker Compose...
"C:\Program Files\Git\bin\bash.exe" -c "docker compose up -d"

echo.
echo sudah selesai
pause
