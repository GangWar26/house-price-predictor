@echo off
echo Starting Flask API...
start cmd /k "cd /d C:\Users\adiga\OneDrive\Desktop\house-price-predictor\backend && python app.py"

echo Opening frontend...
timeout /t 2 >nul
start C:\Users\adiga\OneDrive\Desktop\house-price-predictor\frontend\index.html

echo Done! App is running.