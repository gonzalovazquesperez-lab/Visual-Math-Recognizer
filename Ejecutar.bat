@echo off
title Visual Math Recognizer
echo ===================================================
echo      Iniciando Visual Math Recognizer
echo ===================================================
echo.
echo Verificando e instalando librerias necesarias...
echo (Esto puede tomar un minuto la primera vez)
echo.
pip install -r requirements.txt
echo.
echo Todo listo. Abriendo el sistema...
echo.
python main.py
pause