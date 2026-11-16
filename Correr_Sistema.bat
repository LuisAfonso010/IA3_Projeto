@echo off
chcp 65001 >nul
cd /d "%~dp0"
for /f "usebackq tokens=1,* delims==" %%a in (".env") do set "%%a=%%b"
echo === Sistema multiagente: cenario ataque, LLM real, Telegram real ===
python run_system.py --llm transformers --cenario ataque > resultado_execucao.txt 2>&1
type resultado_execucao.txt
echo.
echo Terminado. O resultado ficou em resultado_execucao.txt
pause
