@echo off

echo Launching Streamlit application using venv...
call .\venv\Scripts\python.exe -m streamlit run app.py

echo Script finished.
pause