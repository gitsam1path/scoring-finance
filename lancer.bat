@echo off
chcp 65001 >nul
title Scoring Finance
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Installation de l'environnement Python ^(une seule fois^)...
  python -m venv .venv || (echo Python introuvable : installez Python 3.11+ depuis python.org & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (pause & exit /b 1)
)
if not exist "models\model.joblib" (
  echo Entrainement du modele...
  pushd src & "..\.venv\Scripts\python.exe" -m scoring.train & popd
)
echo API sur http://localhost:8000  ·  documentation sur http://localhost:8000/docs
start "" http://localhost:8000
".venv\Scripts\python.exe" -m uvicorn api.main:app --port 8000
pause
