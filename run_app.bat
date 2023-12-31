@echo off
echo Installing virtual environment...
python -m venv venv

echo Activating virtual environment...
venv\Scripts\activate

echo Installing dependencies...
pip install Flask scikit-learn pandas

echo Running the application...
python app.py

echo Deactivating virtual environment...
deactivate
