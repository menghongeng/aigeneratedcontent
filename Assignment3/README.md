# AI Text Checker

This project has a front end and a back end. The front end is where you enter the text, and the back end uses the AI model to check it.

## What you need

- Python 3.11
- Node.js 18 or newer
- Git LFS, to download the model file

## First, download the project files

If you have not cloned the repository yet, install Git LFS and then clone it. After opening a terminal in the project folder, run:

```powershell
git lfs install
git lfs pull
```

This downloads the model file needed by the back end.

## Start the back end

Open a terminal in the main project folder (Assignment3) and run:

```
powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd Assignment3\backend
python -m uvicorn main:app --reload --port 5000
```

Leave this terminal open while you use the app.

## Start the front end

Open a second terminal in the `Assignment3\frontend` folder. Copy the example settings file and install the front end packages:

```powershell
Copy-Item .env.example .env
npm install
```

Then start the front end:

```powershell
npm start
```

The app should open at [http://localhost:3000](http://localhost:3000). Keep both terminals open while you use it. If port 3000 is already in use, stop the other app using it and try again.

## Close the app

When you are finished, press `Ctrl+C` in each terminal to stop the front end and back end. You can then close both terminals.
