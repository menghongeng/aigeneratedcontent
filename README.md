AI-Generated Content Detection
Note: This guide is designed based on Windows system. If you are using macOS, the basic process is similar; however, you can skip the step of setting up the System Environment Path.

I. How to Install Python
Step 1: Download Python
1. Go to the Python official website: https://www.python.org/downloads/
2. Click the Download button, and wait for the download to complete.
Step 2: Install Python
1. Run the installer.
2. Make sure you have selected the Add python.exe to PATH option.
3. Click Install Now, and wait for the progress to finish.
4. Then you can close the interface.
Step 3: Check if the Python is Successfully Downloaded
1. Press Win + R on your keyboard to open the Run dialog.
2. Type cmd and press Enter. This will open the Command Prompt.
3. In the Command Prompt, type:
python -V
4. Press Enter. You should see a message that shows the installed Python version, such as:
Python 3.x.x

II. How to Install VS Code
Step 1: Download VS Code
1. Go to the VS Code official website: https://code.visualstudio.com/, and click the download button.
2. Wait until it is downloaded, then open the downloaded file and double click to open.
3. Choose your installation folder then click Next.
4. Keep clicking Next.
5. We recommend you select all the options, then click Next.
6. Click the Install button.
7. Wait for it to be finished.

III. How to setup Conda Environment
Step 1: Create Conda Environment
Open Anaconda prompt
conda create -n aidetect python=3.11
enter y when asked for confirmation
Step 2: Activate the Environment
Enter:
conda activate aidetect
You should now see (aidetect) in the terminal.

Step 3: Install the Required Packages
Anaconda is an open-source distribution designed for scientific computing,
data analysis, and machine learning. It includes Python, the Conda package
manager, and commonly used libraries and tools.

Common Conda Commands
Create a new environment:
conda create --name aidetect python=3.11
When asked for confirmation, enter:
y
Activate the environment:
conda activate aidetect
You should now see (aidetect) in the terminal.
Check the available Python environments:
conda env list
The aidetect environment should appear in the list. When the environment
is active, (aidetect) should also appear in the terminal.

To Deactivate the current environment
conda deactivate

Install the Project Packages
After activating the aidetect environment, make sure you are in the
project folder and run:
pip install -r requirements.txt
The requirements.txt file contains:
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
joblib>=1.3
openpyxl>=3.1

Or if this doesn't work you can install it manually by the following commands:
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install joblib
pip install openpyxl

Check the Python Version
Enter:
python --version
The project uses Python 3.11.

IV. How to Setup the Project in VS Code
1. Open the VS Code we just installed.
2. Open the aigeneratedcontent project folder.
3. Select the aidetect Conda environment as the Python interpreter.
4. Open the VS Code terminal.
Make sure the following dataset files are inside the datasets folder:
base_dataset.csv
secondary_dataset.xlsx
comparison_dataset.jsonl

The project structure should include:
aigeneratedcontent/
├── datasets/
│   ├── base_dataset.csv
│   ├── secondary_dataset.xlsx
│   └── comparison_dataset.jsonl
├── datatransformation.py
├── features.py
├── model_selection.py
├── model_evaluation.py
├── config.py
└── requirements.txt


V. How to Process the Dataset
Run:
python datatransformation.py
The script processes the three datasets and prepares the training and testing data.
The main output files are:
datasets/combined_clean.csv
datasets/train.csv
datasets/test.csv
The project uses:
0 = Human-written
1 = AI-generated

VI. How to Train and Select the Model
Run:
python model_selection.py

The script extracts numerical features using features.py, trains and compares the models, performs cross-validation and tuning, selects the final model, and produces the model-selection outputs.
The trained model is saved as:
datasets/ai_detector.joblib
The feature datasets are also generated:
datasets/train_features.csv
datasets/test_features.csv

ai_detector.joblib is then trained on the next code.
Run:
python model_evaluation.py

VII. How to Evaluate the Model
After running python model_selection.py in the terminal they will be comparisons of models
The evaluation includes:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Source-level accuracy
- Error analysis
- Unseen AI source test
- Unseen prompt/topic test

VIII. Complete Command Sequence
After Python, VS Code, and the Conda environment have been set up, run:
conda activate aidetect
pip install -r requirements.txt
python datatransformation.py
python model_selection.py
python model_evaluation.py

The project workflow is:
Raw datasets
     ↓
datatransformation.py
     ↓
train.csv + test.csv
     ↓
features.py
     ↓
model_selection.py
     ↓
ai_detector.joblib
     ↓
model_evaluation.py
     ↓
evaluation_outputs
