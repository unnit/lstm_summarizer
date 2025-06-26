# Project Installation Instructions
**1. Clone the project from the github repo**<br/>

**2. Make sure python is installed in your machine. You can check using**<br/>
`python --version`<br/>
If Python is not installed, you can download it from the official Python website(https://www.python.org/downloads/)

**3. pip is the package installer for Python, and it should be installed by default with Python. You can check if it’s installed by running:**<br/>
`pip --version`<br/>
If pip is not installed, you can install it using:<br/>
`python -m ensurepip --upgrade`

**4. Create a virtual environment (Optional)**<br/>
`pip install virtualenv`<br/>
`python -m venv {env_name}`<br/>
Activate the environment

**5. Install dependencies**<br/>
`pip install -r requirements.txt`<br/>

**6. Run the server**<br/>
`python app.py` for development server
`gunicorn app:app` for production server

