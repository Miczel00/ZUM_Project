# ZUM_Project
# Setting up a Python Virtual Environment and Installing Dependencies

This guide will walk you through creating a Python virtual environment, activating it, and installing the required dependencies from a `requirements.txt` file.

## Prerequisites

Ensure you have the following installed on your system:

- **Python** (version 3.6 or higher recommended)
- **pip** (Python package manager, usually comes with Python)

To check if Python and pip are installed, run the following commands:

```bash
python --version
pip --version
```

If Python or pip is not installed, download and install Python from the [official Python website](https://www.python.org/downloads/).

## Steps to Set Up the Environment

### 1. Create a Virtual Environment

Run the following command to create a virtual environment named `venv` (you can replace `venv` with any name you prefer):

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

After activation, your terminal prompt should change to indicate that the virtual environment is active. For example, it might look like this:

```
(venv) $ 
```

### 3. Install Dependencies

Once the virtual environment is activated, install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install all the packages listed in `requirements.txt` into the virtual environment.

### 4. Verify Installation

To confirm that the dependencies have been installed correctly, you can list the installed packages:

```bash
pip list
```

## Additional Notes

- To deactivate the virtual environment at any time, run:

  ```bash
  deactivate
  ```

- Make sure to activate the virtual environment each time you work on this project to ensure that the installed dependencies are used.

  
