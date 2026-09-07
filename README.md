## Setup Environment

Disarankan menggunakan virtual environment agar dependensi terisolasi.

python -m venv venv

venv\Scripts\activate      # Windows

source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

## Run steamlit app

cd dashboard

streamlit run dashboard.py
