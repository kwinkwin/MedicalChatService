### 1. Create venv and activate
python -m venv venv
.\venv\Scripts\activate

### 2. Install libraries
pip install -r requirements.txt

### 3. Add .env

### 4. Run
uvicorn app.main:app --reload --port 8000 (port is changeable)
Swagger UI: http://127.0.0.1:8000/docs
