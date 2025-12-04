import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Xác định vị trí file .env
# Logic: File config.py nằm trong app/, nên .env nằm ở thư mục cha của app (tức là root)
env_path = Path(__file__).resolve().parent.parent / ".env"

# 2. Load file .env
load_dotenv(dotenv_path=env_path)

# Debug: In ra để kiểm tra xem đã load được chưa (Xóa dòng này sau khi fix xong)
print(f"DEBUG: Loading .env from: {env_path}")
print(f"DEBUG: NEO4J_URI read as: {os.getenv('NEO4J_URI')}")

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

settings = Settings()