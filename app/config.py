import os
from pathlib import Path
from dotenv import load_dotenv
import itertools

# 1. Xác định vị trí file .env
env_path = Path(__file__).resolve().parent.parent / ".env"

# 2. Load file .env
load_dotenv(dotenv_path=env_path)

# Debug: In ra để kiểm tra
print(f"DEBUG: Loading .env from: {env_path}")
print(f"DEBUG: NEO4J_URI read as: {os.getenv('NEO4J_URI')}")

class Settings:
    # --- Các Config khác ---
    HF_API_KEY = os.getenv("HF_API_KEY")
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
    
    # --- LOAD & GỘP KEY TỰ ĐỘNG ---
    _raw_keys = []

    # 1. Lấy Key chính (nếu có)
    if os.getenv("GOOGLE_API_KEY"):
        _raw_keys.append(os.getenv("GOOGLE_API_KEY"))

    # 2. Quét động các key từ GOOGLE_API_KEY_1 đến GOOGLE_API_KEY_50
    # (Cách này giúp bạn chỉ cần sửa file .env, không cần sửa code Python khi thêm key mới)
    for i in range(1, 60): 
        key_name = f"GOOGLE_API_KEY_{i}"
        key_val = os.getenv(key_name)
        if key_val:
            _raw_keys.append(key_val)

    # 3. Lọc trùng và bỏ key rỗng
    GOOGLE_KEYS = list(set([k for k in _raw_keys if k and isinstance(k, str) and k.strip()]))
    
    # Debug: In ra số lượng key đã load được (ẩn nội dung key để bảo mật)
    print(f"DEBUG: Loaded {len(GOOGLE_KEYS)} Google API Keys for rotation.")

    # 4. Tạo bộ xoay vòng
    _key_cycle = itertools.cycle(GOOGLE_KEYS)
    
    def get_next_google_key(self):
        """Hàm lấy key tiếp theo trong danh sách"""
        if not self.GOOGLE_KEYS:
            raise ValueError("Không tìm thấy Google API Key nào trong .env (kiểm tra lại biến GOOGLE_API_KEY_x)")
        return next(self._key_cycle)
    
    def disable_google_key(self, key):
        if key in self.GOOGLE_KEYS:
            self.GOOGLE_KEYS.remove(key)


settings = Settings()