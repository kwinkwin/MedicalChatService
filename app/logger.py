import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging():
    # --- 1. Sửa lỗi Crash Tiếng Việt trên Windows Console ---
    # Ép buộc console của Python sử dụng UTF-8 thay vì CP1252 mặc định
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    # --- 2. Xác định đường dẫn ---
    current_file_path = os.path.abspath(__file__) 
    app_dir = os.path.dirname(current_file_path)  
    root_dir = os.path.dirname(app_dir)           

    log_dir = os.path.join(root_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # --- 3. Tạo tên file theo format: dd-mm-yyyy HH-MM-SS.txt ---
    # Lưu ý: Windows KHÔNG cho phép dấu hai chấm (:) trong tên file
    # Nên format giờ phải là H-M-S (dùng gạch ngang)
    current_time = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    log_filename = f"{current_time}.txt"
    log_file_path = os.path.join(log_dir, log_filename)

    # --- 4. Cấu hình Logger ---
    # Format nội dung log
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler 1: Ghi ra File (Luôn dùng utf-8 để đọc được tiếng Việt)
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=10*1024*1024, # 10MB
        backupCount=5, 
        encoding='utf-8' # <--- QUAN TRỌNG: Để file log hiển thị đúng tiếng Việt
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Handler 2: Hiện ra Console (Đã được fix encoding ở bước 1)
    # Chúng ta dùng sys.stdout để đảm bảo nó ăn theo cấu hình reconfigure ở trên
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Setup Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    print(f"LOG FILE đã được tạo tại: {log_file_path}")

    return root_logger