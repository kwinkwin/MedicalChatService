# Sử dụng Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Copy file requirements và cài thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Mở cổng 8000
EXPOSE 8000

# Chạy Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]