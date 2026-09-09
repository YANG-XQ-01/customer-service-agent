FROM python:3.13-slim

WORKDIR /app

# 先装依赖再拷代码：依赖层可缓存，改代码不用重装包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY static ./static

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

