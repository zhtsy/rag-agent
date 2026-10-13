
FROM python:3.12-slim

WORKDIR /app

# 安装 uv
RUN pip install --no-cache-dir uv

COPY requirements.txt .

# 用 uv 安装依赖
RUN uv pip install --system --no-cache --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main1:app", "--host", "0.0.0.0", "--port", "8000"]