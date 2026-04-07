# ============================================================
# Adaptive RAG · Docker 镜像
# 基础镜像：Python 3.11-slim；运行 Streamlit 应用
# ============================================================

FROM python:3.11-slim

# 避免交互式 apt 提示
ENV DEBIAN_FRONTEND=noninteractive \
    # 让 Python 日志实时输出到容器日志
    PYTHONUNBUFFERED=1 \
    # 防止 .pyc 文件写入镜像层
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 安装 OS 依赖（pymupdf 需要 libmupdf；chromadb 需要 build-essential）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt 利用层缓存，依赖未变化时跳过 pip install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 复制项目源码
COPY . .

# Streamlit 默认端口
EXPOSE 8501

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动命令：绑定 0.0.0.0 使容器外可访问
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
