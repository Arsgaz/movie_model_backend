FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml requirements.lock ./
COPY src ./src
RUN python -m pip install --no-cache-dir -c requirements.lock -e ".[ml]" \
    && useradd --create-home --uid 10001 movie \
    && mkdir -p /app/data \
    && chown movie:movie /app/data

COPY alembic.ini ./
COPY migrations ./migrations
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/movie-entrypoint

USER movie
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/ready', timeout=4).close()"]
ENTRYPOINT ["movie-entrypoint"]
CMD ["python", "-m", "uvicorn", "main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
