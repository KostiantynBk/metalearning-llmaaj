FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir --upgrade "pip>=26.2" \
 && pip install --no-cache-dir -r requirements.txt -c constraints.txt
COPY src/ src/
COPY rulejudge/ rulejudge/
ENTRYPOINT ["python", "-m", "rulejudge"]
CMD ["--help"]
