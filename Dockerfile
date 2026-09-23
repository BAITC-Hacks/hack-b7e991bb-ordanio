# Образ веб-приложения прогноза выработки ВЭС: FastAPI на порту 8000, данные и кэш погоды внутри образа.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "web.app"]
