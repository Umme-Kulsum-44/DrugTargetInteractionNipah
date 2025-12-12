FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Default PORT (render will override at runtime)
ENV PORT=5000
EXPOSE 5000

# Run gunicorn and bind to the PORT env var provided by Render
# Use shell form so $PORT expands
CMD gunicorn --bind 0.0.0.0:$PORT app:app --workers 2