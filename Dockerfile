FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# expose container port (Render will set $PORT)
ENV PORT 5000
EXPOSE 5000

# Use gunicorn and bind to the PORT environment variable
# Use shell form so $PORT is expanded
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2
