# Base Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first and install (cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src folder (code + data + models)
COPY src/ ./src

# Expose the port
EXPOSE 8000

# Command to run FastAPI
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
