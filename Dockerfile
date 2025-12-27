# Use the official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY titanic_ml/requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . /app/

# Expose Streamlit default port
EXPOSE 8501

# Expose Kafka default port (if needed for local broker)
EXPOSE 9092

# Default command: run main.py first, then Streamlit app
CMD python titanic_ml/main.py && streamlit run titanic_ml/streamlit_app.py
