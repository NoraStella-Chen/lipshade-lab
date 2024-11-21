FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all necessary files
COPY . .

# Expose port and set environment
EXPOSE 8000
ENV STREAMLIT_SERVER_PORT=8000
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set startup command
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
