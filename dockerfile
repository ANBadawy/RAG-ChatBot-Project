# 1) Start with a minimal Python image
FROM python:3.12

# 2) Environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


# 4) Set a working directory
WORKDIR /app

# 5) Copy requirements.txt into the container
COPY requirements.txt .

# 6) Install Python dependencies
RUN pip install -r requirements.txt

# 7) Copy the rest of your project files
COPY . .

# 8) # Expose the default Streamlit port
EXPOSE 8501

# 9) Run Streamlit when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
