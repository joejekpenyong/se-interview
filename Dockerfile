# Use Python 3.12 slim as base image
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first
COPY pyproject.toml poetry.lock ./

# Configure Poetry to not create a virtual environment inside container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --without dev

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the server
CMD ["poetry", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]