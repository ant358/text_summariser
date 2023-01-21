FROM python:3.9-slim-bullseye
# Set the environment variables
ENV APP_HOME=/app
# Set the working directory
WORKDIR $APP_HOME
# Copy the requirements file
COPY requirements-docker.txt .
# Install the Python requirements
RUN pip install --no-cache-dir -r requirements-docker.txt
# Copy the source code - see dockerignore
COPY . /app
# Entrypoint
ENTRYPOINT ["python"]
# Run main
CMD ["main.py"]