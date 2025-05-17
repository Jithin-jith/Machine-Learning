# Use official sklearn base image
FROM python

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade --ignore-installed -r requirements.txt

# Expose the port
EXPOSE 5000

# Start the server
CMD ["python", "app.py"]