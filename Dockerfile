# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Download model files from Google Drive
RUN curl -L "https://drive.google.com/uc?export=download&id=1IB0sLDA_qbrl01c2De-01iSfqX7AvxNV" -o chatbot_model.h5
RUN curl -L "https://drive.google.com/uc?export=download&id=1p5PVhWupSL8SSKi3hOvXaoghy0ICl4zY" -o words.pkl
RUN curl -L "https://drive.google.com/uc?export=download&id=15vxwbxxPZW_vWq_p7AWy2xbMvows7mSD" -o classes.pkl

# Copy the rest of your application code
COPY . .

# Expose the port your app runs on
EXPOSE 8080

# Command to run your app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "chatbot:app"]