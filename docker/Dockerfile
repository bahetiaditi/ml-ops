# Use an official Python runtime as a parent image
FROM python:3.9.17

# Set the working directory in the container
WORKDIR /digits

# Copy the current directory contents into the container at /digits
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Set environment variable
ENV FLASK_APP=api/app

# Run the Gunicorn server
CMD ["gunicorn", "-b", "0.0.0.0:80", "api.app:app"]
