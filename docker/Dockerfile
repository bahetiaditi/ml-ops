FROM python:3.9.17
WORKDIR /digits
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
ENV FLASK_APP=api/app
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0","--port=80"]