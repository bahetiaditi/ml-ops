FROM python:3.9.17
COPY . /digits/
RUN pip3 install --no-cache-dir -r /digits/requirements.txt
WORKDIR /digits
CMD ["python3","digits.py","--runs=1","--test_size=0.2","--dev_size=0.3","--models=svm,tree"]