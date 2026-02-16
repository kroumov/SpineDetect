FROM python:3.14

WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./app ./

CMD [ "python", "app.py", "--help" ]