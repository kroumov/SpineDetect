FROM python:3.14

WORKDIR /usr/local/app

COPY package/requirements.txt ./
RUN pip install -r requirements.txt

COPY get_all_cts.py ./
COPY memoize_svx.py ./app.py

CMD [ "python", "app.py", "--help" ]