FROM python:3.6
COPY . /app
WORKDIR /app
ENV LISTEN_PORT=443
EXPOSE 443
RUN apt-get install -y --allow-unauthenticated libsm6
RUN apt-get install -y --allow-unauthenticated libxrender1
RUN apt-get install -y --allow-unauthenticated libfontconfig1
RUN apt-get install -y --allow-unauthenticated libice6
RUN pip install -r requirements.txt
CMD ["waitress-serve", "--listen=0.0.0.0:443", "obj_detector:app"]