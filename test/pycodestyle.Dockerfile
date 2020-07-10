FROM python:3.6
WORKDIR /tmp/
ARG workDir 
COPY ${workDir} .
RUN pip3 install --upgrade pip && \
  pip3 install pycodestyle
CMD ["pycodestyle","--max-line-length=120", "."]
