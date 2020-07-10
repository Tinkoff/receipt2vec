FROM python:3.6
WORKDIR /tmp/
ARG workDir 
COPY ${workDir} .
RUN pip3 install --upgrade pip && \
  pip3 install flake8
CMD ["flake8", "--ignore", "E501", "."]
