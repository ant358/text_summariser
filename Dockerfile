
# TODO still bulding and not tested yet
FROM ubuntu:18.04
LABEL maintainer="Anthony Wynne <a.wynne@svgc.com>"
LABEL description="Dev environment for machine learning with pytorch and T5 \
                  with access to shared data in network drive."

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   # get python 3.9.7
                   python3.9\
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    -r requirements.txt


WORKDIR /workspace
EXPOSE 8888
VOLUME /text_data
COPY . src/

CMD ["python", "src/main.py", "/bin/bash"]