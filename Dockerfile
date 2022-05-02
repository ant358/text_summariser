# get the right version of python debian
FROM python:3.9
LABEL maintainer="Anthony Wynne <a.wynne@svgc.com>"
LABEL description="Text summariser with pytorch and T5 \
                  with access to shared data in network drive."
LABEL version="0.1"

# set the working directory
WORKDIR /user/src
# copy requirements file to the container
COPY requirements.txt ./
# copy the local src /user/src directory
COPY ./src .
# copy over the test text data
COPY ./text_data /usr/text_data
# install the requirements
RUN pip install --no-cache-dir -r requirements.txt
# expose the port
EXPOSE 8888:80
# add external volume
VOLUME ../text_data
