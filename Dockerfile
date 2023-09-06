FROM python:3.10
# -----------------------------------
# create required folder
RUN mkdir /app
RUN mkdir /app/deepface
# -----------------------------------
# Copy required files from repo into image
COPY ./deepface /app/deepface
COPY ./database /app/database
COPY ./app.py /app/
COPY pyproject.toml /app
# -----------------------------------
# switch to application directory
WORKDIR /app
# -----------------------------------
# update image os
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install
# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1
# -----------------------------------
# run the app (re-configure port if necessary)
EXPOSE 8090
ENTRYPOINT ["python"]
CMD ["app.py"]