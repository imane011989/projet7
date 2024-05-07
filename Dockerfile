# python base image in the container from Docker Hub
FROM python:3.8-slim

# copy files to the /app folder in the container
COPY ./app.py /app/app.py
COPY ./Pipfile /app/Pipfile
COPY ./Pipfile.lock /app/Pipfile.lock
COPY ./model.pkl /app/model.pkl   

# set the working directory in the container to be /app
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y libgomp1

# install the packages from the Pipfile in the container
RUN pip install numpy
RUN pip install pandas
RUN pip install shap
RUN pip install scikit-learn
RUN pip install lightgbm
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# expose the port that uvicorn will run the app on
ENV PORT=8000
EXPOSE 8000

# execute the command python main.py (in the WORKDIR) to start the app
CMD ["python", "app.py"]