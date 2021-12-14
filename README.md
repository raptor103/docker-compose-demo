**Docker-compose demo project**

This is a demo project which aims to deploy a machine-learning
containerized app together with a database.\
 This app is a simple image classification model that is served on a Flask endpoint.

There are the following components:
1) Folder template - includes HTML files which serves as UI
2) app.py - core application that serves out predictions
3) Dockerfile - creates image from app
4) MySQL database - stores history of predictions
5) Docker-compose - spins up the services

Help tips for playing around:\
Use following to restart MySQL database volume\
*docker system prune --volumes*

Re-build app\
*docker-compose build*

Run the whole thing\
*docker-compose -f docker-compose.yml up*
