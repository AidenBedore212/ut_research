import os

import flask
from flask_restful import Resource, Api
from flask import Flask, render_template, request, redirect, url_for, jsonify, request
from flaskr import predict


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True, instance_path='/flaskr/Images')
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return "Hello, World!"

    @app.route("/", methods=["POST", "GET"])
    def index():
        if request.method == "POST":

            if request.form["indexButton"] == "Calculate Accuracy":
                number_of_neighbors = request.form["knn"]
                return redirect(url_for("display_accuracy", knn=number_of_neighbors))

            elif request.form["indexButton"] == "Display Graph":
                method_to_display = request.form["model_to_disp"]

                if (method_to_display == "PCA"):
                    return flask.render_template('index.html', model_to_disp="PCA")

                elif (method_to_display == "TSNE"):
                    return flask.render_template('index.html', model_to_disp="TSNE")

                elif (method_to_display == "PHATE"):
                    return flask.render_template('index.html', model_to_disp="PHATE")

                else:
                    return "how did you get here"
        else:
            return flask.render_template('index.html')#, my_val=4)

#https://docs.microsoft.com/en-us/azure/app-service/quickstart-python?tabs=bash
#azur tutorial

#https://www.geeksforgeeks.org/python-build-a-rest-api-using-flask/
#use a rest api for this instead

#https://www.twilio.com/blog/deploy-flask-python-app-aws

    @app.route("/show_reduc_methods")
    def show_reduc_methods():
        return "Is this thing working"


    @app.route("/display_accuracy/<knn>")
    def display_accuracy(knn):
        knn = int(knn)
        accuracyOfModel = predict.getAccuracyPCA(knn)
        return flask.render_template("disp_accuracy.html", knn=knn, model_accur=accuracyOfModel)

    api = Api(app)
    @app.route("/get_accuracy/<knn>")
    def get(knn):
        knn = int(knn)
        accuracyOfModel = predict.getAcuracyPCA(knn)
        return jsonify({'knn': knn, 'Accuracy': accuracyOfModel})


    return app