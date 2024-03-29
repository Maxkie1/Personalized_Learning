<h1 align="center">Welcome to Personalized Learning :mortar_board:</h1>

> The Personalized Learning project involves retrieving user activity logs from Moodle, predicting their learning style using a machine learning model based on the Felder-Silverman model and assigning them to a corresponding learning style group with access to specific course activities.
  
# Description

The Personalized Learning project is about retrieving user activity logs from Moodle, aggregating them and predicting the learning style of an user. The learning style is then used to assign the user to a learning style group in Moodle. Each learning style group has access to a different set of course activities. The prediction is done by a machine learning model that is trained on synthetic data. For this project the [DHBW Stutgart Moodle 4.x Dev](http://ai-in-education.dhbw-stuttgart.de/moodle/login/?lang=en) instance is used. The [Felder-Silverman learning style model](https://www.engr.ncsu.edu/wp-content/uploads/drive/1QP6kBI1iQmpQbTXL-08HSl0PwJ5BYnZW/1988-LS-plus-note.pdf) is the theoretical foundation of the application.

# Usage

The application can be run in two modes: [Local](#local-usage) and [Docker](#docker-usage). In the Local mode the application can be run from the command line. In the Docker mode the application can be run in a Docker container. Each mode offers a different set of features.

## Local Usage

In the Local mode the application can be run from the command line. For instructions on how to install the application locally, see the [Local Installation](#local-installation) section.

The application can be run with the following arguments:   
```
--train: Train the machine learning model on a synthetic dataset.
--aggregate: Aggregate activity logs from Moodle. Requires a course ID.
--predict: Predict learning style of an user. Requires a course ID and at least one prior --aggregate run.
--assign: Assign an user to learning style group in Moodle. Requires --predict.
--poll: Periodically poll Moodle for user's course completion status and trigger learning style prediction pipeline. Requires a course ID.
--mark: Mark a user's course as completed in Moodle. Requires a course ID.
--help: Show help message.
```

### Examples

Here are some examples of how to run the application from the command line:

* To train the machine learning model, run the following command:
    ```
    python main.py --train
    ```

* To aggregate activity logs from Moodle, predict learning style of an user and assign the user to a learning style group in Moodle, run the following command:
    ``` 
    python main.py --aggregate 4 --predict 4 --assign
    ```

* To periodically poll Moodle for user's course completion status and trigger learning style prediction, run the following command:
    ```
    python main.py --poll 4
    ```

* To mark an user's course as completed in Moodle, run the following command:
    ```
    python main.py --mark 4
    ```

## Docker Usage

In the Docker mode the application exclusively runs with the `--poll` argument which is automatically executed when the Docker container is started. For instructions on how to install the application in a Docker container, see the [Docker Installation](#docker-installation) section.

To start the Docker container, run the following command:
```
docker run personalized_learning
```

# Installation

This application can be installed in two ways: [Local](#local-installation) and [Docker](#docker-installation). The Local installation installs the application on your machine. The Docker installation installs the application in a Docker container.

## Requirements

The application is based on Python 3.10.  
The following Python libraries and their dependencies are required to run the application:

- `pandas`
- `pangres`
- `beautifulsoup4`
- `requests`
- `python-dotenv`
- `sqlalchemy`
- `scikit-learn`
- `sdv`
- `torch`

## Local Installation

**Follow these instructions:**

1. To run the application locally, you need to have Python 3.10 installed on your machine. To install Python 3.10, follow the instructions on the [Python website](https://www.python.org/downloads/).

2. Place a `.env` file in the root directory of the project. Write a message to [Max Kiefer](https://github.com/Maxkie1) to get the contents of the `.env` file.

3. Install the required libraries.  

    To install the required libraries globally, run the following command (**unrecommended**):
    ```
    pip install -r requirements.txt
    ```
    To install the required libraries in a conda environment, run the following command (**recommended**):

    ```
    conda create --name <thisproject>
    conda activate <thisproject>
    pip install -r requirements.txt
    ```

## Docker Installation

**Follow these instructions:**

1. To run the application in a Docker container, you need to have Docker installed on your machine. To install Docker, follow the instructions on the [Docker website](https://docs.docker.com/get-docker/). 

2. Place a `.env` file in the root directory of the project. Write a message to [Max Kiefer](https://github.com/Maxkie1) to get the contents of the `.env` file.

3. To build the Docker image, run the following command:
    ```
    docker build -t personalized_learning .
    ```

# Contributors

:bust_in_silhouette: **[Max Kiefer](https://github.com/Maxkie1)**