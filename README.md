<h1 align="center">Welcome to Personalized Learning :wave:</h1>

> Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum.
  
# Description

The personalized learning project is abouting retrieving student activity logs from Moodle, aggregating them and predicting the learning style of a student. The learning style is then used to assign the student to a learning style group in Moodle. Each learning style group has access to a different set of course activities. The prediction is done by a machine learning model that is trained on synthetic data. The Felder-Silverman learning style model is the theoretical foundation of the application.

# Usage

## Local

The application can be run from the command line with the following arguments:
    
```
--train: Train the machine learning model on a synthetic dataset.
--aggregate: Aggregate activity logs from Moodle. Requires a course ID.
--predict: Predict learning style of a student. Requires a course ID and at least one prior --aggregate run.
--assign: Assign student to learning style group in Moodle. Requires --predict.
--poll: Periodically poll Moodle for student's course completion status and trigger learning style prediction. Requires a course ID. This is the main element of the automated learning style prediction pipeline.
--mark: Mark a student's course as completed in Moodle. Requires a course ID.
--help: Show help message.
```

Typical usage of the application is as follows:

To train the machine learning model, aggregate activity logs from Moodle, predict learning style of a student, assign the student to a learning style group in Moodle, run the following command:
``` 
python main.py --train --aggregate 4 --predict 4 --assign
```

To periodically poll Moodle for student's course completion status and trigger learning style prediction, run the following command:
```
python main.py --poll 4
```

To mark a student's course as completed in Moodle, run the following command:
```
python main.py --mark 4
```

## Docker

In the automated polling mode the application can be run in a Docker container. 

To build the Docker image, run the following command:
```
docker build -t personalized_learning .
```

To run the Docker container, run the following command:
```
docker run personalized_learning
```


# Installation

To run the project locally, you need to install the following libraries and their dependencies:

- `pandas`
- `pangres`
- `beautifulsoup4`
- `requests`
- `python-dotenv`
- `sqlalchemy`
- `scikit-learn`
- `sdv`
- `torch`

To install these libraries globally, run the following command (**unrecommended**):

```
pip install -r requirements.txt
```

To install these libraries in a conda environment, run the following command (**recommended**):

```
conda create --name <thisproject>
conda activate <thisproject>
pip install -r requirements.txt
```

Additionally place a `.env` file in the root directory of the project with the following content. Write a message to [Max Kiefer](https://github.com/Maxkie1) to get the content of the `.env` file.

# Author

:bust_in_silhouette: **[Max Kiefer](https://github.com/Maxkie1)**