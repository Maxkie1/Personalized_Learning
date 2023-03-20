"""This module contains the main function of the application.

This application retrieves activity logs from Moodle.
Based on the activity logs the Felder-Silverman learning style of the student is predicted.
The prediction is done by a machine learning model that is trained on synthetic data.
Finally, the student is assigned to a learning style group in Moodle.
Each learning style group has access to a different set of learning activities.

Typical usage example:

    python main.py --train --aggregate --predict --assign
"""

import argparse
import data
import machine_learning as ml
import moodle


def main(args):
    """Main function."""

    course_id = 4
    user_id = 8
    path = "../models/model.pt"
    learning_styles = {
        1: "sensing",
        2: "intuitive",
        3: "visual",
        4: "verbal",
        5: "active",
        6: "reflective",
        7: "sequential",
    }

    if args.train:
        synthetic_dataset = data.create_synthetic_data(20000)
        x_data, y_data = data.transform_data(synthetic_dataset)
        x_train, y_train, x_test, y_test = data.split_data(x_data, y_data, 0.9)
        model = ml.train_model(x_train, y_train, 300)
        ml.evaluate_model(x_test, y_test, model)
        ml.save_model(model, path)

    if args.aggregate:
        activity_logs = moodle.aggregate_user_activity_logs(course_id)
        df = data.activity_logs_to_dataframe(activity_logs)

    if args.predict:
        loaded_model = ml.load_model(path)
        student = df.loc[df["student_id"] == user_id]
        print("Student data:\n {}".format(student))
        student = student.drop(["student_id"], axis=1)
        ls_id = ml.predict(student, loaded_model)
        print(
            'Predicted learning style "{}" for student with ID {}.'.format(
                learning_styles[ls_id], user_id
            )
        )

    if args.assign:
        moodle.learning_style_assignment(ls_id, user_id)


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        help="train the model on synthetic data",
        action="store_true",
    )
    parser.add_argument(
        "--aggregate",
        help="retrieve activity logs from moodle",
        action="store_true",
    )
    parser.add_argument(
        "--predict",
        help="predict learning style for a student",
        action="store_true",
    )
    parser.add_argument(
        "--assign",
        help="assign student to learning style group in moodle",
        action="store_true",
    )
    args = parser.parse_args()

    if args.assign and (not args.predict or not args.aggregate):
        parser.error("--assign requires --aggregate and --predict")
    elif args.predict and not args.aggregate:
        parser.error("--predict requires --aggregate")
    elif not args.train and not args.aggregate:
        parser.error("at least one of --train or --aggregate is required")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
