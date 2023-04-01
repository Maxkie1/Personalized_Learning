"""This module contains the main function of the personalized learning application.

It retrieves student activity logs from Moodle, aggregates them and predicts the learning style of a student.
The learning style is then used to assign the student to a learning style group in Moodle.
Each learning style group has access to a different set of course activities.
The prediction is done by a machine learning model that is trained on synthetic data.
The Felder-Silverman learning style model is the theoretical foundation of the application.

The application can be run from the command line with the following arguments:

    --train: train the model on a synthetic dataset
    --aggregate: retrieve activity logs from moodle
    --predict: predict learning style of a student (requires --aggregate)
    --assign: assign student to learning style group in moodle (requires --aggregate and --predict)

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
        synthetic_dataset = data.create_synthetic_dataset(10000)
        x_data, y_data = data.transform_data(synthetic_dataset)
        x_train, y_train, x_test, y_test = data.split_data(x_data, y_data, 0.1)
        model = ml.train_model(x_train, y_train, 300)
        ml.evaluate_model(x_test, y_test, model)
        ml.save_model(model, path)

    if args.aggregate:
        activity_logs = moodle.aggregate_user_activity_logs(course_id)
        df = data.activity_logs_to_dataframe(activity_logs)
        print("Aggregated data:\n {}".format(df))

    if args.predict:
        loaded_model = ml.load_model(path)
        users = moodle.get_enrolled_users(course_id)
        for user in users:
            student = df.loc[df["student_id"] == user["id"]]
            print("Student data:\n {}".format(student))
            student = student.drop(["student_id"], axis=1)
            ls_id = ml.predict(student, loaded_model)
            print(
                "Student ID:",
                user["id"],
                "| Predicted Learning Style:",
                learning_styles[ls_id],
            )
            if args.assign:
                moodle.learning_style_assignment(ls_id, user["id"])
        
def parse_arguments():
    """Parse command line arguments.

    Returns:
        The parsed arguments.
    """

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
        help="predict learning style of a student",
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
