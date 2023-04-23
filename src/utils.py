"""This module wraps all functionalities needed for the main script."""

import src.data as data
import src.machine_learning as ml
import src.moodle as moodle
import src.database as db
import argparse
import time

MODEL_PATH = "models/model.pt"
DATABASE_PATH = "database/database.db"


def train():
    """Train the machine learning model on a synthetic dataset and save it to disk."""

    synthetic_dataset = data.create_synthetic_dataset(10000)
    x_data, y_data = data.transform_data(synthetic_dataset)
    x_train, y_train, x_test, y_test = data.split_data(x_data, y_data, 0.1)
    model = ml.train_model(x_train, y_train, 300)
    ml.evaluate_model(x_test, y_test, model)
    ml.save_model(model, MODEL_PATH)


def aggregate(course_id: int, user_id: int = None):
    """Aggregate activity logs from Moodle and insert them into the database.

    Args:
        course_id: The course ID of the course activity logs should be aggregated for.
        user_id: Optional user ID. If provided, only activity logs for the provided user are aggregated.
    """

    if user_id:
        # aggregate activity logs for provided user ID
        activity_logs = moodle.get_user_activity_logs(user_id, course_id)
    else:
        # aggregate activity logs for all users in course
        activity_logs = moodle.aggregate_user_activity_logs(course_id)

    df = data.activity_logs_to_dataframe(activity_logs)

    aggregate_engine = db.create_db_engine(DATABASE_PATH)
    db.initialize_activity_table(aggregate_engine)
    db.update_user_data(aggregate_engine, df)
    aggregate_engine.dispose()


def predict_and_assign(course_id: int, assign: bool = False, user_id: int = None):
    """Predict learning styles for users and assign them to learning style groups in Moodle.

    Prediction and assignment tasks are combined in one function to reduce redudancy.

    Args:
        course_id: The course ID of the course users to be predicted are enrolled in.
        assign: Boolean value indicating whether the user should be assigned to a learning style group.
        user_id: Optional user ID. If provided, only the learning style for the provided user is predicted.
    """

    loaded_model = ml.load_model(MODEL_PATH)
    predict_engine = db.create_db_engine(DATABASE_PATH)

    if user_id:
        # predict learning style for provided user ID
        print(
            "utils.predict_and_assign: Predicting learning style for user ID {}...".format(
                user_id
            )
        )
        user = db.fetch_user_data(predict_engine, user_id)
        ls_id, confidence = ml.predict(user, loaded_model)
        if assign:
            moodle.assign_learning_style(ls_id, user_id, confidence)
    else:
        # predict learning style for all users in course
        enrolled_users = moodle.get_enrolled_users(course_id)
        for enrolled_user in enrolled_users:
            print(
                "utils.predict_and_assign: Predicting learning style for user ID {}...".format(
                    enrolled_user["id"]
                )
            )
            user = db.fetch_user_data(predict_engine, enrolled_user["id"])
            ls_id, confidence = ml.predict(user, loaded_model)
            if assign:
                moodle.assign_learning_style(ls_id, enrolled_user["id"], confidence)

    predict_engine.dispose()


def poll(course_id: int):
    """Poll Moodle for user's course completion status and trigger learning style prediction pipeline.

    If the user meets the prediction criteria, an aggregation, prediction and assignment is triggered.
    It polls Moodle every 60 minutes.
    This function is the main element of the automated learning style prediction pipeline used in the containerized version of the application.

    Args:
        course_id: The course ID of the course to be polled.
    """

    while True:
        print("_________________________________________________________________")
        print("utils.poll: Polling Moodle for course completion status...")

        users = moodle.get_enrolled_users(course_id)
        for user in users:
            if user_meets_prediction_criteria(course_id, user["id"]):
                aggregate(course_id, user["id"])
                predict_and_assign(course_id, True, user["id"])
        time.sleep(60)


def user_meets_prediction_criteria(course_id: int, user_id: int) -> bool:
    """Check if an user meets the prediction criteria.

    A learning style prediction is only triggered if the user completed the course.

    Args:
        course_id: The course ID of the course the user is enrolled in.
        user_id: The user ID.

    Returns:
        Boolean value indicating whether the user meets the prediction criteria.
    """

    course_completion = moodle.get_user_course_completion(user_id, course_id)
    if course_completion["completionstatus"]["completions"][0]["complete"]:
        print(
            "utils.user_meets_prediction_criteria: User ID {} has completed the course ID {}.".format(
                user_id, course_id
            )
        )
        return True
    else:
        print(
            "utils.user_meets_prediction_criteria: User ID {} has not completed the course ID {} yet.".format(
                user_id, course_id
            )
        )
        return False


def mark_completed(course_id: int):
    """Mark an user's course completion status in Moodle.

    Args:
        course_id: The course ID of the course to be marked as completed.
    """

    moodle.mark_course_completed(course_id)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        The parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        help="Train the machine learning model on a synthetic dataset.",
        action="store_true",
    )
    parser.add_argument(
        "--aggregate",
        help="Aggregate activity logs from moodle. Requires a course ID.",
        action="store",
        type=int,
    )
    parser.add_argument(
        "--predict",
        help="Predict learning style of an user. Requires a course ID and at least one prior --aggregate run.",
        action="store",
        type=int,
    )
    parser.add_argument(
        "--assign",
        help="Assign user to learning style group in Moodle. Requires --predict.",
        action="store_true",
    )
    parser.add_argument(
        "--poll",
        help="Periodically poll Modle for user's course completion status and trigger learning style prediction pipeline. Requires a course ID.",
        action="store",
        type=int,
    )
    parser.add_argument(
        "--mark",
        help="Mark a user's course as completed in Moodle. Requires a course ID.",
        action="store",
        type=int,
    )

    args = parser.parse_args()

    if args.assign and not args.predict:
        parser.error(
            "Running --assign requires running --predict as well. Run --help for more information."
        )
    elif (
        not args.train
        and not args.aggregate
        and not args.predict
        and not args.poll
        and not args.mark
    ):
        parser.error(
            "At least one argument is required. Run --help for more information."
        )

    return args
