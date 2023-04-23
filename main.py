"""This module contains the main function of the personalized learning application.

It retrieves user activity logs from Moodle, aggregates them and predicts the learning style of a user.
The learning style is then used to assign the user to a learning style group in Moodle.
Each learning style group has access to a different set of course activities.
The prediction is done by a machine learning model that is trained on synthetic data.
The Felder-Silverman learning style model is the theoretical foundation of the application.

The application can be run from the command line with the following arguments:

    --train: Train the machine learning model on a synthetic dataset.
    --aggregate: Aggregate activity logs from Moodle. Requires a course ID.
    --predict: Predict learning style of an user. Requires a course ID and at least one prior --aggregate run.
    --assign: Assign an user to learning style group in Moodle. Requires --predict.
    --poll: Periodically poll Moodle for user's course completion status and trigger learning style prediction pipeline. Requires a course ID.
    --mark: Mark an user's course as completed in Moodle. Requires a course ID.

Typical usage example:

    python main.py --train --aggregate 4 --predict 4 --assign
    python main.py --poll 4
    python main.py --mark 4
"""

from src.utils import (
    train,
    aggregate,
    predict_and_assign,
    parse_arguments,
    poll,
    mark_completed,
)

# TODO: add handling for polling and course completion status, maybe first flag


def main(args):
    """Main function."""

    if args.train:
        train()

    if args.aggregate:
        aggregate(args.aggregate)

    if args.predict:
        predict_and_assign(args.predict, args.assign)

    if args.mark:
        mark_completed(args.mark)

    if args.poll:
        poll(args.poll)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
