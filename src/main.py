"""Appplication to categorize learning styles of students.

This application retrieves activity log data from Moodle and matches students to the dimensions of the Felder-Silverman learning style model based on their personal acitivity log.
"""

import data
import model

def main():
    """Main function."""

    synthetic_data = data.create_synthetic_data(20000)
    x_data, y_data = data.transform_data(synthetic_data)
    x_train, x_test, y_train, y_test = data.split_data(x_data, y_data)
    model.train_and_evaluate_model(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
