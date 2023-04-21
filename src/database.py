"""This module contains database functionalities."""

from pangres import upsert
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd


def create_db_engine(path):
    """Create engine to SQLite database.

    Create an engine to the SQLite database at the given path.

    Args:
        path: Path to the database file.

    Returns:
        The engine object to the database.
    """

    engine = create_engine(f"sqlite:///{path}", echo=False)

    return engine


def initialize_activity_table(engine):
    """Initialize activity table in database.

    Create the activity table in the database if it does not exist yet.
    The student ID is the primary key.
    Each column represents how many times the student has interacted with a specific learning activity.
    Following activity types are considered: Book, Forum, FAQ, Quiz, Glossary, URL, File, Video, Image, Chat, Workshop, Page, Assignment, Folder, Lesson, Example.

    Args:
        engine: The engine object to the database.
    """

    query = "CREATE TABLE IF NOT EXISTS student_activities (student_id INTEGER PRIMARY KEY, Book INTEGER, Forum INTEGER, FAQ INTEGER, Quiz INTEGER, Glossary INTEGER, URL INTEGER, File INTEGER, Video INTEGER, Image INTEGER, Chat INTEGER, Workshop INTEGER, Page INTEGER, Assignment INTEGER, Folder INTEGER, Lesson INTEGER, Example INTEGER)"

    with engine.connect() as connection:
        connection.execute(text(query))
        connection.commit()


def insert_student_data(engine, df):
    """Insert data into activity table.

    Insert a dataframe into the activity table.
    This either updates an existing entry by overwriting it or creates a new entry if the student ID does not exist yet in the table.

    Args:
        engine: The engine object to the database.
        df: The dataframe to be inserted into the database.
    """

    upsert(engine, df, "student_activities", if_row_exists="update")

    student_ids = ", ".join([str(student_id) for student_id in df.index.values])

    if len(df.index.values) > 1:
        print(
            "db.insert_student_data: Inserted data of student IDs {} into database.".format(
                student_ids
            )
        )
    else:
        print(
            "db.insert_student_data: Inserted data of student ID {} into database.".format(
                student_ids
            )
        )


def fetch_complete_data(engine):
    """Fetch complete data from activity table.

    Args:
        engine: The engine object to the database.
    """

    query = "SELECT * FROM student_activities"

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            rows = result.fetchall()
            print(
                "db.fetch_complete_data: Fetched complete data consisting of {} rows from database.".format(
                    len(rows)
                )
            )
            return rows
    except SQLAlchemyError as e:
        raise e


def fetch_student_data(engine, student_id):
    """Fetch data of a specific student from activity table.

    Args:
        engine: The engine object to the database.
        student_id: The student ID of the student whose data should be fetched.
    """

    query = "SELECT * FROM student_activities WHERE student_id=:student_id"

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query), {"student_id": student_id})
            try:
                rows = result.fetchall()
                print(
                    "db.fetch_student_data: Fetched data of student ID {} from database.".format(
                        student_id
                    )
                )
                return rows[0]
            except IndexError as e:
                raise Exception(
                    "Student ID {} does not exist in database.".format(student_id)
                )
    except SQLAlchemyError as e:
        raise e


def update_student_data(engine, df):
    """Update a specific student's data in activity table.

    This function calculates the difference between the values of the dataframe and the values of the database and inserts the result into the database.

    Args:
        engine: The engine object to the database.
        df: The dataframe containing
    """

    db_data = fetch_student_data(engine, df.index.values[0])
    db_data = list(db_data)[1:]
    df_data = list(df.values[0])

    diff = [df_data[i] - db_data[i] for i in range(len(df_data))]

    df_new = pd.DataFrame()
    df_new["student_id"] = [df.index.values[0]]
    df_new["Book"] = [diff[0]]
    df_new["Forum"] = [diff[1]]
    df_new["FAQ"] = [diff[2]]
    df_new["Quiz"] = [diff[3]]
    df_new["Glossary"] = [diff[4]]
    df_new["URL"] = [diff[5]]
    df_new["File"] = [diff[6]]
    df_new["Video"] = [diff[7]]
    df_new["Image"] = [diff[8]]
    df_new["Chat"] = [diff[9]]
    df_new["Workshop"] = [diff[10]]
    df_new["Page"] = [diff[11]]
    df_new["Assignment"] = [diff[12]]
    df_new["Folder"] = [diff[13]]
    df_new["Lesson"] = [diff[14]]
    df_new["Example"] = [diff[15]]
    df_new.set_index("student_id", inplace=True)

    insert_student_data(engine, df_new)
    print(
        "db.update_student_data: Updated data of student ID {} in database.".format(
            df.index.values[0]
        )
    )
