"""This module contains database functionalities."""

from pangres import upsert
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import src.data as data


def create_db_engine(path: str) -> Engine:
    """Create engine to SQLite database.

    Create an engine to the SQLite database at the given path.

    Args:
        path: Path to the database file.

    Returns:
        The engine object to the database.
    """

    engine = create_engine(f"sqlite:///{path}", echo=False)

    return engine


def initialize_activity_table(engine: Engine):
    """Initialize activity table in database.

    Create the activity table in the database if it does not exist yet.
    The user ID is the primary key.
    Each column represents how many times the user has interacted with a specific learning activity.
    Following activity types are considered: Book, Forum, FAQ, Quiz, Glossary, URL, File, Video, Image, Chat, Workshop, Page, Assignment, Folder, Lesson, Example.

    Args:
        engine: The engine object to the database.
    """

    query = "CREATE TABLE IF NOT EXISTS user_activities (user_id INTEGER PRIMARY KEY, Book INTEGER, Forum INTEGER, FAQ INTEGER, Quiz INTEGER, Glossary INTEGER, URL INTEGER, File INTEGER, Video INTEGER, Image INTEGER, Chat INTEGER, Workshop INTEGER, Page INTEGER, Assignment INTEGER, Folder INTEGER, Lesson INTEGER, Example INTEGER)"

    with engine.connect() as connection:
        connection.execute(text(query))
        connection.commit()


def insert_user_data(engine: Engine, df: pd.DataFrame):
    """Insert data into activity table.

    Insert a dataframe into the activity table.
    This either updates an existing entry by overwriting it or creates a new entry if the user ID does not exist yet in the table.

    Args:
        engine: The engine object to the database.
        df: The dataframe to be inserted into the database.
    """

    upsert(engine, df, "user_activities", if_row_exists="update")

    user_ids = ", ".join([str(user_id) for user_id in df.index.values])

    if len(df.index.values) > 1:
        print(
            "db.insert_user_data: Inserted data of user IDs {} into database.".format(
                user_ids
            )
        )
    else:
        print(
            "db.insert_user_data: Inserted data of user ID {} into database.".format(
                user_ids
            )
        )


def fetch_complete_data(engine: Engine) -> list[tuple]:
    """Fetch complete data from activity table.

    Args:
        engine: The engine object to the database.

    Returns
        A list of tuples containing the complete data from the activity table.
    """

    query = "SELECT * FROM user_activities"

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


def fetch_user_data(engine: Engine, user_id: int) -> tuple:
    """Fetch data of a specific user from activity table.

    Args:
        engine: The engine object to the database.
        user_id: The user ID whose data is to be fetched.

    Returns:
        A tuple containing the data of the user.
    """

    query = "SELECT * FROM user_activities WHERE user_id=:user_id"

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query), {"user_id": int(user_id)})
            try:
                rows = result.fetchall()
                result = rows[0]
                print("db.fetch_user_data: Fetched data of user ID {}.".format(user_id))
                return result
            except IndexError:
                # print with function name
                print("db.fetch_user_data: User ID {} does not exist.".format(user_id))
                return None
    except SQLAlchemyError as e:
        raise e


def update_user_data(engine: Engine, df: pd.DataFrame):
    """Update a specific users's data in activity table.

    This function calculates the difference between the values of the dataframe and the values of the database and inserts the result into the database.

    Args:
        engine: The engine object to the database.
        df: The dataframe containing the user data to be updated.
    """

    for user_id in df.index.values:
        print("db.update_user_data: Updating user ID {}...".format(user_id))

        db_data = fetch_user_data(engine, user_id)
        if db_data is None:
            print(
                "db.update_user_data: Creating new entry for user ID {}...".format(
                    user_id
                )
            )
            user_data = data.df_row_to_new_df(df, user_id, "user_id")
            insert_user_data(engine, user_data)
            continue

        db_data = list(db_data)[1:]
        df_data = list(df.loc[user_id].values)

        if db_data == df_data:
            print(
                "db.update_user_data: No update required for user ID {} as data is already up-to-date.".format(
                    user_id
                )
            )
            continue

        diff = [df_data[i] - db_data[i] for i in range(len(db_data))]

        df_new = pd.DataFrame()
        df_new["user_id"] = [user_id]
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
        df_new.set_index("user_id", inplace=True)

        insert_user_data(engine, df_new)
        print("db.update_user_data: Updated user ID {}.".format(user_id))
