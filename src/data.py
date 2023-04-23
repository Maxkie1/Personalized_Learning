"""This module contains data processing functionalities."""

import pandas as pd
import numpy as np
from sdv.tabular import TVAE
from sklearn.model_selection import train_test_split


def create_synthetic_dataset(sample_size: int) -> pd.DataFrame:
    """Create synthetic dataset.

    Create a synthetic dataset with SDV using the VTAE model.
    The following list shows which activities are associated with which learning style:
        - sensing: Quiz, Glossary, Assignment, Example
        - intuitive: Forum, URL, Book, Page
        - visual: Video, Image, File, Folder
        - verbal: FAQ, Video, Book, Glossary
        - active: Workshop, Assignment, Chat, Forum
        - reflective: Quiz, Book, Folder, Page
        - sequential: Lesson, Folder, Assignment, File
        - global: URL, Video, Forum,  Example

    Args:
        sample_size: Number of entries in the synthetic dataset.

    Returns:
        The synthetic dataset as a dataframe.
    """

    learning_styles = [
        "sensing",
        "intuitive",
        "visual",
        "verbal",
        "active",
        "reflective",
        "sequential",
        "global",
    ]

    activity_mappping = {
        "sensing": ["Quiz", "Glossary", "Assignment", "Example"],
        "intuitive": ["Forum", "URL", "Book", "Page"],
        "visual": ["Video", "Image", "File", "Folder"],
        "verbal": ["FAQ", "Video", "Book", "Glossary"],
        "active": ["Workshop", "Assignment", "Chat", "Forum"],
        "reflective": ["Quiz", "Book", "Folder", "Page"],
        "sequential": ["Lesson", "Folder", "Assignment", "File"],
        "global": ["URL", "Video", "Forum", "Example"],
    }

    dataset_size = 1000

    dataset = pd.DataFrame()
    dataset["student_id"] = np.arange(dataset_size)
    dataset["Book"] = np.random.randint(0, 10, dataset_size)
    dataset["Forum"] = np.random.randint(0, 10, dataset_size)
    dataset["FAQ"] = np.random.randint(0, 10, dataset_size)
    dataset["Quiz"] = np.random.randint(0, 10, dataset_size)
    dataset["Glossary"] = np.random.randint(0, 10, dataset_size)
    dataset["URL"] = np.random.randint(0, 10, dataset_size)
    dataset["File"] = np.random.randint(0, 10, dataset_size)
    dataset["Video"] = np.random.randint(0, 10, dataset_size)
    dataset["Image"] = np.random.randint(0, 10, dataset_size)
    dataset["Chat"] = np.random.randint(0, 10, dataset_size)
    dataset["Workshop"] = np.random.randint(0, 10, dataset_size)
    dataset["Page"] = np.random.randint(0, 10, dataset_size)
    dataset["Assignment"] = np.random.randint(0, 10, dataset_size)
    dataset["Folder"] = np.random.randint(0, 10, dataset_size)
    dataset["Lesson"] = np.random.randint(0, 10, dataset_size)
    dataset["Example"] = np.random.randint(0, 10, dataset_size)
    dataset["label"] = np.random.choice(learning_styles, dataset_size)

    for learning_style in learning_styles:
        for activity in activity_mappping[learning_style]:
            dataset.loc[
                dataset["label"] == learning_style, activity
            ] = np.random.randint(
                30, 50, dataset["label"].value_counts()[learning_style]
            )

    metadata = {
        "fields": {
            "student_id": {"type": "id", "subtype": "integer"},
            "Book": {"type": "numerical", "subtype": "integer"},
            "Forum": {"type": "numerical", "subtype": "integer"},
            "FAQ": {"type": "numerical", "subtype": "integer"},
            "Quiz": {"type": "numerical", "subtype": "integer"},
            "Glossary": {"type": "numerical", "subtype": "integer"},
            "URL": {"type": "numerical", "subtype": "integer"},
            "File": {"type": "numerical", "subtype": "integer"},
            "Video": {"type": "numerical", "subtype": "integer"},
            "Image": {"type": "numerical", "subtype": "integer"},
            "Chat": {"type": "numerical", "subtype": "integer"},
            "Workshop": {"type": "numerical", "subtype": "integer"},
            "Page": {"type": "numerical", "subtype": "integer"},
            "Assignment": {"type": "numerical", "subtype": "integer"},
            "Folder": {"type": "numerical", "subtype": "integer"},
            "Lesson": {"type": "numerical", "subtype": "integer"},
            "Example": {"type": "numerical", "subtype": "integer"},
            "label": {"type": "categorical"},
        },
        "constraints": [],
        "primary_key": "student_id",
    }

    model = TVAE(table_metadata=metadata)
    model.fit(dataset)
    synthetic_dataset = model.sample(num_rows=sample_size)

    return synthetic_dataset


def activity_logs_to_dataframe(activity_logs: list[dict]) -> pd.DataFrame:
    """Convert activity logs to dataframe.

    This function counts how often a student has interacted with a specific type of activity by using the activity logs.
    The result is a dataframe with a row per student and a column per activity type.
    Following activity types are considered: Book, Forum, FAQ, Quiz, Glossary, URL, File, Video, Image, Chat, Workshop, Page, Assignment, Folder, Lesson, Example.

    Args:
        activity_logs: Activity logs as a list of dictionaries.

    Returns:
        The activity logs as a dataframe.
    """

    activity_types = [
        "Book",
        "Forum",
        "FAQ",
        "Quiz",
        "Glossary",
        "URL",
        "File",
        "Video",
        "Image",
        "Chat",
        "Workshop",
        "Page",
        "Assignment",
        "Folder",
        "Lesson",
        "Example",
    ]

    # initialize dataframe with student IDs as index and activity types as columns
    student_ids = set(
        [int(log["description"].split(" ")[4][1:-1]) for log in activity_logs]
    )
    student_ids = sorted(list(student_ids))
    df = pd.DataFrame(index=student_ids, columns=activity_types).fillna(0)
    df.index.name = "student_id"

    # increment count for each student-activity combination in the logs
    for log in activity_logs:
        student_id = int(log["description"].split(" ")[4][1:-1])
        activity_type = log["component"]
        df.loc[student_id, activity_type] += 1

    return df


def df_row_to_new_df(df: pd.DataFrame, index: int, index_name: str) -> pd.DataFrame:
    """Convert dataframe row to new dataframe.

    This function converts a row of a dataframe to a new dataframe.

    Args:
        df: The dataframe to be accessed.
        index: The index of the row to be converted.
        index_name: The name of the index.

    Returns:
        The row as a new dataframe.
    """

    new_df = df.loc[index].to_frame().T
    new_df.index.name = index_name

    return new_df


def transform_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform data.

    Transform incoming data into separate feature and label dataframes that can be used for training.

    Args:
        data: Raw dataframe.

    Returns:
        The transformed data split into a feature and label dataframe.

    """

    x_data = data.drop(["label", "student_id"], axis=1)
    y_data = data["label"]
    y_data = y_data.replace(
        {
            "sensing": 0,
            "intuitive": 1,
            "visual": 2,
            "verbal": 3,
            "active": 4,
            "reflective": 5,
            "sequential": 6,
            "global": 7,
        }
    )

    return x_data, y_data


def split_data(
    x_data: pd.DataFrame, y_data: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data.

    Split incoming data into train and test sets.

    Args:
        x_data: The feature dataframe.
        y_data: The label dataframe.
        test_size: The size of the test set.

    Returns:
        The train test sets split into respective feature and label dataframes.
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=y_data,
    )

    return x_train, y_train, x_test, y_test
