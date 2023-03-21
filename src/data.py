"""This module contains data processing functionalities."""

import pandas as pd
import numpy as np
from sdv.lite import TabularPreset
from sklearn.model_selection import train_test_split


def create_synthetic_dataset(sample_size):
    """Create synthetic dataset.

    Create a synthetic dataset with SDV using the FAST_ML preset.
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

    np.random.seed(42)

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
            ] = np.random.randint(40, 50)

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

    model = TabularPreset(name="FAST_ML", metadata=metadata)
    model.fit(dataset)
    synthetic_dataset = model.sample(num_rows=sample_size, randomize_samples=False)

    return synthetic_dataset


def activity_logs_to_dataframe(activity_logs):
    """Convert activity logs to dataframe.

    This functions counts how often a student has interacted with a specific type of activity by using the activity logs.
    The result is a dataframe with a row per student and a column per activity type.
    Following activity types are considered: Book, Forum, Quiz, Glossary, Video, Picture, FAQ, Page, Assignment, Chat, Workshop, Folder, Lesson.

    Args:
        activity_logs: Activity logs as a list of dictionaries.

    Returns:
        The activity logs as a dataframe.
    """

    df = pd.DataFrame()
    df["student_id"] = np.arange(10)
    df["Book"] = 0
    df["Forum"] = 0
    df["FAQ"] = 0
    df["Quiz"] = 0
    df["Glossary"] = 0
    df["URL"] = 0
    df["File"] = 0
    df["Video"] = 0
    df["Image"] = 0
    df["Chat"] = 0
    df["Workshop"] = 0
    df["Page"] = 0
    df["Assignment"] = 0
    df["Folder"] = 0
    df["Lesson"] = 0
    df["Example"] = 0

    for log in activity_logs:
        df.loc[
            df["student_id"] == int((log["description"].split(" ")[4])[1:-1]),
            log["component"],
        ] += 1

    return df


def transform_data(data):
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


def split_data(x_data, y_data, test_size):
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
