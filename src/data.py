"""This module contains data processing functionalities.

It creates a synthetic dataset which is used to train the model.
"""

import pandas as pd
import numpy as np
from sdv.lite import TabularPreset
from sklearn.model_selection import train_test_split


def create_synthetic_data(sample_size):
    """Create synthetic dataset.
    
    Create a synthetic dataset with SDV using the FAST_ML preset.

    Args:
        sample_size: Number of rows in the synthetic dataset.

    Returns:
        synthetic_data: Synthetic dataset.
    """

    dimensions = ['visual', 'auditory', 'read/write', 'kinesthetic']

    dataset = pd.DataFrame()
    dataset['student_id'] = np.arange(10)
    dataset['video'] = np.random.randint(0, 10, 10)
    dataset['quiz'] = np.random.randint(0, 10, 10)
    dataset['forum'] = np.random.randint(0, 10, 10)
    dataset['text'] = np.random.randint(0, 10, 10)
    dataset['audio'] = np.random.randint(0, 10, 10)
    dataset['image'] = np.random.randint(0, 10, 10)
    dataset['exercise'] = np.random.randint(0, 10, 10)
    dataset['interactive'] = np.random.randint(0, 10, 10)
    dataset['simulation'] = np.random.randint(0, 10, 10)
    dataset['label'] = np.random.choice(dimensions, 10)

    metadata = {
        'fields': {
            'student_id': {'type': 'id', 'subtype': 'integer'},
            'video': {'type': 'numerical', 'subtype': 'integer'},
            'quiz': {'type': 'numerical', 'subtype': 'integer'},
            'forum': {'type': 'numerical', 'subtype': 'integer'},
            'text': {'type': 'numerical', 'subtype': 'integer'},
            'audio': {'type': 'numerical', 'subtype': 'integer'},
            'image': {'type': 'numerical', 'subtype': 'integer'},
            'exercise': {'type': 'numerical', 'subtype': 'integer'},
            'interactive': {'type': 'numerical', 'subtype': 'integer'},
            'simulation': {'type': 'numerical', 'subtype': 'integer'},
            'label': {'type': 'categorical'}
        },
        'constraints': [],
        'primary_key': 'student_id'
    }

    model = TabularPreset(name='FAST_ML', metadata=metadata)
    model.fit(dataset)
    synthetic_data = model.sample(num_rows=sample_size)

    return synthetic_data

def transform_data(data):
    """Transform data.

    Transform data into a format that can be used for training.
    x_data contains the features and y_data contains the labels.

    Args:
        data: Untransformed data.
    
    Returns:
        x_data: Features.
        y_data: Labels.
    """

    x_data = data.drop(['label', 'student_id'], axis=1)
    y_data = data['label']
    y_data = y_data.replace({'visual': 0, 'auditory': 1, 'read/write': 2, 'kinesthetic': 3})

    return x_data, y_data

def split_data(x_data, y_data):
    """Split data.

    Split data into train and test sets.

    Args:
        x_data: Features.
        y_data: Labels.
    
    Returns:
        x_train: Training data.
        x_test: Testing data.
        y_train: Training labels.
        y_test: Testing labels.
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
