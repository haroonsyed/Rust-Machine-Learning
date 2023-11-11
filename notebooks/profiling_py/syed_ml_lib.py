import rust_machine_learning
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Ensure categoriical data is one-hot encoded
# columns is list of ints
def one_hot_encode(df, columns):
    encoded = df.copy()
    feature_to_cols = dict()
    for col in columns:
        # Get the unique values
        unique_vals = encoded.iloc[:, col].unique()

        # Start col
        start_col = encoded.shape[1] - 1

        # Now for each unique value add a col to df
        for val in unique_vals:
            col_encoded = encoded.iloc[:, col] == val
            col_encoded = [(1.0 if label == True else 0.0) for label in col_encoded]
            encoded["is_" + encoded.columns[col] + "_" + str(val)] = col_encoded

        end_col = encoded.shape[1] - 1
        feature_to_cols[col] = list(range(start_col, end_col))

    encoded.drop(encoded.columns[columns], axis=1, inplace=True)
    return (encoded, feature_to_cols)


# Split the data into train and test
def split_train_test(df, percent_train, randomize=True):
    # Randomize the rows
    if randomize:
        df = df.sample(frac=1.0)

    # Create train and test data
    df_size = df.shape[0]
    train_size = df_size * percent_train

    train_data = df.iloc[: int(train_size), :]
    test_data = df.iloc[int(train_size) :, :]
    return (train_data, test_data)


def remove_encoded_category_data_and_label(encoded, feature_to_encoded_cols, label_col):
    cols_to_remove = []
    did_remove_label_col = False
    for k, v in feature_to_encoded_cols.items():
        if label_col in v:
            encoded.drop(encoded.columns[v], axis=1, inplace=True)
            did_remove_label_col = True
    if not did_remove_label_col:
        encoded.drop(encoded.columns[label_col], axis=1, inplace=True)


# Get categorical based on indices
def get_is_categorical(categorical_original, df_original, encoded):
    is_categorical = []

    all_categories = categorical_original
    if df_original.shape[1] != encoded.shape[1]:
        all_categories.extend(range(df_original.shape[1], encoded.shape[1]))

    for i in range(encoded.shape[1]):
        is_categorical.append(True if i in all_categories else False)
    return is_categorical


def generate_2d_data_from_function(coefficients, xStart, xEnd, numOfPoints):
    x = []
    y = []
    deltaX = (xEnd - xStart) / numOfPoints
    for i in range(0, numOfPoints):
        currX = xStart + i * deltaX

        currY = 0.0
        for idx, coefficient in enumerate(coefficients):
            currY += coefficient * currX**idx
        x.append(currX)
        y.append(currY)
    return (x, y)

def view_image(data, img_x, img_y, colorscale=""):

    img_data = []

    if colorscale == "":
        # data is channel -> list of values
        # Convert the data from [[red_values], [green_values], [blue_values]] to [[red, green, blue]]
        pixel_count = img_x * img_y
        img_data = []
        for i in range(pixel_count):
            img_data.append([[data[0][i], data[1][i], data[2][i]]])
        data = np.array(img_data)
        depth = 3
        img_data = data.reshape(img_x, img_y, depth)
    else:
        img_data = data.reshape(img_x, img_y)

    # Quick image viewer
    fig = px.imshow(img_data, color_continuous_scale=colorscale)
    fig.show()