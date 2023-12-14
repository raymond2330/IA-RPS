import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

# Function to import and combine datasets for rock, paper, and scissor
def import_and_combine_datasets():
    # Read CSV files for rock, paper, and scissor gestures
    gestures = ['rock', 'paper', 'scissor']
    data_frames = []
    column_names = [
        'WRIST_X', 'WRIST_Y', 'WRIST_Z',
    'THUMB_CMC_X', 'THUMB_CMC_Y', 'THUMB_CMC_Z',
    'THUMB_MCP_X', 'THUMB_MCP_Y', 'THUMB_MCP_Z',
    'THUMB_IP_X', 'THUMB_IP_Y', 'THUMB_IP_Z',
    'THUMB_TIP_X', 'THUMB_TIP_Y', 'THUMB_TIP_Z',
    'INDEX_FINGER_MCP_X', 'INDEX_FINGER_MCP_Y', 'INDEX_FINGER_MCP_Z',
    'INDEX_FINGER_PIP_X', 'INDEX_FINGER_PIP_Y', 'INDEX_FINGER_PIP_Z',
    'INDEX_FINGER_DIP_X', 'INDEX_FINGER_DIP_Y', 'INDEX_FINGER_DIP_Z',
    'INDEX_FINGER_TIP_X', 'INDEX_FINGER_TIP_Y', 'INDEX_FINGER_TIP_Z',
    'MIDDLE_FINGER_MCP_X', 'MIDDLE_FINGER_MCP_Y', 'MIDDLE_FINGER_MCP_Z',
    'MIDDLE_FINGER_PIP_X', 'MIDDLE_FINGER_PIP_Y', 'MIDDLE_FINGER_PIP_Z',
    'MIDDLE_FINGER_DIP_X', 'MIDDLE_FINGER_DIP_Y', 'MIDDLE_FINGER_DIP_Z',
    'MIDDLE_FINGER_TIP_X', 'MIDDLE_FINGER_TIP_Y', 'MIDDLE_FINGER_TIP_Z',
    'RING_FINGER_MCP_X', 'RING_FINGER_MCP_Y', 'RING_FINGER_MCP_Z',
    'RING_FINGER_PIP_X', 'RING_FINGER_PIP_Y', 'RING_FINGER_PIP_Z',
    'RING_FINGER_DIP_X', 'RING_FINGER_DIP_Y', 'RING_FINGER_DIP_Z',
    'RING_FINGER_TIP_X', 'RING_FINGER_TIP_Y', 'RING_FINGER_TIP_Z',
    'PINKY_MCP_X', 'PINKY_MCP_Y', 'PINKY_MCP_Z',
    'PINKY_PIP_X', 'PINKY_PIP_Y', 'PINKY_PIP_Z',
    'PINKY_DIP_X', 'PINKY_DIP_Y', 'PINKY_DIP_Z',
    'PINKY_TIP_X', 'PINKY_TIP_Y', 'PINKY_TIP_Z',
    'label'
    ]

    for gesture in gestures:
        file_path = f'dataset/{gesture}/{gesture}_hand_landmarks.csv'
        df = pd.read_csv(file_path, skiprows=1, names=column_names)
        data_frames.append(df)

    # Concatenate data frames for all gestures into a single data frame
    combined_df = pd.concat(data_frames)
    return combined_df

# Function to split the dataset
def split_dataset(combined_data):
    X = combined_data.drop('label', axis=1)  # Features (hand landmarks data)
    y = combined_data['label']  # Target variable (gesture labels)

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=100)
    return X_train, X_test, y_train, y_test

# Function to train the decision tree classifier
def train_decision_tree(X_train, X_test, y_train):
    clf = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=10, min_samples_leaf=5)
    clf.fit(X_train, y_train)
    return clf

# Function to make predictions
def make_predictions(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

# Function to calculate accuracy
def calculate_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    print("Report : ", classification_report(y_test, y_pred))

# Driver code
def main():
    # Import and combine datasets for rock, paper, and scissor gestures
    combined_data = import_and_combine_datasets()

    # Split the combined dataset
    X_train, X_test, y_train, y_test = split_dataset(combined_data)

    # Train the decision tree classifier
    clf = train_decision_tree(X_train, X_test, y_train)

    joblib.dump(clf, "hotdog.hotdog")

    # Visualize the decision tree
    plt.figure(figsize=(30, 20))
    plot_tree(clf, feature_names=X_train.columns, class_names=clf.classes_, filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()
    # plt.savefig('decision_tree_visualization.svg') 

    # Make predictions
    y_pred = make_predictions(X_test, clf)

    # Calculate accuracy
    calculate_accuracy(y_test, y_pred)

# Calling main function
if __name__ == "__main__":
    main()
