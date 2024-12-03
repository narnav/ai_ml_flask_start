from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
from sklearn import tree

app = Flask(__name__)

# Route to serve the HTML form (GET request)
@app.route('/')
def index():
    return render_template("index.html")

# Route to handle the form submission (POST request)
@app.route('/predict', methods=['POST',"GET"])
def submit_data():
    if request.method == "GET":
        return render_template("index.html")
    gender = request.form.get('gender')
    age = request.form.get('age')
    model=joblib.load( 'our_pridction.joblib')
    new_data = pd.DataFrame([[age, gender]], columns=['age', 'gender'])
    predictions= model.predict(new_data)
    print(predictions)
    return render_template("prediction.html",predict=predictions)


# Route to handle the form submission (POST request)
@app.route('/learn', methods=['POST',"GET"])
def learn():
    if request.method == "GET":
        return render_template("learn.html")
    
    gender = request.form.get('gender')
    age = request.form.get('age')
    genre = request.form.get('genre')  # Assuming the form has a genre field

    try:
        df = pd.read_csv('music.csv')
    except FileNotFoundError:
        # If the file doesn't exist, create an empty DataFrame with the required columns
        df = pd.DataFrame(columns=["age", "gender", "genre"])

    # Create a new row with the received data
    new_row = pd.DataFrame({"age": [age], "gender": [gender], "genre": [genre]})

    # Append the new data to the DataFrame
    df = pd.concat([df, new_row], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv('music.csv', index=False)

    music_dt = pd.read_csv('music.csv')  # Ensure 'music.csv' is in your working directory

    # update model
    # Prepare the feature set (X) and target variable (Y)
    X = music_dt.drop(columns=['genre'])  # Drop the 'genre' column for features
    Y = music_dt['genre']  # 'genre' is the target output variable

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Initialize and train the DecisionTreeClassifier model
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    joblib.dump(model, 'our_pridction.joblib')
    return render_template("index.html")
    # model=joblib.load( 'our_pridction.joblib')

    # new_data = pd.DataFrame([[age, gender]], columns=['age', 'gender'])
    # predictions= model.predict(new_data)
    # print(predictions)
    # return render_template("prediction.html",predict=predictions)

if __name__ == '__main__':
    app.run(debug=True)
