from flask import Flask, render_template, url_for, request, redirect
import sqlite3
import os
import shutil
import csv
import pandas as pd
 
connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('userlog.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()
        print(result)
        if result:
            return render_template('userlog.html')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/Result', methods=['POST', 'GET'])
def Result():
    # Load the CSV file
    df = pd.read_csv("tracked_objects.csv")
    df["Gender"] = df["Gender"].replace("None", pd.NA)
    df["Age"] = df["Age"].replace("None", pd.NA)
    print(df)

    # Count values
    total_male = (df["Gender"] == "Male").sum()
    total_female = (df["Gender"] == "Female").sum()

    # Count animals (where Gender is None)
    animal_counts = df[df["Gender"].isna()]["Object Name"].value_counts()
    print(animal_counts)
    # Print each animal count without the extra line
    animals = []
    counts = 0
    for animal, count in animal_counts.items():
        animals.append(f"{animal}: {count}")
        counts += count
    # Print results
    result = []
    result.append(f"Total Males: {total_male}")
    result.append(f"Total Females: {total_female}")
    total = total_male + total_female
    result.append(f"Total People: {total}")
    result.append(f'Animal Counts: {counts}')
    result.append(animals)

    f = open('tracked_objects.csv', 'r')
    reader = csv.reader(f)
    rows = []
    for row in reader:
        rows.append(row)

    return render_template('userlog.html', result=result, rows=rows)

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        image = request.form['img']
        path = 'static/test/'+image
        f = open('video.txt', 'w')
        f.write(path)
        f.close()
        os.system('python detect.py')
        return redirect(url_for('Result'))
    return redirect(url_for('Result'))

@app.route('/livestream')
def livestream():
    f = open('video.txt', 'w')
    f.write('http://192.168.137.185:8080/video')
    f.close()
    os.system('python detect.py')
    return redirect(url_for('Result'))

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
