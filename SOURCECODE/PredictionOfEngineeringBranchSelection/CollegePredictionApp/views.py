from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
from django.core.handlers.wsgi import WSGIRequest
import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


global uname, dataset, sc, rf_cls
accuracy = []
precision = []
recall = []
fscore = []
encoder = []
columns = ['gender', 'caste', 'region', 'branch', 'college']


def LoadDataset(request):
    if request.method == 'GET':
        global dataset
        dataset = pd.read_csv("CollegeDataset/Dataset.csv", nrows=2000)
        dataset.drop(["branch_code", "college_code"], axis=1, inplace=True)
        dataset.fillna(0, inplace=True)
        cols = dataset.columns
        output = '<table class="table table-striped table-hover text-center">'
        output += '<thead class="table-dark sticky-top" style="top:75px"><tr>'
        output += '<th>#</th>'
        for i in range(1, len(cols)):
            output += '<th>'+cols[i]+'</th>'
        output += "</tr></thead>"
        output += '<tbody class="table-group-divider">'
        dataset = dataset.values
        for i in range(dataset.shape[0]):
            output += "<tr>"
            for j in range(dataset.shape[1]):
                output += "<td>"+str(dataset[i, j])+"</td>"
            output += "</tr>"
        output += '</tbody></table>'
        dataset = pd.read_csv("CollegeDataset/Dataset.csv", usecols=[
                              'rank', 'gender', 'caste', 'region', 'branch', 'college'], nrows=2000)
        dataset.fillna(0, inplace=True)
        context = {'data': output,
                   'message': 'Engineering College Admission Dataset'}
        return render(request, 'AdminScreen.html', context)


def calculateMetrics(algorithm, predict, y_test):
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    a = accuracy_score(y_test, predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)


def TrainML(request):
    if request.method == 'GET':
        global dataset, encoder, accuracy, precision, recall, fscore, sc, columns, rf_cls
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()
        encoder.clear()
        sc = MinMaxScaler(feature_range=(0, 1))
        for i in range(len(columns)):
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(
                le.fit_transform(dataset[columns[i]].astype(str)))
            encoder.append(le)
        dataset1 = dataset.values
        X = dataset1[:, 0:dataset1.shape[1]-1]
        Y = dataset1[:, dataset1.shape[1]-1]
        X = sc.fit_transform(X)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2)
        X_train, X_test1, y_train, y_test1 = train_test_split(
            X, Y, test_size=0.1)

        rf_cls = RandomForestClassifier()
        rf_cls.fit(X_train, y_train)
        predict = rf_cls.predict(X_test)
        calculateMetrics("Random Forest", predict, y_test)

        svm_cls = SVC()
        svm_cls.fit(X_train, y_train)
        predict = svm_cls.predict(X_test)
        calculateMetrics("SVM", predict, y_test)

        dt_cls = DecisionTreeClassifier()
        dt_cls.fit(X_train, y_train)
        predict = dt_cls.predict(X_test)
        calculateMetrics("Decision Tree", predict, y_test)

        cols = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE']
        output = '<table class="table table-striped table-hover text-center">'
        output += '<thead class="table-dark sticky-top" style="top:75px"><tr>'
        for col in cols:
            output += '<th>'+col+'</th>'
        output += '</tr></thead>'

        algorithm = ['Random Forest', 'SVM', 'Decision Tree']
        output += '<tbody class="table-group-divider">'
        for i in range(len(algorithm)):
            output += "<tr><td>%s</td>" % (algorithm[i])
            output += "<td>%.3f</td>" % (accuracy[i])
            output += "<td>%.3f</td>" % (precision[i])
            output += "<td>%.3f</td>" % (recall[i])
            output += "<td>%.3f</td>" % (fscore[i])
        output += '</tbody></table>'
        context = {'data': output, 'message': "Model Training Results"}
        return render(request, 'AdminScreen.html', context)


def PredictCollege(request):
    if request.method == 'GET':
        return render(request, 'PredictCollege.html', {})


def PredictCollegeAction(request):
    if request.method == 'POST':
        global dataset, sc, rf_cls, encoder
        rank = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        caste = request.POST.get('t3', False)
        region = request.POST.get('t4', False)
        branch = request.POST.get('t5', False)

        testData = [int(rank), gender, caste, region, branch]
        temp = []
        temp.append(testData)
        temp = np.asarray(temp)
        print(temp.shape)
        df = pd.DataFrame(
            temp, columns=['rank', 'gender', 'caste', 'region', 'branch'])
        for i in range(len(encoder)-1):
            df[columns[i]] = pd.Series(
                encoder[i].transform(df[columns[i]].astype(str)))

        df = df.values
        df = sc.transform(df)
        predict = rf_cls.predict(df)
        print(predict)
        predict = encoder[4].inverse_transform(predict)
        context = {'prediction': predict[0]}
        return render(request, 'UserScreen.html', context)


def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})


def AdminLogin(request):
    if request.method == 'GET':
        return render(request, 'AdminLogin.html', {})


def UserLogin(request):
    if request.method == 'GET':
        return render(request, 'UserLogin.html', {})


def Signup(request):
    if request.method == 'GET':
        return render(request, 'Signup.html', {})


def AdminLoginAction(request):
    global uname
    if request.method == 'POST':
        username: str = request.POST.get('t1', False)
        password: str = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            uname = username
            context = {
                'message': f'Welcome {username.title()}', }
            return render(request, 'AdminScreen.html', context)
        else:
            if username == 'admin':
                error = "Please enter correct password"
            else:
                error = "Please enter a valid name"
            context = {'error': error}
            return render(request, 'AdminLogin.html', context)


def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                              password='root123', database='CollegePrediction', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break
        if index == 1:
            context = {'message': 'Welcome '+uname.title()}
            return render(request, 'UserScreen.html', context)
        else:
            context = {'error': 'Login Failed.'}
            return render(request, 'UserLogin.html', context)


def SignupAction(request: WSGIRequest):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        error = ""
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root',
                              password='root123', database='CollegePrediction', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    return render(request, 'Signup.html', {
                        "error": f"The user {username} already exists."})
        if error == "":
            db_connection = pymysql.connect(
                host='127.0.0.1', port=3306, user='root', password='root123', database='CollegePrediction', charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,gender,email,address) VALUES('" + \
                username+"','"+password+"','"+contact+"','" + \
                gender+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                message = 'Signup Process Completed'
            context = {'message': message}
        return render(request, 'Signup.html', context)
