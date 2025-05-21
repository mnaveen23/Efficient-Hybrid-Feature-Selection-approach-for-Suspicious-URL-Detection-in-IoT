#importing required libraries

from flask import Flask, request, render_template,session,redirect, url_for
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
from sklearn.ensemble import GradientBoostingClassifier
import pickle
warnings.filterwarnings('ignore')
from feature import FeatureExtraction
import joblib

file = open("pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()


app = Flask(__name__)
app.secret_key = '371023ed2754119d0e5d086d2ae7736b'
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/submit_signup', methods=['POST'])
def submit_signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    return render_template('signup_success.html', username=username)


@app.route('/submit_login', methods=['POST'])
def submit_login():
    username = request.form.get('username')
    password = request.form.get('password')
    session['username'] = username
    return render_template('dashboard.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/phishing-url-detection', methods=['GET', 'POST'])
def phishing_url_detection():
    return render_template('Phishing-URL-Detection.html')




@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

data = pd.read_csv("phishing.csv")
data.head()
data = data.drop(['Index'],axis = 1)

X = data.drop(["class"],axis =1)
y = data["class"]
# Splitting the dataset into train and test sets: 80-20 split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.ensemble import GradientBoostingClassifier

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X_train,y_train)
#predicting the target value from the model for the samples
y_train_gbc = gbc.predict(X_train)
y_test_gbc = gbc.predict(X_test)
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 
        

        y_pred=gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred == 1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)