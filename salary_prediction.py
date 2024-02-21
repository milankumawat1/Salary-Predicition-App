from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st

st.header("Salary Prediction WebApp by Milan Kumawat")
st.sidebar.header("User Input")

exp = st.sidebar.slider("No. of experience", 0, 15, 4)
cgpa = st.sidebar.slider("CGPA Score", 0, 10, 4)
age = st.sidebar.slider("Age", 18, 60, 20)
interview_score = st.sidebar.slider("Interview Score", 0, 100, 50)
ds = pd.read_csv(
    'C:/Users/milan/Documents/AI-ML LNB/ML/20 july 2023/salary_predict_dataset.csv')
print(ds)
x = ds.iloc[:, :4]
y = ds.iloc[:, 4:]
# print(x)
# print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)


lr = LinearRegression()

lr.fit(x_train, y_train)
pred = lr.predict([[exp, cgpa, age, interview_score]])
y_pred = lr.predict(x_test)

st.subheader("Predicted Salary")
st.write(pred)


st.write(r2_score(y_test, y_pred))
