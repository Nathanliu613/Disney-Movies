# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:17:23 2021

@author: natha
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress most warnings in TensorFlow
from pandas.api.types import is_numeric_dtype
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import altair as alt
import streamlit as st
import calendar

st.title("Disney Movies!")

df = pd.read_csv("disney_movies.csv", na_values = " ")
df["release_date"] = pd.to_datetime(df["release_date"])
df['release_year']=pd.to_datetime(df['release_date']).dt.year
df['release_month']=pd.to_datetime(df['release_date']).dt.month.apply(lambda x:calendar.month_abbr[x])

st.write("*Here is what the data looks like*")

df

st.write("*This is the total amount of each genre of movie*")

st.write(df['genre'].value_counts())

st.write("*Here's the list of movies for the selected genre*")

movie_genre = st.selectbox('choose a genre', ("Comedy","Adventure",
"Drama","Action","Thriller/Suspense", "Romantic Comedy", "Documentary", "Musical",
"Western", "Horror", "Black Comedy", "Concert/Performance"))

st.write(df[df["genre"] == movie_genre])

st.write("*Make your own graph!*")

brush = alt.selection_interval(empty='none')
x_axis = st.selectbox('Choose an x-value',df.columns)
y_axis = st.selectbox('Choose a y-value',df.columns)
colors = st.selectbox("Choose a color",df.columns)
chart = alt.Chart(df).mark_circle().encode(
    x = x_axis,
    y = y_axis,
    color = colors
).properties(
    width = 720,
    height = 450,
)

st.altair_chart(chart)

st.write("*Now we will see if the genre has any effect on the total gross*")

chart2 = alt.Chart(df).mark_circle().encode(
    x = "genre",
    y = "total_gross",
    tooltip = ["mpaa_rating", "movie_title"],
    color = alt.condition(brush,
                          alt.Color('total_gross:Q', scale=alt.Scale(scheme='turbo',reverse=True)),
                          alt.value("lightgrey")),
).add_selection(
    brush,
).properties(
    width = 720,
    height = 450,
)

st.altair_chart(chart2)

st.write("*We can see that Adventure and Action have the highest gross in terms of individual movies*")

st.write("*Now we'll see if we can develop a neural network that can predict when a movie will be a Comedy*")

df = df[df.notna().all(axis=1)].copy()
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
scaler = StandardScaler()
scaler.fit(df[numeric_cols])
df[numeric_cols] = scaler.transform(df[numeric_cols])
def count_movies(g):
    return len(df[df["genre"].map(lambda g_list: g in g_list)])
all_genres = sorted(list(set(df["genre"].sum())))
df["is_Comedy"] = df["genre"].map(lambda g_list: "Comedy" in g_list)

X_train = df[numeric_cols]
y_train = df["is_Comedy"]

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (3,)),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(100, activation="sigmoid"),
        keras.layers.Dense(10,activation="softmax")
    ]
)

model.compile(
    loss="binary_crossentropy", 
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    metrics=["accuracy"],
)

history = model.fit(X_train,y_train,epochs=50,validation_split = 0.9, verbose = 0)

st.write("*Here is the graph of the train model and the validation*")

fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')

st.pyplot(fig)

predictions = (model.predict(X_train))

st.write("*Here is the table of the predicted values*")

st.write(predictions)

st.write("*None of these values are above 0.5, so they think it's unlikely any of the movies will be comedy*")

st.title("References")

st.markdown("Idea of the project was taken from https://www.kaggle.com/igorristovski/exploring-walt-disney-movies-and-box-office-succes/notebook ")
st.markdown("Dataset source: https://www.kaggle.com/prateekmaj21/disney-movies/code")
st.markdown("Github Link: https://github.com/Nathanliu613/Disney-Movies")
