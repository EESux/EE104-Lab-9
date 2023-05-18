# EE104-Lab-9
Name: Jeffrey Mattos-Arpilleda
Class: EE 104
Demonstration Link: 
Reference: https://youtu.be/qDUlGUP2cLA
https://sjsu.instructure.com/courses/1559910/modules

Here is the updated README user guide, including the custom Python game file you provided.

Risk Assessment Model:
his project aims to build a risk assessment model using the Random Forest Classifier algorithm. The model predicts the risk level of a particular event based on the given features.

Prerequisites
Make sure you have the following libraries installed:

pandas
scikit-learn
You can install these libraries using pip:

Copy code
pip install pandas scikit-learn
Getting Started
Follow the steps below to set up and run the risk assessment model:

Download the 'hmeq.csv' file containing the dataset.
Place the 'hmeq.csv' file in the same directory as your code files.
Running the Code
Import the required libraries:
python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
Read the data into a pandas DataFrame:
python
Copy code
data = pd.read_csv('hmeq.csv')
Perform data preprocessing:
One-hot encode the categorical columns:
python
Copy code
data = pd.get_dummies(data)
Fill missing values with the mean of the respective column:
python
Copy code
data = data.fillna(data.mean())
Define the feature matrix X and target y:
python
Copy code
X = data.drop(columns=['BAD'])
y = data['BAD']
Note: If your data is structured differently, adjust the code accordingly.

Split the data into training and testing sets:
python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Initialize and train a RandomForestClassifier:
python
Copy code
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
Predict the risk groups on the test set:
python
Copy code
predictions = clf.predict_proba(X_test)
Define thresholds for the risk groups:
python
Copy code
low_risk_threshold = 0.33
medium_risk_threshold = 0.66
Add a new column with the risk group to the test set:
python
Copy code
X_test['risk_group'] = ['low risk' if pred[1] <= low_risk_threshold else 'medium risk' if pred[1] <= medium_risk_threshold else 'high risk' for pred in predictions]
Save the DataFrame to a new csv file:
python
Copy code
X_test.to_csv('risk_assessment.csv', index=False)
Make sure you have the 'hmeq.csv' file in the same directory as your code files before running the code. The output will be saved in the 'risk_assessment.csv' file.

Feel free to adjust the parameters and thresholds according to your specific requirements.

That's it! You can now run the code and perform risk assessments based on your dataset.

Risk Assessment Model Part 2:
This project aims to build a risk assessment model using the Random Forest Classifier algorithm. The model predicts the risk level of a particular event based on the given features.

Prerequisites
Make sure you have the following libraries installed:

pandas
scikit-learn
matplotlib
You can install these libraries using pip:

Copy code
pip install pandas scikit-learn matplotlib
Getting Started
Follow the steps below to set up and run the risk assessment model:

Download the 'hmeq.csv' file containing the dataset.
Place the 'hmeq.csv' file in the same directory as your code files.
Running the Code
Import the required libraries:
python
Copy code
import pandas as pd
import matplotlib.pyplot as plt
Read the data into a pandas DataFrame:
python
Copy code
data = pd.read_csv('risk_assessment.csv')
Count the number of instances in each risk group:
python
Copy code
risk_counts = data['risk_group'].value_counts()
Plot a bar chart:
python
Copy code
plt.figure(figsize=(10, 6))
plt.bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
plt.xlabel('Risk Group')
plt.ylabel('Count')
plt.title('Risk Group Distribution')
plt.show()
Make sure you have the 'risk_assessment.csv' file in the same directory as your code files before running the code. The code will generate a bar chart showing the distribution of instances in each risk group.

Feel free to adjust the chart's appearance or customize the code as needed.

That's it! You can now run the code and visualize the risk group distribution.

COVID-19 Total Cases Forecast:
This project aims to forecast the total number of COVID-19 cases using the ARIMA (Autoregressive Integrated Moving Average) model. The forecasted data will be compared with the actual data to evaluate the model's performance.

Prerequisites
Make sure you have the following libraries installed:

pandas
matplotlib
statsmodels
You can install these libraries using pip:

Copy code
pip install pandas matplotlib statsmodels
Getting Started
Follow the steps below to set up and run the COVID-19 total cases forecast:

Download the 'COVID-19_case_counts_by_date.csv' and 'COVID-19_case_counts_by_date_Full.csv' files containing the dataset.
Place the 'COVID-19_case_counts_by_date.csv' and 'COVID-19_case_counts_by_date_Full.csv' files in the same directory as your code files.
Running the Code
Import the required libraries:
python
Copy code
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
Import the datasets:
python
Copy code
data = pd.read_csv('COVID-19_case_counts_by_date.csv')
full_data = pd.read_csv('COVID-19_case_counts_by_date_Full.csv')
Convert the dates to pandas datetime format and set them as the index:
python
Copy code
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
full_data['Date'] = pd.to_datetime(full_data['Date'])
full_data.set_index('Date', inplace=True)
Get the COVID-19 case numbers:
python
Copy code
cases = data['Total_cases']
full_cases = full_data['Total_cases']
Fit the ARIMA model:
python
Copy code
model = ARIMA(cases, order=(5,1,0))
model_fit = model.fit()
Make a prediction for the next six months:
python
Copy code
start_index = len(cases)
end_index = start_index + 180  # predict the next six months
forecast = model_fit.predict(start=start_index, end=end_index)
Create a date range for the forecasted data:
python
Copy code
forecast_dates = pd.date_range(start=cases.index[-1], periods=len(forecast)+1)[1:]
Plot the actual data, forecasted data, and the full actual data:
python
Copy code
plt.plot(cases.index, cases, color='blue', label='Actual')
plt.plot(forecast_dates, forecast, color='red', label='Forecasted')
plt.plot(full_cases.index, full_cases, color='green', label='Full Actual')
Set labels and title:
python
Copy code
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.title('COVID-19 Total Cases Forecast vs Actual')
Show the legend and the plot:
python
Copy code
plt.legend()
plt.show()
Make sure you have the 'COVID-19_case_counts_by_date.csv' and 'COVID-19_case_counts_by_date_Full.csv' files in the same directory as your code files before running the code. The code will generate a plot showing the actual COVID-19 case data, the forecasted data, and the full actual data.

Feel free to customize the code or the plot's appearance as needed.

That's it! You can now run the code and visualize the COVID-19 total cases forecast.

Dance Dance Revolution:
This project is a Dance Dance Revolution (DDR) game implemented using the Pygame Zero framework. The game allows two players to compete by following a sequence of dance moves displayed on the screen.

Prerequisites
Make sure you have the following libraries installed:

Pygame Zero
You can install the Pygame Zero library using pip:

Copy code
pip install pgzero
Getting Started
Follow the steps below to set up and run the Dance Dance Revolution game:

Download the necessary image and music files for the game.
Place the image files in a directory called "images" and the music files in a directory called "music".
Save the code file with the ".py" extension in the same directory as the "images" and "music" directories.
Running the Code
Import the required libraries:
python
Copy code
import pgzrun
import pygame
import pgzero
import random
from pgzero.builtins import Actor
from random import randint
import os
Set the game's configuration parameters:
python
Copy code
WIDTH = 800
HEIGHT = 600
CENTER_X = WIDTH / 2
CENTER_Y = HEIGHT / 2
Define the game's variables and load the music files:
python
Copy code
move_list = []
display_list = []
score = 0
current_move = 0
count = 4
dance_length = 4
say_dance = False
show_countdown = True
moves_complete = False
game_over = False

music_dir = 'music'
music_files = [f for f in os.listdir(music_dir) if f.endswith('.ogg')]
selected_music = random.choice(music_files)
Set up the game's actors and their initial positions:
python
Copy code
dancer = Actor("dancer-start")
dancer.pos = CENTER_X - 70, CENTER_Y - 40
up = Actor("up")
up.pos = CENTER_X - 200, CENTER_Y + 110
right = Actor("right")
right.pos = CENTER_X - 140, CENTER_Y + 170
down = Actor("down")
down.pos = CENTER_X - 200, CENTER_Y + 230
left = Actor("left")
left.pos = CENTER_X - 260, CENTER_Y + 170

# Second player
dancer2 = Actor("dancer-start2")
dancer2.pos = CENTER_X + 70, CENTER_Y - 40
up2 = Actor("up")
up2.pos = CENTER_X + 200, CENTER_Y + 110
right2 = Actor("right")
right2.pos = CENTER_X + 260, CENTER_Y + 170
down2 = Actor("down")
down2.pos = CENTER_X + 200, CENTER_Y + 230
left2 = Actor("left")
left2.pos = CENTER_X + 140, CENTER_Y + 170
score2 = 0
current_move2 = 0
move_list2 = []
game_over2 = False
Implement functions to control the game's logic:
reset_dancer() and reset_dancer2(): Reset the dancer's and dancer2's images to their starting positions.
update_dancer(move) and update_dancer2(move): Update the dancer's and dancer2's images based on the given move.
display_moves(): Display the dance moves on the screen.
generate_moves(): Generate a sequence of dance moves for the players to follow.
countdown(): Implement a countdown before the dance moves are displayed.
next_move() and next_move2(): Update the current move for each player.
on_key_up(key): Handle key events when a player releases a key.
update(): Update the game's state.
draw(): Draw the game's elements on the screen.
restart(): Restart the game.
Start the game:
python
Copy code
pgzrun.go()
Make sure you have the necessary image and music files in the correct directories before running the code. The code will start the Dance Dance Revolution game, allowing two players to compete by following the displayed dance moves.

Feel free to customize the game's appearance, add more dance moves, or modify the logic to suit your preferences.

That's it! You can now run the code and enjoy the Dance Dance Revolution game.
