#ML model to predict WAR (wins against replacement) using MLB stats (https://www.youtube.com/watch?v=ZO3HAVm9IdQ&t=10s)
import os
import pandas as pd
import numpy as np
from pybaseball import batting_stats
from pybaseball import statcast
import math as m
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
'''
start = 2002
end = 2022

df = pd.read_csv('batting.csv') #this is post downloading from pybaseball, switch to read

batting = df.groupby('IDfg', group_keys = False).filter(lambda x: x.shape[0] > 1) #not working same as tutorial, but filtering something, diff csv?

def next_season(player):
    player = player.sort_values('Season')
    player['Next_WAR'] = player['WAR'].shift(-1)
    return(player)

batting = batting.groupby('IDfg', group_keys = False).apply(next_season)

#print(batting[['Name', 'Season', 'WAR', 'Next_WAR']])

null_count = batting.isnull().sum()
complete_cols = list(batting.columns[null_count == 0])
batting = batting[complete_cols + ['Next_WAR'].copy()]

#print(batting.dtypes[batting.dtypes == 'object']) finding out which columns are objects for later deletion, can only use int for ML
del batting['Dol']
del batting['Age Rng'] #should find how to complete these two rows in one

batting['Team'] = batting['Team'].astype('category').cat.codes #this switches the team identifier from str to int

batting_full = batting.copy()
batting = batting.dropna()

#now for ML regression model
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit

rr = Ridge(alpha = 1) #the alpha parameter goes from 0 --> 1 (I think), and the higher the parameter the greater reduction in overfitting, the closer to zero the more 'basic' the linear model??
split = TimeSeriesSplit(n_splits = 3) #three separate divisions of time series data
sfs = SequentialFeatureSelector(rr, n_features_to_select = 20, direction = 'forward', cv = split, n_jobs = 4)

removed_columns = ['Next_WAR', 'Name', 'IDfg', 'Season'] #don't want to include what we are trying to optimize ('Next_WAR') or any strings ('Name'), not sure about others
selected_columns = batting.columns[~batting.columns.isin(removed_columns)] #this basically takes all of our columns (batting.columns) and then takes out those we removed (what is ~?)

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
batting.loc[:, selected_columns] = scalar.fit_transform(batting[selected_columns]) #this series of three lines transforms the regression to levels of 0 --> 1, taking out negative values

#at this point the data has been scaled from 0 --> 1, can take a look below
#print(batting)
#print(batting.describe())

sfs.fit(batting[selected_columns], batting['Next_WAR']) #this fits the data, or rather picks the twenty greatest predictors under the ridge regression model
predictors = list(selected_columns[sfs.get_support()])
#above will return an array of column headers which most 'fit' for Next_WAR, if you wanted a full True/False array of all column headers just use sfs.get_support()
#but we put it into a list and assigned it to a variable 'predictors'


#typically one would use cross-validate the datasets to make predictions; i.e. using groups 1 & 2 to train the algorithm and make predictions on group 3, using groups 1 & 3 to train algo, make predictions on group 2, etc.; referenced in n_splits above
#but because this is time series, can only use past data to predict future and cannot randomly group without constraints
def backtest(data, model, predictors, start = 5, step = 1): #step is important because we want at least 5 years of data to train, adding one year step each time
    all_predictions = []
    years = sorted(batting['Season'].unique())
    for i in range(start, len(years), step):
        current_year = years[i]
        train = data[data['Season'] < current_year]
        test = data[data['Season'] == current_year]
        model.fit(train[predictors], train['Next_WAR'])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index = test.index)
        combined = pd.concat([test['Next_WAR'], preds], axis = 1) #this concatenation with axis = 1 just adds a separate column for our predictions
        combined.columns = ['actual', 'prediction']
        all_predictions.append(combined)
    return pd.concat(all_predictions) #note the default is to concat with axis = 0, which combines into one vertical dataframe, what we want here as opposed to above

#print(predictions)

def player_history(df):
    df = df.sort_values('Season')
    df['player_season'] = range(0, df.shape[0])

    df['war_corr'] = list(df[['player_season', 'WAR']].expanding().corr().loc[(slice(None), 'player_season'), 'WAR'])
    df['war_corr'].fillna(1, inplace = True)

    df['war_diff'] = df['WAR'] / df['WAR'].shift(1) #shift(1) brings the previous seasons value up to the current year
    df['war_diff'].fillna(1, inplace = True)
    df['war_diff'][df['war_diff'] == np.inf] = 1 #this replaces any infinite values captured by the line two above and replaces it with one
    return df

batting = batting.groupby('IDfg', group_keys = False).apply(player_history)

def group_averages(df):
    return df['WAR'] / df['WAR'].mean() #this helps to offset against any reduced appearances due to injury, etc.

batting['war_season'] = batting.groupby('Season', group_keys = False).apply(group_averages)

new_predictors = predictors + ['player_season', 'war_corr', 'war_diff', 'war_season']

predictions = backtest(batting, rr, new_predictors)

from sklearn.metrics import mean_squared_error

m_sq_err = mean_squared_error(predictions['actual'], predictions['prediction'])
print('\nThe mean squared error is',round(m_sq_err,2),'and its square root is', round(m.sqrt(m_sq_err),2))
print('The standard deviation of the wins above replacement is', round(batting['Next_WAR'].std(),2))
if m.sqrt(m_sq_err) < batting['Next_WAR'].std():
    print('Because the square root of the MSE is less than the standard deviation of the win against replacement statistic, the predictors have proven useful')
else:
    print('Because the square root of the MSE is more than the standard deviation of the win against replacement statistic, the predictors have not proven useful\n')

print(pd.Series(rr.coef_, index = new_predictors).sort_values()) #this should be added to dashboard as well as statistics and some form of print statement above

diff = predictions['actual'] - predictions['prediction']
merged = predictions.merge(batting, left_index = True, right_index = True)
merged['diff'] = (predictions['actual'] - predictions['prediction']).abs()
print(merged[['IDfg', 'Season', 'Name', 'WAR', 'Next_WAR', 'diff']].sort_values(['diff']))'''
#the above four lines just measures the absolute value of the difference between the actual WAR and our predicted WAR;
#if they have a big diff then the player is being systematically miscategorized; how to display in dashboard (bottom & top 10?)

##################################
#now for pitch stuff

#pybaseball.cache.enable() #how does this work?

df_2 = pd.read_csv('pitch_stuff.csv')
'''
df_3 = df_2.dropna(subset = ['release_spin_rate', 'release_extension', 'effective_speed'])
fig, ax = plt.subplots(figsize = (8, 8))
sns.despine(fig, left = True, bottom = True)
sns.scatterplot(x = 'release_spin_rate', y = 'release_extension',
                hue = 'effective_speed',
                palette = 'viridis',
                data = df_3,
                ax = ax)
ax.set_title('Effective Speed as a Function of Extension and Spin Rate')
plt.show()

df_3 = df_2.dropna(subset = ['release_speed', 'release_pos_x', 'release_pos_z'])
fig, ax = plt.subplots(figsize = (8, 8))
sns.despine(fig, left = True, bottom = True)
sns.scatterplot(x = 'release_pos_x', y = 'release_pos_z',
                hue = 'release_speed',
                palette = 'viridis',
                data = df_3,
                ax = ax)
ax.set_title('Release Speed as a function of Release Position')
plt.show()
'''
df_3 = df_2.dropna(subset = ['launch_speed', 'launch_angle', 'type_num'])
fig, ax = plt.subplots(figsize = (8, 8))
sns.despine(fig, left = True, bottom = True)
sns.scatterplot(x = 'launch_speed', y = 'launch_angle',
                hue = 'type_num',
                palette = 'viridis',
                data = df_3,
                ax = ax)
ax.set_title('Contact Result as a function of Launch Speed & Angle')
plt.show()
