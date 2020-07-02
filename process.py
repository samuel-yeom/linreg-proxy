import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

def process_communities():
    '''
    Processes the Communities and Crimes dataset, found in the UCI machine
    learning repository. Each row corresponds to a community in the US.
    All features are already scaled to be between 0 and 1.
    Race-related features are removed from `X`.

    Returns
    -------
    X : 2-D numpy array with shape `(n,d)`
        input features; rows are data points and columns are features
    y : 1-D numpy array with shape `(n,)`
        response variable (violent crimes per 100k population)
    gender_m: None
        returned for compatibility with other data processing methods
    race_blk: 1-D numpy array with shape `(n,)`
        fraction of the population that is black minus the fraction of the
        population that is white
    colnames: 1-D numpy array with shape `(d,)`
        textual descriptions of each column in `X`
    '''
    
    ycolname = 'ViolentCrimesPerPop'
    racecols = ['racePctBlack',
                'racePctWhite',
                'racePctAsian',
                'racePctHisp',
                'whitePerCap',
                'blackPerCap',
                'indianPerCap',
                'asianPerCap',
                'otherPerCap',
                'hispPerCap']
    
    # drop community name as it isn't predictive
    df = pd.read_csv('data/communities-clean.csv').drop(['communityname'], axis=1)
    
    #extract race info and then drop race-related columns
    black_minus_white = df['racePctBlack'] - df['racePctWhite']
    black_minus_white = np.squeeze(black_minus_white.values)
    df = df.drop(racecols, axis=1)
    
    #y
    y = np.squeeze(df[ycolname].values)
    df = df.drop(ycolname, axis=1)
    
    #X and colnames
    X = df.values
    colnames = np.array(df.columns)
    
    X = X.astype(float)
    y = y.astype(float)
    gender_m = None
    race_blk = black_minus_white.astype(float)
    
    return X, y, gender_m, race_blk, colnames

def process_chicago_ssl():
    '''
    Processes the Chicago SSL data. Each row corresponds to a person, and the
    goal is to predict the SSL score, which is score used by the Chicago
    police to predict how likely the person is to be involved in a shooting,
    as a perpetrator or a victim. Filters the data such that each person is
    either male or female and either black or white.

    Returns
    -------
    X : 2-D numpy array with shape `(n,d)`
        input features; rows are data points and columns are features
    y : 1-D numpy array with shape `(n,)`
        response variable (SSL score)
    gender_m: 1-D numpy array with shape `(n,)`
        1 if the corresponding person is male; 0 if female
    race_blk: 1-D numpy array with shape `(n,)`
        1 if the corresponding person is black; 0 if white
    colnames: 1-D numpy array with shape `(d,)`
        textual descriptions of each column in `X`
    '''
    
    agecolname = 'PREDICTOR RAT AGE AT LATEST ARREST'
    gendercolname = 'SEX CODE CD'
    racecolname = 'RACE CODE CD'
    ycolname = 'SSL SCORE'
    
    df = pd.read_csv('data/chicago-ssl-clean.csv')
    
    #Convert the ages to integers
    def convert_age(age_str):
        try:
            age_int = int(age_str[0:2])
        except ValueError:
            if age_str == 'less than 20':
                age_int = 10
            else:
                raise ValueError(age_str)
        return age_int
    new_ages = df[agecolname].map(convert_age)
    df[agecolname] = new_ages
    
    #Filter out uncommon gender and race
    df = df[df[gendercolname].isin(['F', 'M'])]
    df = df[df[racecolname].isin(['WHI', 'BLK'])]
    
    #z
    gender = np.squeeze(df[gendercolname].values)
    race = np.squeeze(df[racecolname].values)
    df = df.drop([gendercolname, racecolname], axis=1)
    
    #y
    y = np.squeeze(df[ycolname].values)
    df = df.drop(ycolname, axis=1)
    
    #X and colnames
    X = df.values
    colnames = np.array(df.columns)
    for i in range(len(colnames)):
        colnames[i] = colnames[i].replace('PREDICTOR RAT ', '')
    
    #Use more specific dtypes (instead of object)
    X = X.astype(float)
    y = y.astype(int)
    gender_m = np.where(gender == 'M', 1, 0) #male
    race_blk = np.where(race == 'BLK', 1, 0) #black
    
    return X, y, gender_m, race_blk, colnames

def scale(x):
    '''
    Scales feature(s) to zero mean and unit sample variance.

    Parameters
    ----------
    x : 1-D or 2-D numpy array
        If 1-D, the whole input is scaled. If 2-D, each column is scaled
        separately.

    Returns
    -------
    output: 1-D or 2-D numpy array
        scaled numpy array with the same shape as `x`
    '''
    
    output = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=1)
    return output

def train_model(X, y):
    '''
    Trains a linear regression model and prints the root mean squared error.
    Returns the model.

    Parameters
    ----------
    X : 2-D numpy array with shape `(n,d)`
        input features; rows are data points and columns are features
    y : 1-D numpy array with shape `(n,)`
        response variable

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        linear regression model trained with `X` and `y`
    '''
    
    model = LinearRegression().fit(X, y)
    print('Model standard error: {:.6f}'.format(np.sqrt(mse(y, model.predict(X)))))
    return model
