import numpy as np
from flask import Flask, request, jsonify, render_template,Markup
import pickle
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
import io
import plotly.graph_objects as go
import pygal
from pygal.style import Style
from pprint import pprint

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
#@app.route('/plot.png')

def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    id = [int(x) for x in request.form.values()]

    id_current = id[0]
    # Extract the ids
    features = pd.read_csv('/application_train.csv')
    test_features2 = pd.read_csv('/application_test.csv')
    test_features = test_features2.loc[test_features2['SK_ID_CURR'] == id_current]
    #print(row)
    pprint(type(test_features))

    
    try :
        missing_values = (test_features.loc[test_features['SK_ID_CURR'] == id_current]).iloc[0].isnull().sum()/len(test_features.columns)
    except IndexError:
        value_error = True
        return render_template('home.html', value_error=value_error)

    age_sans_diffculte = (features.loc[features['TARGET'] == 0, 'DAYS_BIRTH'] // -365).head(30)
    age_sans_diffculte_list = age_sans_diffculte.tolist() 

    age_difficulte = (features.loc[features['TARGET'] == 1, 'DAYS_BIRTH'] // -365).head(30)
    age_diffculte_list = age_difficulte.tolist() 

    age_current = test_features.loc[test_features.index[0], 'DAYS_BIRTH'] // -365
    
    line_chart = pygal.Line()
    line_chart.title = 'Répartition de l\'âge'
    line_chart.x_labels = map(str, range(0, 1))
    line_chart.add('Pas de difficulté', age_sans_diffculte_list)
    line_chart.add('Difficulté', age_diffculte_list)
    line_chart.add('Age du client', [age_current])
    age_chart_data=line_chart.render_data_uri()

    encoding = 'ohe'
    n_folds = 3
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    # région de manière globale. 
    region = test_features.loc[test_features.index[0], 'REGION_RATING_CLIENT']

    # région de résidence avec prise en compte de la ville.
    ville = test_features.loc[test_features.index[0], 'REGION_RATING_CLIENT_W_CITY']
    
    pprint(region)
    pprint(ville)
        # Extract the labels for training
    labels = features['TARGET']

        # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])


        # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)

            # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)

            # No categorical indices to record
        cat_indices = 'auto'

        # Integer label encoding
    elif encoding == 'le':

            # Create a label encoder
        label_encoder = LabelEncoder()

            # List for storing categorical indices
        cat_indices = []

            # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                    # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                    # Record the categorical indices
                cat_indices.append(i)

        # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")

    #print('Training Data Shape: ', features.shape)
    #print('Testing Data Shape: ', test_features.shape)

        # Extract feature names
    feature_names = list(features.columns)

        # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

        # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)

        # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

        # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])

        # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])

        # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

        # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):

            # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
            # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

            # Create the model
        model = lgb.LGBMClassifier(n_estimators=50, objective = 'binary', 
                                    learning_rate = 0.1, 
                                    reg_alpha = 0.1, reg_lambda = 0.1, 
                                    subsample = 1, n_jobs = 3, random_state = 50)

            # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                    eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                    eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                    early_stopping_rounds = 100, verbose = 200)

        #model = pickle.load(open('model.pkl', 'rb'))

            # Record the best iteration
        best_iteration = model.best_iteration_

            # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits

            # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits

            # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]

            # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']

        valid_scores.append(valid_score)
        train_scores.append(train_score)

            # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

        # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

        # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)

        # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

        # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('Moyenne')

        # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 

    #final_features = [np.array(int_features)]
    #prediction = model.metrics
    Row =  submission
    print(submission.iloc[0]['TARGET'])
    #fig = Figure()
    #output = round(prediction[0], 2)
    #labels = 'Difficulté de paiement', 'Pas de difficulté'
    #sizes = [submission.iloc[0]['TARGET'], (1-submission.iloc[0]['TARGET'])]
    #explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    #fig1, ax1 = plt.subplots()
    #ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    #ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    pngImage = io.BytesIO()
    labels = ['Difficulté de paiement', 'Pas de difficulté']
    values = [submission.iloc[0]['TARGET'], (1-submission.iloc[0]['TARGET'])]


    #pngImageB64String = "data:image/png;base64,"
    #pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    #fig = Figure()
    #axis = fig.add_subplot(1, 1, 1)
    #axis.set_title("title")
    #axis.set_xlabel("x-axis")
    #axis.set_ylabel("y-axis")
    #axis.grid()
    #axis.plot(range(5), range(5), "ro-")
    #labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    #sizes = [15, 30, 45, 10]
    #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    #fig1, ax1 = plt.subplots()
    #ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
    #shadow=True, startangle=90)
    #ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Convert plot to PNG image
    #pngImage = io.BytesIO()
    #FigureCanvas(ax1).print_png(pngImage)
    
    # Encode PNG image to base64 string
    #pngImageB64String = "data:image/png;base64,"
    #pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    pie_chart = pygal.Pie(half_pie=True,legend_at_bottom=True)
    pie_chart.title = 'Prévisions sur les facilités de paiement (in %)'
    pie_chart.add('Pas Difficulté', (1-submission.iloc[0]['TARGET']))
    pie_chart.add('Difficulté', submission.iloc[0]['TARGET'])
    pie_chart_data=pie_chart.render_data_uri()


    pie_chart_missing = pygal.Pie(half_pie=True,legend_at_bottom=True)
    pie_chart_missing.title = '% de données manquantes'
    pie_chart_missing.add('% Value', (1-missing_values))
    pie_chart_missing.add('% Missing Value', (missing_values))
    pie_chart_data_missing=pie_chart_missing.render_data_uri()

    if(missing_values > 0.7):
        pourcentage_manquant = "VALEURS NON FIABLES"
    else:
        pourcentage_manquant = "VALEURS FIABLES"
    
    score = (1-submission.iloc[0]['TARGET']) * 100

    pprint(score)
    # 100038 -> 84% Score A 
    # 100168 -> 98% Score A 
    # 113410 -> Score C
    # 164766 -> Score D
    # 164819 -> Score B



    if score >= 0 and score < 20:
        score = "E"
    elif score >= 20 and score < 40:
        score = "D"
    elif score >= 40 and score < 60:
        score = "C"
    elif score >= 60 and score < 80: 
        score = "B"
    elif score >= 80 and score <= 100:
        score = "A"
    

    return render_template('blank.html', age_chart_data=age_chart_data, score=score, region=region, ville=ville, pourcentage_manquant=pourcentage_manquant, id_client=id_current, pie_chart_data=pie_chart_data, pie_chart_data_missing=pie_chart_data_missing)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)