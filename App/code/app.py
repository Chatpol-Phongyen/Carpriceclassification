# Import packages
from dash import Dash, html, callback, Output, Input, State, dcc, ctx
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash_bootstrap_components as dbc
import pickle
import warnings
warnings.simplefilter("ignore")

# Import csv file
df = pd.read_csv("/root/code/Cars.csv")

# Split mileage, max_power into value and number
df[["mileage_value","mileage_unit"]] = df["mileage"].str.split(pat=' ', expand = True)
df[["max_power_value","max_power_unit"]] = df["max_power"].str.split(pat=' ', expand = True)
df.drop(["mileage","max_power"], axis=1, inplace=True)

# Filter dataframe not to include LPG and CNG in fuel column
df = df.loc[(df["fuel"] != 'LPG') & (df["fuel"] != 'CNG')]

# convert mileage, max_power from string to float64
df[["mileage" ,"max_power"]] = df[["mileage_value","max_power_value"]].astype('float64')
df.drop(["mileage_value","max_power_value",
        "mileage_unit","max_power_unit"], axis=1, inplace = True)

# Dicard dataframe containing test drive car in owner column
df = df[df["owner"] != 'Test Drive Car']

# Prepare chosen features
df["log_km_driven"] = np.log(df["km_driven"])
features = df[["max_power","mileage","log_km_driven", "year"]]

# Fill in missing data in features
features_max_power_median = features['max_power'].median()
features_mileage_mean = features['mileage'].mean()
features['max_power'].fillna(features_max_power_median, inplace=True)
features['mileage'].fillna(features_mileage_mean, inplace=True)

# Prapare pattern of transformation for dataset
scaler = StandardScaler()
load_model = pickle.load(open("/root/code/model (1).pkl", 'rb')) # Import new model


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.JOURNAL]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Create app layout
app.layout = html.Div([
        html.H1("Selling car price classification by logistic regression", style={'color':'blue', 'font-size': '50px', 'text-align': 'center', 'background-color': 'yellow','opacity': 0.5}),
        html.H1("", style={'background-color': 'red','opacity': 0.75}),
        html.Br(),
        html.H3("Welcome to third car prediction website", style={'color':'blue', 'font-size': '35px', 'text-decoration-line': 'underline'}),
        html.Br(),
        html.H6("This is where you can classify by putting numbers in parameters. Car price prediction depends on four features, including maximum power, mileage, kilometers driven, and year.", style={'font-size': '18px'}),
        html.H6("Firstly, you have to fill at least one input boxes, choose year and then click submit to get result below the submit button.", style={'font-size': '18px'}),
        html.H6("The car price class is displayed in 0,1,2 and 3. Please make sure that filled number are not negative.", style={'font-size': '18px'}),
        html.Br(),
        html.H3("Definition", style={'color':'blue', 'font-size': '35px', 'text-decoration-line': 'underline'}),
        html.Br(),
        html.H6("Maximum power: Maximum power of a car in bhp", style={'font-size': '17px'}),
        html.H6("Mileage: The fuel efficieny of a car or ratio of distance which a car could move per unit of fuel consumption measuring in km/l", style={'font-size': '17px'}),
        html.H6("Kilometers driven: Total distance driven in a car by previous owner in km", style={'font-size': '17px'}),
        html.H6("Year: Year of production", style={'font-size': '17px'}),
        html.Br(),
        html.Div(["Maximum power (0 - 500) ", dcc.Input(id = "max_power", type = 'number', min = 0, max = 500, step = 0.02, value = 70, placeholder="please insert")]), html.Br(),
        html.Div(["Mileage (0 - 80) ", dcc.Input(id = "mileage", min = 0, max = 100, step = 0.02, value = 30, type = 'number', placeholder ="please insert")]), html.Br(),
        html.Div(["Kilometers driven (0 - 10000000) ", dcc.Input(id = "km_driven", type = 'number', min = 0, max = 10000000, value = 20000, placeholder="please insert")]), html.Br(),
        html.Div([dbc.Label("Year", html_for = "Year"), 
                  dcc.Dropdown(id = "year", options = [{"label": "2011", "value": 2011},
                  {"label": "2012", "value": 2012}, {"label": "2013", "value": 2013},
                  {"label": "2014", "value": 2014}, {"label": "2015", "value": 2015},
                  {"label": "2016", "value": 2016}, {"label": "2017", "value": 2017},
                  {"label": "2018", "value": 2018}, {"label": "2019", "value": 2019},
                  {"label": "2020", "value": 2020}, {"label": "2021", "value": 2021},
                  {"label": "2022", "value": 2022}, {"label": "2023", "value": 2023}
            ])]), html.Br(),
        html.Button(id="submit", children="submit", className="me-1"),
        html.Div(id="output"),
        html.Br(),
        html.Div(id='expected_inputs'),
        html.Div(id='shape_output')
])
# Callback input and output
@callback(
    Output(component_id = "output", component_property = "children"),
    State(component_id = "max_power", component_property = "value"),
    State(component_id = "mileage", component_property = "value"),
    State(component_id = "km_driven", component_property = "value"),
    State(component_id = "year", component_property = "value"),
    Input(component_id = "submit", component_property = "n_clicks"),
    prevent_initial_call=True
)

# Function for finding estimated car price
def prediction (max_power, mileage, km_driven, year, submit):
    if max_power == None:
        max_power = features_max_power_median # Fill in maximum power if dosen't been inserted
    if mileage == None:
        mileage = features_mileage_mean # Fill in mileage if dosen't been inserted
    if km_driven == None:
        km_driven = math.exp(features["km_driven"].median()) # Fill in kilometers driven if doesn't been inserted
    if year == None:
        year = features["year"].median() # Fill in year if doesn't been inserted
    sample = [[max_power, mileage, math.log(km_driven), year]]
    scaler.fit(features) #make standard scale for dataset
    sample = scaler.transform(sample) # transform standard scal for samples
    result = load_model.predict(sample) #class of car price
    return f"Your car price is in class {np.int64(result)[0]}"

@callback(
    Output('expected_inputs', 'children'),
    Input(component_id = "max_power", component_property = "value"),
    Input(component_id = "mileage", component_property = "value"),
    Input(component_id = "km_driven", component_property = "value"),
    Input(component_id = "year", component_property = "value"))

def input(max_power, mileage, km_driven, year):
    return f'max_power: {max_power}, mileage: {mileage}, km_driven: {km_driven}, year: {year}'


@callback(
    Output('shape_output','children'),
    Input(component_id = "max_power", component_property = "value"),
    Input(component_id = "mileage", component_property = "value"),
    Input(component_id = "km_driven", component_property = "value"),
    Input(component_id = "year", component_property = "value"),
    prevent_initial_call=True)

def size(max_power, mileage, km_driven, year):
    if max_power == None:
        max_power = features_max_power_median # Fill in maximum power if dosen't been inserted
    if mileage == None:
        mileage = features_mileage_mean # Fill in mileage if dosen't been inserted
    if km_driven == None:
        km_driven = math.exp(features["km_driven"].median()) # Fill in kilometers driven if doesn't been inserted
    if year == None:
        year = features["year"].median() # Fill in year if doesn't been inserted
    sample = np.array([[max_power, mileage, math.log(km_driven), year]])
    scaler.fit(features) #make standard scale for dataset
    sample = scaler.transform(sample) # transform standard scal for samples
    result = load_model.predict(sample) #class of car price
    return f"Shape of output is {result.shape}"

    
if __name__ == '__main__':
    app.run(debug = True)