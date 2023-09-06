import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import requests

app = dash.Dash()
server = app.server
scaler=MinMaxScaler(feature_range=(0,1))
api_endpoint = 'https://api.twelvedata.com/time_series'

df= pd.read_csv("./stock_data.csv")

app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data',children=[
			html.Div([
				html.H2("Closing Prices",style={"textAlign": "center"}),
                dcc.Input(
                    id='search-bar',
                    type='text',
                    placeholder='Enter a search value',
                    value='TSLA'
                ),
                html.Button('Search', id='search-button'),
                html.Button('Load Graph', id='load-button'),
                html.Div(id='output'),
                dcc.Graph(
                    id="Closing_Data"
                )		
			])        		

        ]),
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7,
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                             '#FF7400', '#FFF400', '#FF0056'],
                                height=600,
                                title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)}",
                                xaxis={"title":"Date",
                                       'rangeselector': {'buttons': list([
                                            {'count': 1, 'label': '1M', 'step': 'month', 
                                             'stepmode': 'backward'},
                                            {'count': 6, 'label': '6M', 'step': 'month', 
                                             'stepmode': 'backward'},
                                            {'count': 1, 'label': 'YTD', 'step': 'year', 
                                             'stepmode': 'todate'},
                                            {'count': 1, 'label': '1Y', 'step': 'year', 
                                             'stepmode': 'backward'},
                                            {'step': 'all'}])},
                                      'rangeslider': {'visible': True}, 'type': 'date'},
                                yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                             '#FF7400', '#FFF400', '#FF0056'],
                                height=600,
                                title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)}",
                                xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure


@app.callback(
    Output(component_id='output', component_property='children'),
    Input(component_id='search-button', component_property='n_clicks'),
    State(component_id='search-bar', component_property='value')
)
def update_output_div(n_clicks, value):
    global api_params, data, df_nse, scaler, dataset, train, valid, model, inputs, X_test, closing_price, smoothed_predicted_prices
    api_params = {
    'symbol': 'TSLA',
    'interval': '1day',
    'outputsize': 5000,
    'apikey': 'YOUR_KEY' # replace with your API key from 12data
    }

    if(value != ''):
        api_params['symbol'] = value.upper()
    
    #api_params['symbol'] = value.upper()
    print(value)
    # make the API request
    response = requests.get(api_endpoint, params=api_params)

    #parse the response JSON data into a pandas dataframe
    data = response.json()['values']
    df_nse = pd.DataFrame.from_records(data)

    # set the column names
    df_nse.columns = ['Date', 'open', 'high', 'low', 'Close', 'volume']

    df_nse["Date"] = pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
    df_nse.index = df_nse['Date']

    data = df_nse.sort_index(ascending=True,axis=0)
    new_data = pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

    for i in range(0,len(data)):
        new_data["Date"][i]=data['Date'][i]
        new_data["Close"][i]=data["Close"][i]

    new_data.index=new_data.Date
    new_data.drop("Date",axis=1,inplace=True)

    dataset = new_data.values

    train = dataset[0:987,:]
    valid = dataset[987:,:]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])

    x_train,y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    model = load_model("saved_model.h5")

    inputs = new_data[len(new_data)-len(valid)-60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    smoothed_predicted_prices = pd.DataFrame(closing_price).rolling(window=20).mean().values.flatten()
    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = smoothed_predicted_prices
    return f'The stock data for {value.upper()} has been loaded successfully.'


@app.callback(
    Output('Closing_Data', 'figure'),
    [Input('load-button', 'n_clicks')],
    State(component_id='search-bar', component_property='value'))
def update_graph(n_clicks, value):
    print("update value :" ,value)

    figure={
        "data":[
            go.Scatter(
                x=valid.index,
                y=valid["Close"],
                name="Actual closing price",
                mode='lines+markers'
            ),
            go.Scatter(
                x=valid.index,
                y=valid["Predictions"],
                name="LSTM predicted closing price",
                mode='lines+markers'
            )
        ],
        "layout":go.Layout(
            title='Closing prices',
            xaxis={'title':'Date'},
            yaxis={'title':'Closing Rate'}
        )
    }

    if value is None:
        return figure

    return figure


if __name__=='__main__':
    app.run_server(debug=True)