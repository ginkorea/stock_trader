import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.graph_objs as go
from connector.client import APIConnection
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import pytz


def connect():
    est_api_connection = APIConnection()
    return est_api_connection


def utc_to_eastern(utc_dt):
    eastern_tz = pytz.timezone('US/Eastern')
    return utc_dt.astimezone(eastern_tz)


def is_trading_hour(utc_timestamp):
    et_timestamp = utc_to_eastern(utc_timestamp)
    return datetime(1900, 1, 1, 9, 30).time() <= et_timestamp.time() <= datetime(1900, 1, 1, 16, 0).time()


def fetch_stock_data(api, ticker, start_date, end_date, timeframe):
    # Format start and end dates only if they don't already include a time part
    if 'T' not in start_date:
        start_date_formatted = f"{start_date}T00:00:00Z"
    else:
        start_date_formatted = start_date

    if 'T' not in end_date:
        end_date_formatted = f"{end_date}T23:59:59Z"
    else:
        end_date_formatted = end_date

    df = api.get_stock_data(ticker, start_date_formatted, end_date_formatted, timeframe)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).apply(utc_to_eastern)
    df['is_trading'] = df['timestamp'].apply(is_trading_hour)
    return df


def add_segment_to_fig(fig, segment_df, name, color, show_legend):
    fig.add_trace(go.Scatter(
        x=segment_df['timestamp'],
        y=segment_df['close'],
        mode='lines',
        line=dict(color=color),
        name=name,
        showlegend=show_legend
    ))


def create_figure(df, ticker):
    fig = go.Figure()
    last_val = None
    segment = []
    trading_legend_added = False
    non_trading_legend_added = False

    for index, row in df.iterrows():
        if last_val is not None and last_val != row['is_trading']:
            add_segment_to_fig(fig, pd.DataFrame(segment), 'Trading Hours' if last_val else 'Non-Trading Hours',
                               'blue' if last_val else 'red',
                               trading_legend_added if last_val else non_trading_legend_added)
            if last_val and not trading_legend_added:
                trading_legend_added = True
            if not last_val and not non_trading_legend_added:
                non_trading_legend_added = True
            segment = []
        segment.append(row)
        last_val = row['is_trading']
    if segment:
        add_segment_to_fig(fig, pd.DataFrame(segment), 'Trading Hours' if last_val else 'Non-Trading Hours',
                           'blue' if last_val else 'red',
                           trading_legend_added if last_val else non_trading_legend_added)

    fig.update_layout(
        title=f'{ticker} Stock Price',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='cyan'
    )
    return fig


def start():

    api = connect()

    d_ticker = "AAPL"
    d_start_date = "2023-01-01T00:00:00Z"
    d_end_date = "2023-01-06T23:59:59Z"
    timeframe = TimeFrame.Minute

    df = fetch_stock_data(api, d_ticker, d_start_date, d_end_date, timeframe)
    fig = create_figure(df, d_ticker)

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    app.layout = html.Div([
        html.Div([
            html.H1('Stock Data Visualization', style={'text-align': 'center', 'color': 'cyan'})
        ], style={'margin-bottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Ticker Symbol:"),
                dcc.Input(id='input_ticker', value='AAPL', type='text', style={'width': '100px'}),
                html.Label("Select Date Range:"),
                dcc.DatePickerRange(
                    id='input_date_range',
                    start_date=datetime(2023, 1, 1),
                    end_date=datetime(2023, 1, 6),
                    display_format='YYYY-MM-DD'
                )
            ], style={'width': '100%', 'display': 'inline-block'}),

        ], style={'margin-bottom': '20px'}),

        html.Div([
            dcc.Graph(id='stock_graph', figure=fig)
        ])
    ], style={'backgroundColor': '#000000', 'padding': '20px'})


    @app.callback(
        Output('stock_graph', 'figure'),
        [Input('input_ticker', 'value'),
         Input('input_date_range', 'start_date'),
         Input('input_date_range', 'end_date')]
    )
    def update_graph(ticker, start_date, end_date):
        update_api = connect()
        this_df = fetch_stock_data(update_api, ticker, start_date, end_date, TimeFrame.Minute)
        return create_figure(this_df, ticker)
    app.run_server(debug=True)
