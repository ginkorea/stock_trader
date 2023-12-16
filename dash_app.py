import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.graph_objs as go
from connector.client import APIConnection
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import pytz
from data.helper import TickerList


class StockDashboardApp:

    def __init__(self):
        self.api = self.connect()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.ticker_list = TickerList().dict
        self.setup_layout()
        self.setup_callbacks()

    def connect(self):
        return APIConnection()

    def utc_to_eastern(self, utc_dt):
        eastern_tz = pytz.timezone('US/Eastern')
        return utc_dt.astimezone(eastern_tz)

    def is_trading_hour(self, utc_timestamp):
        et_timestamp = self.utc_to_eastern(utc_timestamp)
        return datetime(1900, 1, 1, 9, 30).time() <= et_timestamp.time() <= datetime(1900, 1, 1, 16, 0).time()

    def fetch_stock_data(self, ticker, start_date, end_date, timeframe):
        if 'T' not in start_date:
            start_date_formatted = f"{start_date}T00:00:00Z"
        else:
            start_date_formatted = start_date

        if 'T' not in end_date:
            end_date_formatted = f"{end_date}T23:59:59Z"
        else:
            end_date_formatted = end_date

        df = self.api.get_stock_data(ticker, start_date_formatted, end_date_formatted, timeframe)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).apply(self.utc_to_eastern)
        df['is_trading'] = df['timestamp'].apply(self.is_trading_hour)
        return df

    def add_segment_to_fig(self, fig, segment_df, name, color, show_legend):
        fig.add_trace(go.Scatter(
            x=segment_df['timestamp'],
            y=segment_df['close'],
            mode='lines',
            line=dict(color=color),
            name=name,
            showlegend=show_legend
        ))

    def create_figure(self, df, ticker):
        fig = go.Figure()
        last_val = None
        segment = []
        trading_legend_added = False
        non_trading_legend_added = False

        for index, row in df.iterrows():
            if last_val is not None and last_val != row['is_trading']:
                self.add_segment_to_fig(fig, pd.DataFrame(segment),
                                        'Trading Hours' if last_val else 'Non-Trading Hours',
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
            self.add_segment_to_fig(fig, pd.DataFrame(segment), 'Trading Hours' if last_val else 'Non-Trading Hours',
                                    'blue' if last_val else 'red',
                                    trading_legend_added if last_val else non_trading_legend_added)

        fig.update_layout(
            title=f'{ticker} Stock Price',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='cyan'
        )
        return fig

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Stock Data Visualization', style={'text-align': 'center', 'color': 'cyan'}),
            dcc.Dropdown(
                id='input_ticker',
                options=[{'label': v, 'value': k} for k, v in self.ticker_list.items()],
                value='AAPL'
            ),
            dcc.DatePickerRange(
                id='input_date_range',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 6),
                display_format='YYYY-MM-DD'
            ),
            dcc.Graph(id='stock_graph')
        ], style={'backgroundColor': '#000000', 'padding': '20px'})

    def setup_callbacks(self):
        @self.app.callback(
            Output('stock_graph', 'figure'),
            [Input('input_ticker', 'value'),
             Input('input_date_range', 'start_date'),
             Input('input_date_range', 'end_date')]
        )
        def update_graph(ticker, start_date, end_date):
            df = self.fetch_stock_data(ticker, start_date, end_date, TimeFrame.Minute)
            return self.create_figure(df, ticker)

    def run(self):
        self.app.run_server(debug=True)


if __name__ == '__main__':
    app = StockDashboardApp()
    app.run()
