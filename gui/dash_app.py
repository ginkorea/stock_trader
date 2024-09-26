import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime
import plotly.graph_objs as go
from portfolio.ticker import Ticker # Import the Ticker class
import pandas as pd

class StockDashboardApp:

    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Stock Data Visualization', style={'text-align': 'center', 'color': 'cyan'}),
            dcc.Input(
                id='input_ticker',
                type='text',
                value='AAPL',  # Default value for the input
                placeholder="Enter a stock ticker",
                style={'width': '50%', 'margin-bottom': '10px'}
            ),
            dcc.DatePickerRange(
                id='input_date_range',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 6),
                display_format='YYYY-MM-DD'
            ),
            html.Button('Search', id='search_button', n_clicks=0),
            dcc.Graph(id='stock_graph')
        ], style={'backgroundColor': '#000000', 'padding': '20px'})

    def setup_callbacks(self):
        @self.app.callback(
            Output('stock_graph', 'figure'),
            [Input('search_button', 'n_clicks')],
            [Input('input_ticker', 'value'),
             Input('input_date_range', 'start_date'),
             Input('input_date_range', 'end_date')]
        )
        def update_graph(n_clicks, ticker_symbol, start_date, end_date):
            if n_clicks > 0:  # Only fetch data after the button is clicked
                ticker = Ticker(ticker_symbol)  # Initialize the Ticker class
                df = ticker.fetch_stock_data(start_date, end_date)
                return self.create_figure(df, ticker_symbol)
            return go.Figure()  # Empty figure before search

    @staticmethod
    def add_segment_to_fig(fig, segment_df, name, color, show_legend):
        fig.add_trace(go.Scatter(
            x=segment_df['timestamp'],
            y=segment_df['close'],
            mode='lines',
            line=dict(color=color),
            name=name,
            showlegend=show_legend
        ))

    def create_figure(self, df, ticker_symbol):
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
            title=f'{ticker_symbol} Stock Price',
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='cyan'
        )
        return fig

    def run(self):
        self.app.run_server(debug=True)

if __name__ == '__main__':
    app = StockDashboardApp()
    app.run()
