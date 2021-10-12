import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def produce_indicators(ticker, period, interval):
    # get ticker data
    df = yf.Ticker(ticker).history(period=period, interval=interval)[map(str.title, ['open', 'close', 'low', 'high', 'volume'])]

    # calculate MACD values
    df.ta.macd(close='close', fast=12, slow=26, append=True)
    # calculate bollinger bands
    df.ta.bbands(close="Close", length=20, std=2, append=True)
    # calculate stochastic oscillator
    df.ta.stoch(high='High', low='Low', k=14, d=3, append=True)

    df.ta.sma(close='Close', length=50, append=True)
    df.ta.sma(close='Close', length=250, append=True)

    # Force lowercase (optional)
    df.columns = [x.lower() for x in df.columns]

    risk_df = pd.DataFrame()
    risk_df['risk'] = df['sma_50'] / df['sma_250']
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(risk_df)
    risk_df['risk_normalized'] = np_scaled

    # calculate RSI
    close_delta = df['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)

    # Use exponential moving average
    periods = 14
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    df['rsi'] = rsi
    df['30'] = 30
    df['70'] = 70

    df = df.iloc[14:]
    risk_df = risk_df.iloc[14:]


    # Construct a 2 x 1 Plotly figure
    fig = make_subplots(rows=5, cols=1, subplot_titles=("Candles w/ BBands", "MACD(12, 26)", "RSI(14)", "Stochastic Oscillator(14, 3)", "Risk Calculation"))

    # upper band Line
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['bbu_20_2.0'],
            line=dict(color='#51087E', width=1),
            name='upper band',
            # showlegend=False,
            legendgroup='1',

        ), row=1, col=1,
    )

    # middle Line
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['bbm_20_2.0'],
            line=dict(color='#51087E', width=1),
            name='middle band',
            # showlegend=False,
            legendgroup='2',

        ), row=1, col=1
    )
    # lower band Line
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['bbl_20_2.0'],
            line=dict(color='#51087E', width=1),
            name='lower band',
            # showlegend=False,
            legendgroup='3',

        ), row=1, col=1
    )

    # Candlestick chart for pricing
    fig.append_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#ff9900',
            decreasing_line_color='black',
            showlegend=False

        ), row=1, col=1
    )

    # Fast Signal (%k)
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['macd_12_26_9'],
            line=dict(color='#ff9900', width=2),
            name='macd',
            # showlegend=False,
            legendgroup='4',

        ), row=2, col=1
    )

    # Slow signal (%d)
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['macds_12_26_9'],
            line=dict(color='#000000', width=2),
            # showlegend=False,
            legendgroup='5',
            name='signal'
        ), row=2, col=1
    )

    # Colorize the histogram values
    colors = np.where(df['macdh_12_26_9'] < 0, '#000', '#ff9900')

    # Plot the histogram
    fig.append_trace(
        go.Bar(
            x=df.index,
            y=df['macdh_12_26_9'],
            name='histogram',
            legendgroup='6',
            marker_color=colors,

        ), row=2, col=1
    )

    # plot RSI
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi'],
            line=dict(color='#51087E', width=2),
            name='RSI',
            showlegend=False,

        ), row=3, col=1,
    )
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['30'],
            line=dict(color='#51087E', width=0),
            hoverinfo='none',
            opacity=0.1,
            # fill='tonexty'
            showlegend=False,

        ), row=3, col=1,
    )
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['70'],
            line=dict(color='#51087E', width=0),
            fill='tonexty',
            hoverinfo='none',
            opacity=0.1,
            showlegend=False,

        ), row=3, col=1,
    )

    # Fast Signal (%k)
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['stochk_14_3_3'],
            line=dict(color='#ff9900', width=2),
            name='fast',
            showlegend=False,
        ), row=4, col=1
    )
    # Slow signal (%d)
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['stochd_14_3_3'],
            line=dict(color='#000000', width=2),
            name='slow',
            showlegend = False,
        ), row=4, col=1
    )

    # plot risk
    fig.append_trace(
        go.Scatter(
            x=risk_df.index,
            y=risk_df['risk_normalized'],
            line=dict(color='#51087E', width=2),
            name='risk',
            showlegend=False,
            legendgroup='7',

        ), row=5, col=1,
    )

    # Add overbought/oversold levels
    fig.add_hline(y=0, col=1, row=2, line_color='#00008b', line_width=2, line_dash='dash')
    fig.add_hline(y=30, col=1, row=3, line_color='#00008b', line_width=2, line_dash='dash')
    fig.add_hline(y=70, col=1, row=3, line_color='#00008b', line_width=2, line_dash='dash')
    fig.add_hline(y=20, col=1, row=4, line_color='#00008b', line_width=2, line_dash='dash')
    fig.add_hline(y=80, col=1, row=4, line_color='#00008b', line_width=2, line_dash='dash')
    fig.add_hline(y=0.25, col=1, row=5, line_color='#00008b', line_width=2, line_dash='dash')
    fig.add_hline(y=0.75, col=1, row=5, line_color='#00008b', line_width=2, line_dash='dash')

    # Make it pretty
    layout = go.Layout(
        title=str(ticker),
        height=1500,
        plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=20,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )

    # Update options and show plot
    fig.update_layout(layout)
    fig.show()


def produce_linear_regression():
    df = yf.Ticker('BTC-USD').history(period='5y')[['Close']]
    df.ta.ema(close='Close', length=10, append=True)
    df = df.iloc[10:]
    print(df.head(10))
    # Split data into testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(df[['Close']], df[['EMA_10']], test_size=0.33, shuffle=False)

    # Create Regression Model
    model = LinearRegression()
    # Train the model
    model.fit(X_train, y_train)

    # Printout relevant metrics
    print('slope:', model.coef_)
    r_sq = model.score(X_train, y_train)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    pred = model.predict(X_test)
    df = df.iloc[len(df.index) - (len(pred)):]
    df['pred'] = pred
    print("Mean Absolute Error:", mean_absolute_error(y_test, pred))
    print("Coefficient of Determination:", r2_score(y_test, pred))

    fig = make_subplots(rows=1, cols=1)

    # price markers
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'], mode='markers',
            line=dict(color='#ff9900', width=0.5),
            name='Close',
            # showlegend=False,
            legendgroup='1',

        ), row=1, col=1
    )
    # ema Line
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA_10'],
            line=dict(color='#ff9900', width=1),
            name='ema',
            # showlegend=False,
            legendgroup='1',

        ), row=1, col=1
    )
    # prediction line
    fig.append_trace(
        go.Scatter(
            x=df.index,
            y=df.pred,
            line=dict(color='#FF0000', width=1),
            name='Prediction',
            # showlegend=False,
            legendgroup='2',

        ), row=1, col=1
    )
    # Make it pretty
    layout = go.Layout(
        plot_bgcolor='#efefef',
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=20,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )

    # Update options and show plot
    fig.update_layout(layout)
    fig.show()

    # Plot outputs
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


# produce_linear_regression()
produce_indicators('BTC-USD', '5y', '1d')
