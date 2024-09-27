# tickers.py

tickers = [
    'AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'PYPL', 'ADBE',
    'INTC', 'AMD', 'CRM', 'ORCL', 'CSCO', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT',
    'IBM', 'INTU', 'LRCX', 'XLNX', 'ADI', 'NVAX', 'SHOP', 'MRNA', 'BA', 'JPM',
    'V', 'MA', 'UNH', 'PFE', 'JNJ', 'PG', 'DIS', 'KO', 'PEP', 'COST',
    'HD', 'WMT', 'TGT', 'MCD', 'NKE', 'SBUX', 'GS', 'MS', 'BKNG', 'AXP'
]

# Provided list of tickers
tickers_250 = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'ADGI', 'ADI', 'ADSK', 'AGEN', 'AIG', 'AJG', 'ALGN', 'ALL', 'ALNY',
               'ALT', 'ALVR', 'AMAT', 'AMD', 'AMP', 'AMWL', 'AMZN', 'ANSS', 'AON', 'APDN', 'ARCT', 'ASAN', 'ASML', 'AVGO',
               'AXON', 'AXP', 'AYX', 'AZN', 'BA', 'BABA', 'BAX', 'BDX', 'BE', 'BIDU', 'BIGC', 'BILI', 'BILL', 'BK', 'BKNG',
               'BLI', 'BLK', 'BLNK', 'BNTX', 'BSX', 'CAT', 'CB', 'CDNS', 'CFLT', 'CHPT', 'CLDR', 'CLOV', 'CME', 'CMI',
               'COP', 'COST', 'CRM', 'CRWD', 'CSCO', 'CSIQ', 'CTMX', 'CTSH', 'CURE', 'CVAC', 'CVX', 'DDOG', 'DE', 'DGX',
               'DHR', 'DIS', 'DOCU', 'DOYU', 'DXC', 'ENPH', 'ETSY', 'EW', 'EXAS', 'F', 'FCEL', 'FDX', 'FIS', 'FISV', 'FLGT',
               'FLT', 'FSLR', 'FSLY', 'FSR', 'GD', 'GE', 'GH', 'GILD', 'GM', 'GOOG', 'GPN', 'GS', 'GSK', 'HD', 'HIG', 'HII',
               'HMC', 'HON', 'HPE', 'HPQ', 'HUYA', 'IBM', 'ICE', 'ILMN', 'INO', 'INTC', 'INTU', 'IQ', 'ISRG', 'JD', 'JNJ',
               'JPM', 'KLAC', 'KLXE', 'KMI', 'KO', 'LCID', 'LH', 'LI', 'LLY', 'LMT', 'LNC', 'LRCX', 'MA', 'MAXN', 'MCD',
               'MCO', 'MDB', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MOMO', 'MPWR', 'MRNA', 'MRVL', 'MS', 'MSFT', 'MU', 'NDAQ',
               'NET', 'NFLX', 'NIO', 'NKE', 'NKLA', 'NOC', 'NOW', 'NSTG', 'NTES', 'NTRS', 'NVAX', 'NVDA', 'NVTA', 'NXPI',
               'OKTA', 'ON', 'ONEM', 'ORCL', 'OSCR', 'OXY', 'PANW', 'PEP', 'PFE', 'PG', 'PGR', 'PINS', 'PLTR', 'PLUG', 'PRU',
               'PYPL', 'QCOM', 'QDEL', 'QRVO', 'QS', 'REGN', 'RGEN', 'RIVN', 'RKLB', 'ROKU', 'RTX', 'RUN', 'S', 'SBUX',
               'SCHW', 'SEDG', 'SHOP', 'SLB', 'SMAR', 'SNAP', 'SNOW', 'SNPS', 'SPCE', 'SPGI', 'SPLK', 'SPWR', 'SQ',
               'SRNE', 'SRPT', 'STT', 'STX', 'SWKS', 'SYK', 'T', 'TDOC', 'TEAM', 'TER', 'TGT', 'TM', 'TME', 'TMO', 'TROW',
               'TRV', 'TSLA', 'TSM', 'TTD', 'TWLO', 'TWTR', 'TXG', 'TXN', 'TXT', 'U', 'UNH', 'UPS', 'V', 'VEEV', 'VIR',
               'VRNS', 'VRSN', 'VXRT', 'VZ', 'WDAY', 'WDC', 'WMT', 'WRB', 'XLNX', 'XOM', 'XPEV', 'YI', 'YY', 'ZBH', 'ZI',
               'ZM', 'ZS']

def soft_and_dedup(tickers_list):
    # Sorting the tickers alphabetically
    sorted_tickers = sorted(tickers_list)

    # Replacing duplicates with placeholder tickers
    unique_tickers = []
    seen_tickers = set()

    for ticker in sorted_tickers:
        if ticker not in seen_tickers:
            unique_tickers.append(ticker)
            seen_tickers.add(ticker)

    return unique_tickers, len(unique_tickers)


