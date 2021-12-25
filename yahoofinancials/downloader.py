from yahoofinancials import YahooFinancials as YF
import os, json, csv, argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ticker_csv',default='',help='read multiple tickers from csv file')
parser.add_argument('--ticker',default='AAPL',help='single or multiple tickers')
parser.add_argument('--start_date',default='2018-08-20',help='starting date in YYYY-MM-DD format')
parser.add_argument('--end_date',default='2018-09-20',help='end date in YYYY-MM-DD format')
parser.add_argument('--time_interval',default='daily',help='daily, weekly or monthly')
parser.add_argument('--json_file',default='data.json',help='output as json file')

args = parser.parse_args()

if __name__=='__main__':
    if os.path.exists(args.ticker_csv):
        a = []
        with open(args.ticker_csv, "r") as f:
            lines = csv.reader(f, delimiter="\t")
            for line in lines:
                a.append(line[0].strip())
        print(','.join(a))
        print(len(a))
        print(a[0])
        a.append('^GSPC')
        yf = YF(a)
    else:
        yf = YF(args.ticker.split(','))
    tmp = yf.get_historical_price_data(args.start_date, args.end_date, args.time_interval)
    """
    new = {}

    for idx in tmp:
        new[idx] = {}
        for jdx in range(len(tmp[idx]['prices'])):
            tt = tmp[idx]['prices']
            for kdx in tt[0]:
                new[idx][kdx] = []
            count = 0
            for kdx in tt[0]:
                new[idx][kdx].append(tt[jdx][kdx])
    """
    with open(args.json_file, 'w') as out_file:
        json.dump(tmp, out_file)
    print('done.') 
