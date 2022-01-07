import requests
import pandas as pd

def load_FRED_data(series_ids, params):

    base_url = 'https://api.stlouisfed.org/fred/series/observations'

    responses = []

    for serie in series_ids:
        params['series_id'] = serie
        print(f'Downloading FRED series {serie}...', end =" ")
        response = requests.get(base_url, params)
        df = pd.DataFrame(response.json()['observations'])
        df = df[['date', 'value']]
        df['value'] = pd.to_numeric(df['value'], errors='coerce')    
        df.rename(columns={'date': 'date', 'value':serie}, inplace=True)
        df.set_index('date', inplace=True)
        print('Done')
        responses.append(df)
        
    data = pd.concat(responses, axis=1)

    return data