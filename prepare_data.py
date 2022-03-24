import os
if not os.path.isdir('datasets'):
    import requests

    os.makedirs('datasets/fer2013')

    url = 'https://www.dropbox.com/s/tntubvc72bilk1q/fer2013.csv?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Data/fer2013/fer2013.csv', 'wb').write(r.content)
    print("Success download data")
