import requests
import sys
import zipfile
import StringIO

# login credentials
creds = sys.argv[1:]

# the direct link to the Kaggle data set
kaggle_url = 'http://www.kaggle.com/c/flavours-of-physics/download/'
datasets = ['training.csv.zip', 'test.csv.zip', 'check_correlation.csv.zip', 
	    'check_agreement.csv.zip']

# Kaggle username and password
kaggle_info = {'UserName': creds[0], 'Password': creds[1]}

# login to kaggle and retrieve the data.
for dataset in datasets:
    r = requests.get(kaggle_url+dataset)
    r = requests.post(r.url, data=kaggle_info)

    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall()
