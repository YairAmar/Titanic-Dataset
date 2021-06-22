from optparse import OptionParser
import requests
from lxml import html
import pandas as pd

"""
******************************************************
********************* ASSIGNMENT *********************
******************************************************

1. create TitanicWebsite class, ctor parameters: url, port
     hint! construct dummy functions to see high-level program logic is
           working

2. implement fetch_all_entries, return value:
     array ->
          ['column name 1', 'column name 2', 'column name 3'],
          ['column value 11','column value 12', 'column value 13'],
          ['column value 21','column value 22', 'column value 23']
     hint! please take a look at request & lxml packages

3. implement fetch_readme, return value: string

******************************************************
******************************************************
******************************************************
"""


class TitanicWebsite:
    
    def __init__(self, url, port):
        self.port = port
        self.url = url

    def fetch_all_entries(self):
        pass

    def fetch_readme(self):
      pass


if __name__ == '__main__':
    # command line parser
    parser = OptionParser()
    parser.add_option('-a', '--address', help='address', dest='address', default='http://127.0.0.1')
    parser.add_option('-p', '--port', help='port', dest='port', default='8000')
    parser.add_option('-o', '--output', help='output file', dest='output', default='../data/titanic.csv')
    parser.add_option('-r', '--readme', help='readme file', dest='readme', default='../data/readme.md')
    (options, args) = parser.parse_args()

    # pulling data
    print(' *** pulling data from website...')
    tws = TitanicWebsite(options.address, options.port)
    entries = tws.fetch_all_entries()
    readme = tws.fetch_readme()

    # writing data out to a file
    df = pd.DataFrame.from_records(entries[1:], columns=entries[0]).drop(['PassengerId'], axis=1)
    df.to_csv(options.output, index=False)
    print(' *** data file written successfully!')

    with open(options.readme, 'w+') as o:
        o.write(readme)
        o.close()
        print(' *** readme file written successfully!')
