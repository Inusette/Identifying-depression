import json
import urllib2
import sys


def get_submissions(url):
    # Gets the JSON file at the given url and returns the relevant
    # part of it as python object

    # script "identifier"
    headers = {'User-Agent': 'depression'}

    # create new request here to open the url with above set headers
    req = urllib2.Request(url, None, headers)

    # Open url and make data variable equal whatever we get
    data = urllib2.urlopen(req).read()

    # Load json from variable defined above as "data", it should be string/text that we get from reddit
    # turns it into python object
    json_data = json.loads(data)

    # The file currently contains many fields, the data we need is in data/children keys
    return json_data["data"]["children"]


# make sure there's a correct number of arguments
if len(sys.argv) is not 2:
    print('Data_collector.py <subreddit page url>')
    sys.exit(2)


# URL of the page that we want to fetch (read the argument)
url = str(sys.argv[1])

# "https://www.reddit.com/r/depression/.json"

# get the data from the reddit JSON file
j_data = get_submissions(url)

# iterate each post in the json data
for post in j_data:
    # extract the author, the title and the text of the post
    title = post['data']['title']
    author = post['data']['author']
    text = post['data']['selftext']

    # create a file with the post's author's nickname
    filename = './reddit_data/' + author + '.txt'

    # open the file to write
    post_file = open(filename, 'w')

    # write the title on the first line and then the post itself
    post_file.write(title.encode('utf-8') + '\n' + text.encode('utf-8'))

    # close the file
    post_file.close()
