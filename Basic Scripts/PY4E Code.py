#Conditional Exercises

x=5
if x>2:
    print('Bigger than 2')
    print('Still bigger')
print('Done with 2')

for i in range(5):
    print(i)
    if i>2:
        print('Bigger than 2')
    print('Done with i',i)
print('All done')

#Ex 04_01

def thing():
    print('Hello')
    print('Fun')

thing()
print('Zip')
thing()

def (greet(lang)):
    if lang == 'es':
        print('Hola')
    elif lang =='fr':
        print('Bonjour')
    else:
        print('Hello')
greet('en')
greet('es')
greet('fr')

#Ex 4.6 - How to compute pay with overtime, based on 45hrs + 10.5r

def computepay(h,r):
    if h<=40:
        pay=h*r
    elif h>40:
        pay=40*r+(h-40)*r*1.5
	return(pay)

hrs = input("Enter Hours:")
h = float(hrs)
rate = input("Enter rate:")
r = float(rate)
p = computepay(h,r)
print('Pay',p)

#indefinite loop iterations from last chapter; notice diff between the two @56/65

while True:
    line = raw_input('>')
    if line[0] == '#':
        continue
    if line == 'done':
        break
    print(line)
print('Done!')

while True:
    line = input('>')
    if line[0] == '#':
        continue
    if line == 'done':
        break
    print(line)
print('Done!')

#definite loop iterations

for i in [5,4,3,2,1]:
    print(i)
print('Blastoff!')

#basic loops; if you change 'larg' to 'small' and change direction of >, you get smallest so far

print('Before')
for thing in [9,41,12,3,74,15]:
    print(thing)
print('After')

largest_so_far = -1
print('Before', largest_so_far)
for the_num in [9,41,12,3,74,15]:
    if the_num>largest_so_far:
        largest_so_far=the_num
    print(largest_so_far, the_num)
print('After,largest_so_far')

>>> num = 0
>>> tot = 0.0
>>> while True:
...     sval = input('Enter a number: ')
...     if sval == 'done':
            break
        try:
            fval = float(sval)
        except:
            print('Invalid Input')
            continue
...     print(fval)
...     num = num + 1 #note this and below are accumulator patterns
...     tot = tot + fval
print('All done')
print(tot,num,tot/num) #cumulative, counter, average

largest = None
smallest = None

while True:
    num = input("Enter a number: ")
    if num == "done":
        break
    try:
        output = float(num)
    except:
        print('Invalid input')
        continue
    if smallest is None:
        smallest = output
    if output > largest:
        largest = output
    elif num < smallest:
        smallest = num
    print(output)

print("Maximum is", largest)
print("Minimum is", smallest)



#Starting 'Python Data Structures'

#Exercise 6.5, getting "0.8475" out of a string
text = "X-DSPAM-Confidence:    0.8475" #this is what was given
a = text.find(':')
b = text.find('5')
c = text[a+1:b+1]
d = float(c.lstrip())
print(d)

#Chapter 7 - Python Data Structures

#file length counter

fhand = open('mbox.txt')
count = 0
for line in fhand:
    count = count + 1
print ('Line Count:', count)
$ python open.py

#reading the whole file

fhand = open('mbox-short.txt')
inp = fhand.read()
print(len(inp))

#to search though text files for something

fhand = open('mbox-short.txt')
for line in fhand:
    if line.startswith('From: '):
        print(line)
#note that the print function adds a newline(">n"); so to avoid this, use rstrip
fhand = open('mbox-short,txt')
for line in fhand:
    line = line.rstrip()
    if line.startswith('From: '):
        print(line)
#can add 'continue' function after the startswith if you want to keep going when your condition not met

#can also use the 'if not' and 'continue' function

#Ex 07_01
fh = open('mbox-short.txt')

for = lx in fh:
    ly = lx.rstrip().upper()
    print(ly)

#Ex 07.02 - Getting the numerical average from a text which has numbers (here "X-DSPAM-Confidence: "is what precedes number)

count=0
s=0
fname = input("Enter file name: ")
fh = open(fname)
for line in fh:
    if not line.startswith("X-DSPAM-Confidence:") :
        continue
    else:
        count=count+1
        a=line.find('0')
        x=line[a:]
        s=s+float(x)
avg=s/count
print("Average spam confidence:",avg)

#Lists- to change part of the Lists
lotto = [2,14,26,41,63]
print(lotto)
lotto[2] = 28
print(lotto)
[2,14,28,41,63]

#range
print(range(4))
[0,1,2,3]
friends = ['Joseph', 'Glenn', 'Sally']
print(len(friends))
3
print(range(len(friends)))
[0,1,2]

friends = ['Joseph', 'Glenn', 'Sally']
for i in range(len(friends))
friend = friends[i]
print('Happy New Year:',friend)

#Two separate ways to calculate an average:

#1:
total = 0
count = 0
while True:
    inp = input('Enter a number: ')
    if inp == 'done' : break
    value = float(inp)
    total = total + value
    count = count + 1
average = total / count
print('Average:',count)

#2
numlist = list()
while True:
    inp = input('Enter a number: ')
    if inp == 'done' : break
    value = float(inp)
    numlist.append(value)
average = sum(numlist) / len(numlist)
print('Average:',average)

#splitting strings
abc = 'With three words'
stuff = abc.split()
print(stuff)
['With','three','words']
print(len(stuff))
3
print(stuff[0])
With

>>> abc = 'With three words'
>>> stuff = abc.split()
>>> print(stuff)
['With', 'three', 'words']
>>> for w in stuff:
...     print(w)
With
three
words

#if the target text is: From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008
#and if the goal is to get the host name out;
words = line(split)
email = words[1]
pieces = email.split('@')
print(pieces(1))
uct.ac.za



#example 8.4 - for each line in a file, split into a list of words, then check if the word is already in the list [unique] and if not append to the list, then short
fname = input("Enter a file name: ")
fh = open(fname)
data = []
for each in fh:
    words = each.split()
    for word in words:
        if word not in data:
            data.append(word)
print(sorted(data))


#Ex 8.5
fname = input("Enter file name: ")
if len(fname) < 1 : fname = "mbox-short.txt"
#opening the file
fh = open(fname)
count = 0
#to store the lines
data=[]
for each in fh:
    # To check whether the line have more than two elements space seperated
    if each.startswith("From") and len(each.split())>2:
        temp=each.split()
        data.append(temp[1])
for each in data:
    print(each)
print("There were", len(data), "lines in the file with From as the first word")


#Defining a Dictionary
ddd = dict()
ddd['age'] = 21
ddd['course'] = 182
print(ddd)
{'course' : 182, 'age' : 21}
ddd['age'] = 23
print(ddd)
{'course' : 182, 'age' : 23}

#how to run a historgram on Dictionary
counts = dict()
names = ['Bob', 'Gail', 'Bob', 'Frank', 'Gail']
for name in names :
    if name not in counts :
        counts[name] = 1
    else :
        counts[name] = counts[name] + 1
print(counts)

#but the quicker way to get histrogram is using "GET" function as follows:
counts = dict()
names = ['Bob', 'Gail', 'Bob', 'Frank', 'Gail']
for name in names :
    counts[name] = counts.get(name, 0) + 1    #note this does the same thing as the if/else statement above
print(counts)

#Full exercise of creating historgram of files
counts = dict()
print('Enter a line of text:')
line = input('')
words = line.split()
print('Words:',words)
print('Counting...')
for word in words:
    counts[word] = counts.get(word,0) + 1
print('Counts',counts)

>>> counts = {'chuck':1,'fred':42,'jan':100}
>>> for key in counts:
...     print(key,counts[key])

#in order to get all the KEY/VALUE pairs
jjj = {'chuck':1,'fred':42,'jan':100}
for K,V in jjj.items():
    print(K,V)


#Two nested loops for histogram file searches
name = input('Enter file:')
handle = open(name)

counts = dict()
for line in handle:
    words = line.split()
    for word in words:
        counts[word] = counts.get(word,0) + 1

bigcount = None
bigword = None
for word,count in counts.items():
    if bigcount is None or count > bigcount:
        bigword = word
        bigcount = count
print(bigword,bigcount)


#basic Tuple assignment
d = dict()
d['csev'] = 2
d['cwen'] = 4
for (k,v) in d.items():
    print(k,v)
csev 2
cwen 4
tups = d.items()
print(tups)
dict_items([('csev',2),('cwen',4)])

#Tuple comparisons
(0,1,2) < (5,1,2)
True

#Tuple sorts
d = {'a':10,'b':1,'c':22}
t = sorted(d.items())
t
[('a',10)('b',1)('c',22)]
for k,v in sorted(d.items()):
    print(k,v)
a 10
b 1
c 22

#to SORT via VALUE instead of KEY
c = {'a':10,'b':1,'c':22}
tmp = list()
for k,v in c.items():
    tmp.append((v,k))
print(tmp)
[(10,'a'),(22,'c'),(1,'b')]
tmp = sorted(tmp,reverse = TRUE)
print(tmp)
[(22,'c'),(10,'a'),(1,'b')]

#Example to get the x most common things, in this instance 10 most common
fhand = open('romeo.txt')
counts = dict()
for line in fhand:
    words = line.split()
    for word in words:
        counts[word] = counts.get(word,0) + 1
#Note there is a shorthand below for everything below this
lst = list()
for key,val in counts.items():
    newtup = (val,key)
    lst.append(newtup)
lst = sorted(lst,reverse=True)
for val,key in lst[:10]:
    print(key,val)

#the shorthand for the flipping of KEY/VALUE can be done as follows:
print(sorted([(v,k) for k,v in c.items()]))
#this is what's known as 'List Comprehension'


#Example 10.2 - stripping out text and printing histogram
fanme = input('Enter File: ')
if len(fname) < 1: fname = 'clown.txt'
hand = open(fnam)

di = dict()
for lin in hand:
    lin = lin.rstrip()
    wds = lin.split()
    for w in wds:
        di[w] = d.get(w,0) + 1
print(di)


#Regular Expression example for finding digits in text
import re
x = 'I am currently 45 years old, but will be 46 in 1 month'
y = re.findall('[0-9]+',x)
print(y)
['45','46','1']


#Ex: Extracting Data with Regular Expressions
import re
sum = 0
file = open('regex_sum_1549232.txt','r')
for line in file:
    numbers = re.findall('[0-9]+',line)
    if not numbers:
        continue
    else:
        for number in numbers:
            sum += int(number)
print(sum)
#output = 424737

#Ex of difference between re.search() and find()
hand = open('mbox-short.txt')
for line in hand:
    line = line.rstrip()
    if line.find('From:') >= 0:
        print(line)

#OR
import re
hand = open('mbox-short.txt')
for line in hand:
    line = line.rstrip()
    if re.search('From:', line):
        print(line)

#Ex of difference between re.search() and startswith()
hand = open('mbox-short.txt')
for line in hand:
    line = line.rstrip()
    if line.startswith('From:'):
        print(line)
#OR
import re
hand = open('mbox-short.txt')
for line in hand:
    line = line.rstrip()
    if re.search('^From:', line):
        print(line)

#When adding space after re.findall, you dictate when to start the regex rules
# from the text: From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008
y = re.findall('\S+@\S+',x)
print(y)
['stephen.marquard@uct.ac.za']
#OR
y = re.findall('^From (\S+@\S+)',x)
print(y)
['stephen.marquard@uct.ac.za']

#Just as a refresher, the beginner way is
# from the text: From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008
words = text.split()
email = words[1]
pieces = email.split('@')
print(pieces[1])
#OR
import re
y = re.findall('@([^ ]*)',text)
print(text)

#To change directory
print(os.getcwd())
path = 'C:/Users/Will Mason/Dropbox/School B/Linear Programming'
os.chdir(path)
print(os.getcwd())

#To make a socket connection, get vitals and print
import socket

mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.connect(('hostname',portnumber))
cmd = 'GET hostname&document HTTP\1.0/r/n/r/n'.encode()
mysock.send(cmd)

while True:
    data = mysock.recv(#of characters)
    if (len(data) < 1):
        break
    print(data.decode())
mysock.close()

#The socket work can be completed using the URLLIB function; same as above
imort urllib.request, urllib.parse, urllib.error
fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
for line in fhand:
    print(line.decode().strip())

#Using Beautiful Soup to parse html language
imort urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

url = input('Enter - ')
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, 'html.parser')
#the following gets all of the anchor tags
tags = soup('a')
for tag in tags:
    print(tag.get('href', None))

#in order to avoid SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#Ex 12.6
url = input('Enter - ')
html = urllib.request.urlopen(url, context=ctx).read()
soup = BeautifulSoup(html, 'html.parser')

#Ex called "Scraping HTML Data with BeautifulSoup"
from urllib import request
from bs4 import BeautifulSoup
html = request.urlopen('http://py4e-data.dr-chuck.net/comments_1549234.html').read()
soup = BeautifulSoup(html)
tags = soup('span')
sum = 0
for tag in tags:
     sum = sum + int(tag.contents[0])
print(sum)

#Ex of Following links in html using BeautifulSoup
import urllib.request, urllib.parse, urllib.errors
from bs4 import BeautifulSoup
import re
url = 'http://py4e-data.dr-chuck.net/known_by_Piotr.html'
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, 'html.parser')
numlist = list()
position = 18
repeat = 7
tags = soup('a')
while repeat -1 >=0:
    print('process round',repeat)
    target = tags[position - 1]
    print('target:',target)
    url = target.get('href',2)
    print('Current url',url)
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html,'html.parser')
    tags = soup('a')
    repeat = repeat - 1

#To parse through XML data, use things
#first is the XML Data (note the diff by putting <person> on a new line, or same line in 620 with <stuff>)
import xml.etree.ElementTree as ET
data = '''
<person>
    <name>Chuck</name>
    <phone type="intl">
        +1 734 303 4456
    </phone>
    <email hide='yes'/>
</person>'''
#now the python code to parse
tree = ET.fromstring(data)
print('Name:',tree.find('name').text)
print('Attr:',tree.find('email').get('hide'))

#Another example of XML --> python
import xml.etree.ElementTree as ET
input = '''<stuff>
    <users>
        <user x='2'>
            <id>001</id>
            <name>Chuck</name>
        </user>
        <user x='7'>
            <id>009</id>
            <name>Brent</name>
        </user>
    </users>
</stuff>'''
#now the python code to parse
stuff = ET.fromstring(input)
lst = stuff.findall('users/user')
print('User count:',len(lst))
for item in lst:
        print('Name',item.find('name').text)
        print('Id',item.find('id').text)
        print('Attribute',item.get('x'))

#Ex Extracting Data from xml
import urllib.request as ur
import xml.etree.ElementTree as ET
url = 'http://py4e-data.dr-chuck.net/comments_1549236.xml'
total_num = 0
sum = 0
print('Retrieving',url)
xml = ur.urlopen(url).read()
print('Retreived',len(xml),'characters')
tree = ET.fromstring(xml)
counts = tree.findall('.//count')
for count in counts:
    sum += int(count.text)
    total_num += 1
print('Count:',total_num)
print('Sum',sum)

#JSON Example #1 (unclear whether double apostrophe necessary, I think YES)
import json
data = '''
{
    "name" : "Chuck",
    "phone" : {
        "type" : "intl",
        "number" : "+1 734 303 4456"
    },
    "email" : {
        "hide" : "yes"
    }
}'''
info = json.loads(data)
print('Name:',info['name'])
print('Hide:',info['email']['hide'])

#JSON Example #2
import json
input = '''
[
    {"id" : "001",
    "x" : "2",
    "name" : "Chuck"
    },
    {"id" : "009",
    "x" : "7",
    "name" : "Chuck"
    }
]'''
info = json.loads(input) #this is your parsing for getting into python
print('User count:',len(info))
for item in info:
    print('Name',item['name'])
    print('Id',item['id'])
    print('Atrribute',item['x'])

#Geolocation example for JSON
import urllib.request, urllib.parse, urllib.error
import json

serviceurl = 'http://maps.googleapis.com/maps/api/geocode/json?' #appears to now be behind API key

while True:
    address = input('Enter location:')
    if len(address) < 1: break

    url = serviceurl + urllib.parse.urlencode({'address':address})

    print('Retrieving',url)
    uh = urllib.request.urlopen(url)
    data = uh.read().decode()
    print('Retreived',len(data),'characters')

    try:
        js = json.loads(data)
    except:
        js = None

    if not js or 'status' not in js or js['status'] != 'OK':
        print('====Failure to Receive====')
        print(data)
        continue

    print(json.dumps(js, indent=2)) #Note this 'pretty prints' the JSON with good indentation; opposite of loads;

    lat = js["results"][0]["geometry"]["location"]["lat"] #note this is drilling down into the layers of JSON
    lng = js["results"][0]["geometry"]["location"]["lng"]
    print('lat',lat,'lng',lng)
    location = js['results'][0]['formatted_address']
    print(location)

#Twitter API
import urllib.request, urllib.parse, urllib.error
import twurl
import json

TWWITTER_URL = 'https://api.twitter.com/1.1/friends/list.json'

while True:
    print('')
    acct = input('Enter Twitter Account:')
    if(len(acct) < 1:) break
    url = twurl.augment(TWITTER_URL, {'screen_name': acct, 'count': '5'})
    print('Retreiving',url)
    connection = urllib.request.urlopen(url)
    data = connection.read().decode()
    headers = dict(connection.getheaders())
    print('Remaining',headers['x-rate-limit-remaining'])
    js = json.loads(data)
    print(json.dumps(js, indent=4))

    for us in js['users']:
        print(u['screen_name'])
        s = u['status']['text']
        print('  ',s[:50])

#Worked Exercise - Extracing Data from json
import urllib.request as ur
import json

url = 'http://py4e-data.dr-chuck.net/comments_42.json'
print("Retrieving ", url)
data = ur.urlopen(url).read().decode()
print('Retrieved', len(data), 'characters')
json_obj = json.loads(data)

sum = 0
total_number = 0

for comment in json_obj["comments"]:
    sum += int(comment["count"])
    total_number += 1

print('Count:', total_number)
print('Sum:', sum)

#Worked Exercise - Calling a JSON API
import urllib.request, urllib.parse, urllib.error
import json
import ssl

api_key = False

if api_key is False:
    api_key = 42
    serviceurl = 'http://py4e-data.dr-chuck.net/json?'
else :
    serviceurl = 'https://maps.googleapis.com/maps/api/geocode/json?'

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

while True:
    address = input('Enter location: ')
    if len(address) < 1: break

    parms = dict()
    parms['address'] = address
    if api_key is not False: parms['key'] = api_key
    url = serviceurl + urllib.parse.urlencode(parms)

    print('Retrieving', url)
    uh = urllib.request.urlopen(url, context=ctx)
    data = uh.read().decode()
    print('Retrieved', len(data), 'characters')

    try:
        js = json.loads(data)
    except:
        js = None

    if not js or 'status' not in js or js['status'] != 'OK':
        print('==== Failure To Retrieve ====')
        print(data)
        continue

# print(json.dumps(js, indent=4))
    pid = js['results'][0]['place_id']
    print('Place id ',pid)

#To write classes in Python
class PartyAnimal:
        x = 0

        def party(self):
            self.x = self.x + 1
            print("So far",self.x)

an = PartyAnimal()

an.party() #this is equivalent to "PartyAnimal.party(an)" and will get basic counter return

#To find the capabilities or methods of the class, same as above, but instead of an.party()
print("Type",type(an))
print("Dir",dir(an))

#To use contructor/destructor:
class PartyAnimal:
        x = 0

        def __init__(self):
            print('I am constructed')

        def party(self):
            self.x = self.x + 1
            print("So far",self.x)

        def __del__(self):
            print('I am destructed',self.x)

an = PartyAnimal()
an.party()
an.party()
an = 42
print('an contains',an)

#Similar example to show instance variables
class PartyAnimal:
        x = 0
        name = ""
        def __init__(self,z):
            self.name = z
            print(self.name,"constructed")

        def party(self):
            self.x = self.x + 1
            print(self.name,"party count",self.x)
w = PartyAnimal("Will")
j = PartyAnimal("Jim")
w.party()
j.party()

#To extend a class through inheritance
class PartyAnimal:
    x = 0
    name = ""
    def __init__(self,nam):
        self.name = nam
        print(self.name,"constructed")
    def party(self):
        self.x = self.x + 1
        print(self.name,"party count",self.x)
class FootballFan(PartyAnimal): #So this adds to the class PartyAnimal everything that we define below in FootballFan
    points = 0
    def touchdown(self):
        self.points = self.points + 7
        self.party()
        print(self.name,"points",self.points)

#SQL example 1
INSERT INTO Users(name,email) VALUES('Kristin','kf@umich.edu')
DELETE FROM Users WHERE email='fred@umich.edu'
UPDATE Users SET name='Charles' WHERE email='csev@umich.edu'
SELECT*FROM Users WHERE email='csev@umich.edu'
SELECT*FROM Users ORDER BY email

#SQL example 2
import sqlite3

conn = sqlite3.connect('emaildb.sqlite')
cur = conn.cursor()

cur.execute('''DROP TABLE IF EXISTS Counts''') #these two lines are merely to delete if already exists, which it does not

cur.execute('''CREATE TABLE Counts(org TEXT, count INTEGER)''')

fname = input('Enter file name: ')
if(len(fname) < 1): fname = 'mbox-short.txt'
fh = open(fname)
for line in fh:
    if not line.startswith('From: '): continue
    pieces = line.split()[1]
    org = pieces.split('@')[1]
    cur.execute('SELECT count FROM Counts WHERE org = ? ', (org, )) #note the question mark is placeholder to avoid mistakes
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (org, count)
            VALUES(?,1)''', (org,))
    else:
        cur.execute('UPDATE Counts SET count = count + 1 WHERE org = ?', (org,))

conn.commit()

sqlstr = 'SELECT org, count FROM Counts ORDER BY count DESC LIMIT 10'

for row in cur.execute(sqlstr):
    print(str(row[0]))
    print(str(row[1]))

cur.close()

#SQL work in connection with Itunes XML data
import xml.etree.ElementTree as ET
import sqlite3

conn = sqlite3.connect('trackdb.sqlite')
cur = conn.cursor

cur.executescript('''
DROP TABLE IF EXISTS Artist;
DROP TABLE IF EXISTS Album;
DROP TABLE IF EXISTS Track;

CREATE TABLE Artist (
    id          INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name        TEXT UNIQUE
);

CREATE TABLE Album (
    id          INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    artist_id   INTEGER,
    title       TEXT UNIQUE
);

CREATE TABLE Track (
    id          INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    title       TEXT UNIQUE,
    album_id    INTEGER,
    len         INTEGER, rating INTEGER, count INTEGER
);
''')

fname = input('Enter file name: ')
if (len(fname) < 1) : fname = 'ItunesLib.xml'

def lookup(d, key):
    found = False
    for child in d:
        if found : return child.text
        if child.tag == 'key' and child.text == key :
            found = True
    return None

stuff = ET.parse(fname)
all = stuff.findall('dict/dict/dict')
print('Dict count:',len(all))
for entry in all:
    if(lookup(entry, 'Track ID') is None) : continue

    name = lookup(entry, 'Name')
    artist = lookup(entry, 'Artist')
    album = lookup(entry, 'Album')
    count = lookup(entry, 'Play Count')
    rating = lookup(entry, 'Rating')
    length = lookup(entry, 'Total Time')

    if name is None or artist is None or album is None:
        continue

    print(name, artist, album, count, rating, length)

    cur.execute('''INSERT OR IGNORE INTO Artist (name) VALUES ( ? )''', (artist, ))
    cur.execute('SELECT id FROM Artist WHERE name = ? ', (artist, ))
    artist_id = cur.fetchone()[0]
    cur.execute('''INSERT OR IGNORE INTO Album (title, artist_id) VALUES (?, ?)''', (album, artist_id))
    cur.execute('SELECT id FROM Album WHERE title = ? ', (album, ))
    album_id = cur.fetchone()[0]
    cur.execute('''INSERT OR REPLACE INTO Track (title, album_id, len, rating, count) VALUES ( ?, ?, ?, ?, ?)''', (name, album_id, length, rating, count))

conn.commit()

#Example of many to many relationships using User/Member/Course analogy
CREATE TABLE User (
    id      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name    TEXT,
    email   TEXT
);

CREATE TABLE Course (
    id      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    title   TEXT
)

CREATE TABLE Member (
    user_id     INTEGER,
    course_id   INTEGER,
        role        INTEGER,
    PRIMARY KEY (user_id, course_id) #note this is what creates the many to many unique relationships
)

INSERT INTO User (name, email) VALUES ('Jane', 'jane@tsugi.org');
INSERT INTO User (name, email) VALUES ('Ed', 'ed@tsugi.org');
INSERT INTO User (name, email) VALUES ('Sue', 'sue@tsugi.org');
INSERT INTO Course (title) VALUES('Python');
INSERT INTO Course (title) VALUES('SQL');
INSERT INTO Course (title) VALUES('PHP');

INSERT INTO Member (user_id, course_id, role) VALUES (1, 1, 1);
INSERT INTO Member (user_id, course_id, role) VALUES (2, 1, 0);
INSERT INTO Member (user_id, course_id, role) VALUES (3, 1, 0);
INSERT INTO Member (user_id, course_id, role) VALUES (1, 2, 0);
INSERT INTO Member (user_id, course_id, role) VALUES (2, 2, 1);
INSERT INTO Member (user_id, course_id, role) VALUES (2, 3, 1);
INSERT INTO Member (user_id, course_id, role) VALUES (3, 3, 0);

SELECT User.name, Member.role, Course.title
FROM User JOIN Member JOIN Course
ON Member.user_id = User.id AND Member.course_id = Course.id
ORDER BY Course.title, Member.role DESC, User.name

#Worked Example on many to many table
conn = sqlite3.connect('rosterdb.sqlite')
cur.conn.cursor()

cur.executescript('''
DROP TABLE IF EXISTS User;
DROP TABLE IF EXISTS Member;
DROP TABLE IF EXISTS Course;

CREATE TABLE User (
    id      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name    TEXT UNIQUE
);

CREATE TABLE Course(
    id      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    title   TEXT UNIQUE
);

CREATE TABLE Member (
    user_id     INTEGER,
    course_id   INTEGER,
    role        INTEGER,
    PRIMARY KEY (user_id, course_id)
)
''')

fname = input('Enter file name: ')
if len(fname) < 1:
    fname = 'roster_data_sample.json'

str_data = open(fname).read()
json_data = json.loads(str_data)

for entry in json_data:
    name = entry[0];
    title = entry[1];
    role = entry[2];
    print((name, title, role))

    cur.execute('''INSERT OR IGNORE INTO User (name) VALUES ( ? )''', ( name, ))
    cur.execute('SELECT id FROM User WHERE name = ? ', (name, ))
    user_id = cur.fetchone()[0]

    cur.execute('''INSERT OR IGNORE INTO Course (title) VALUES ( ? )''', ( title, ))
    cur.execute('SELECT id FROM Course WHERE title = ? ', (title, ))
    title_id = cur.fetchone()[0]

    cur.execute('''INSERT OR REPLACE INTO Member (user_id, course_id, role) VALUES ( ?, ?, ?)''', ( user_id, course_id, role ))

    conn.commit()

#Google geospatial API example in Week5 lecture
import urllib.request, urllib.parse, urllib.error
import http
import sqlite3
import json
import time
import ssl
import sys

api_key = False
# If you have a Google Places API key, enter it here
# api_key = 'AIzaSy___IDByT70'

if api_key is False:
    api_key = 42
    serviceurl = "http://py4e-data.dr-chuck.net/json?"
else :
    serviceurl = "https://maps.googleapis.com/maps/api/geocode/json?"

# Additional detail for urllib
# http.client.HTTPConnection.debuglevel = 1

conn = sqlite3.connect('geodata.sqlite')
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS Locations (address TEXT, geodata TEXT)''')

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

fh = open("where.data")
count = 0
for line in fh:
    if count > 200 :
        print('Retrieved 200 locations, restart to retrieve more')
        break

    address = line.strip()
    print('')
    cur.execute("SELECT geodata FROM Locations WHERE address= ?",
        (memoryview(address.encode()), ))

    try:
        data = cur.fetchone()[0]
        print("Found in database ",address)
        continue
    except:
        pass

    parms = dict()
    parms["address"] = address
    if api_key is not False: parms['key'] = api_key
    url = serviceurl + urllib.parse.urlencode(parms)

    print('Retrieving', url)
    uh = urllib.request.urlopen(url, context=ctx)
    data = uh.read().decode()
    print('Retrieved', len(data), 'characters', data[:20].replace('\n', ' '))
    count = count + 1

    try:
        js = json.loads(data)
    except:
        print(data)  # We print in case unicode causes an error
        continue

    if 'status' not in js or (js['status'] != 'OK' and js['status'] != 'ZERO_RESULTS') :
        print('==== Failure To Retrieve ====')
        print(data)
        break

    cur.execute('''INSERT INTO Locations (address, geodata)
            VALUES ( ?, ? )''', (memoryview(address.encode()), memoryview(data.encode()) ) )
    conn.commit()
    if count % 10 == 0 :
        print('Pausing for a bit...')
        time.sleep(5)

print("Run geodump.py to read the data from the database so you can vizualize it on a map.")

#and then the follow up to that is the geodump code:
import sqlite3
import json
import codecs

conn = sqlite3.connect('geodata.sqlite')
cur = conn.cursor()

cur.execute('SELECT * FROM Locations')
fhand = codecs.open('where.js', 'w', "utf-8")
fhand.write("myData = [\n")
count = 0
for row in cur :
    data = str(row[1].decode())
    try: js = json.loads(str(data))
    except: continue

    if not('status' in js and js['status'] == 'OK') : continue

    lat = js["results"][0]["geometry"]["location"]["lat"]
    lng = js["results"][0]["geometry"]["location"]["lng"]
    if lat == 0 or lng == 0 : continue
    where = js['results'][0]['formatted_address']
    where = where.replace("'", "")
    try :
        print(where, lat, lng)

        count = count + 1
        if count > 1 : fhand.write(",\n")
        output = "["+str(lat)+","+str(lng)+", '"+where+"']"
        fhand.write(output)
    except:
        continue

fhand.write("\n];\n")
cur.close()
fhand.close()
print(count, "records written to where.js")
print("Open where.html to view the data in a browser")

#################BREAK######################################BREAK######################################BREAK#####################
#Codesignal practice stuff

isinstance(x,y)
#where x = variable and y = test, such as 'int' if you want to check if its an integer
hex(int(x,y)) #where x represents the number and y represents the new base you are changing to
# note that '\t' is a tab space
#note that you can repeat a string by multiplying, for example:
'\t'*4 will be 4 spaces
