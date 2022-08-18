#Example of 'toggle' ability in functions to include/exclude paramaters
def add_numbers(x, y, z=None):
    return x + y + z
#here z is currently None, but if you changed that the return statement would stand; could also be:
def add_numbers(x, y, z=None):
    if (z == None):
        return x + y
    else:
        return x + y + z

#notes/review on tuples/lists/dicts
[1, 2] + [3, 4]
[1, 2, 3, 4]

[1] *3
[1, 1, 1]

1 in [1, 2, 3]
True

#Unpacking example
x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
fname, lname, email = x
fname
#output is 'Christopher'

#Ex of string formatting statement
sales_record = {'price': 3.24,
                'num_items': 4,
                'person': 'Chris'}
sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'
print(sales_statement.format(sales_record['person'],
                            sales_record['num_items'],
                            sales_record['price'],
                            sales_record['num_items']*sales_record['price']))
#output is: Chris brought 4 item(s) at a price of 3.24 each for a total of 12.96

#date/time issues
import datetime as dt
import time as tm
tm.time() #this gives current time as a number
dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow
#output is: #output is: datetime.datetime(2022, 7, 19, 7, 57, 20, 483906)
#above is (year, month, day, hour, minute, second, ?)

#class definition
class Person:
    department = 'Modeling & Simulation'
#this would mean that everyone is part of M&S department

#map function
map(function, iterable1, iterable2, ...) #Example
store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
cheapest
#output = <map object at 0x000002B3365669E0>

#lambda functions
my_function = lambda a, b, c : a + b #have to use lambda, followed by function, followed by colon, followed by expression

#numpy
import numpy as np
import math
a = np.array([1, 2, 3])
print(a)

#note the dimensions function for lists
print(a.ndim)
#also the shape attribute, which returns a tuple
a.shape
#also the type can be returned from numpy functions
c = np.array([2.2, 5, 1.1])
print(c.dtype.name)
#output is: float64

#random number generator for arrays
np.random.rand(2,3)
#output are tuples with shape 2 by 3, random numbers below 1, can also use linspace function to define the number of numbers you generate
np.linspace(0, 2, 15) #this is 15 numbers between 0 and 2

#can use the @ sign instead of * for multiplication when you want a matrix product
A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])
print(A*B)
print(A@B)

#Example of creating array with the shape we decide, based on parameters we choose
b = np.arange(1, 16, 1).reshape(3,5)
print(b)

#Example of using math to invert a picture image
from PIL import image
from IPython.display import display
im = Image.open('filename')
display(im) #note this has to be done in Jupyter notebook to see, but utlimately can write to a file
array = np.array(im)
print(array.shape)
print(array)
mask = np.full(array.shape,255)
print(mask)
modified_array = array - mask
modified_array = modified_array * -1
modified_array = modified_array.astype(np.uint8)
display(Imaage.fromarray(modified_array))

#helpful hint to get certain data from arrays, use the index location; **important - for multidimensional arrays, the first argument refers to the row, second to the column
a = np.array([[1,2], [3,4], [5,6]])
print(a[1,1])
#output is 4, because when starting from zero, 4 would be the second row, second column in the array, although denoted from first position in both
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a[:2,1:3])
#output for this multidimensional array would be to only get the first two arrays, then slicing by 1:3, best to print out the full array (a), then visualize from there
#this is also called 'passing by reference'

#to insert new values into a multidimensional array, use this method
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
sub_array = a[:2,1:3]
print('sub array index [0,0] value before change:', sub_array[0,0]) #for visual
sub_array[0,0] = 50
print('sub array index [0,0] value after change:', sub_array[0,0])

#using genfromtxt() function
wines = np.genfromtxt('filename', delimiter=';', skip_header=1)
print(wines)
#for the same array, what if you want non-consecutive columns? like some of the code structure is irrelevant...
print(wines[:, [0,2,3]]) #output is only the 0,2,3 columns

#Regular Expressions
#can use search() and match()
text = 'This is a good day'
if re.search('good',text):
    print('Found it!')

#tokenizing NLP work
text = 'Will works diligently. Will gets decent grades. The student Will is not successful'
print(re.split('Will',text))
print(re.findall('Will',text))

#anchors
# '^' indicates the the start, and '$' is the end
print(re.search('^Will',text))
print(re.search('$Will',text))
#note it also acts as a delete function when using square brackests, i.e.
print(re.findall('[^Will]',text))

#if you want to find a pattern where two things are next to each other
grades = 'AABABABBBBCBACB'
print(re.findall('[A][B-C]',grades))
#output tells you instances where A is followed by either B or C

#not sure if this is the same thing or not, but same output
grades = 'AABABABBBBCBACB'
print(re.findall('AB|AC',grades))

#Quantifiers = the number of times you want a pattern to be matched in order to produce desired output
#Ex: e{m,n}, where e is the character/expression we are matching, m is the minimum number of times and n is max
grades = 'AABABABBBBCBACB'
print(re.findall('A{2,10}',grades)) #how many times did you get an A, twice in a row, with a max of 10 (placeholder)

#using finditer() and re.VERBOSE functions
#assume the data is a large text file describing buddhist colleges with locations
pattern = '''
(?P<title>.*)       #finds the title of the school
(-\ located\ in\ )  #an indicator for the location of the school
(?P<city>\w*)       #city
(,\ )               #separator for the state
(?P<state>\w*)      #state
'''
for item in re.finditer(pattern,filename,re.VERBOSE):
    print(item.groupdict())
#output would be a dictionary of all the schools with city, state

#parsing tweets and hashtags (or whatever character you wanted)
with open('nytimeshealth.txt','r') as file:
    health = file.read()
pattern = '#[\w\d]*(?=\s)'
#note that \w is words/whitespace, \d are digits, * is any number of characters, and \s is whitespace
print(re.findall(pattern,health))

#pandas
import pandas as pd
students = ['Alice','Jack','Molly']
print(pd.Series(students)) #note you can also pass a list of numbers/integers into the same thing, in place of names; can also pass None into list

#diff between NaN and None, which are NOT equivalent
import pandas as pd
numbers = [1,2,None]
print(pd.Series(numbers))
#NaN is actually not even equal to itself, ergo
import numpy as np
np.nan == np.nan
#Output is False
#but you can verify it using the isnan() function:
import numpy as np
print(np.isnan(np.nan))
#output is True

#using pandas to create a series of tuples
import pandas as pd
students = [('Lucy','Mason'),('Mac','Mason'),('Will','Mason')]
print(pd.Series(students))

#for passing in new values where they don't exist in dictionaries
import pandas as pd
student_scores = {'Alice':'Chemistry','Jack':'Math','Molly':'English'}
s = pd.Series(student_scores,index = ['Alice','Molly','Sam'])
print(s)
#output is to print out a series and replace Jack with Sam and NaN

#for searching through series, can use the iloc() and loc() functions
import pandas as pd
student_classes = {'Alice':'Chemistry','Jack':'Math','Molly':'English','Sam':'History'}
s = pd.Series(student_classes)
print(s.iloc[1])
print(s.loc['Sam'])

#to begin analyzing the Series, not much diff than usual; here is an averaging
import pandas as pd
grades = pd.Series([90, 80, 70, 60])
total = 0
for grade in grades:
    total = total + grade
    average = total / len(grades)
print(average)

#when using larger data sets, there is a choice on how to compute things, which becomes important in database management; an example:
import pandas as pd
import numpy as np
numbers = pd.Series(np.random.randint(0,1000,10000)) #note this is asking for 10,000 numbers between 0-1000
print(numbers.head) #just to print out and double check they are random numbers, head function gives you the 'top' five, meaning 0-5
#now in order to test the amount of time to run the code, can use the timeit cellular function, but you have to use it at the beginning, and preface with double %
&&timeit -n 100
total = 0
for number in numbers:
    total = total + number
    print(total/len(numbers))

#Dataframe structures
import pandas as pd
record1 = pd.Series({'Name':'Alice','Class':'Physics','Score':85})
record2 = pd.Series({'Name':'Jack','Class':'Chemistry','Score':82})
record3 = pd.Series({'Name':'Helen','Class':'Biology','Score':90})
df = pd.DataFrame([record1, record2, record3],index=['school1','school2','school3'])
print(df.head)

#another exmaple; note this is same result as above, just in a diff way
import pandas as pd
students = [{'Name':'Alice','Class':'Physics','Score':85},{'Name':'Jack','Class':'Chemistry','Score':82},{'Name':'Helen','Class':'Biology','Score':90}]
df = pd.DataFrame(students,index=['school1','school2','school3'])
print(df.head)

#to extract data, use iloc(); can also check the data type by adding type()
import pandas as pd
students = [{'Name':'Alice','Class':'Physics','Score':85},{'Name':'Jack','Class':'Chemistry','Score':82},{'Name':'Helen','Class':'Biology','Score':90}]
df = pd.DataFrame(students,index=['school1','school2','school3'])
print(df.loc['school2'])
print(type(df.loc['school2']))

#how to transpose things
import pandas as pd
students = [{'Name':'Alice','Class':'Physics','Score':85},{'Name':'Jack','Class':'Chemistry','Score':82},{'Name':'Helen','Class':'Biology','Score':90}]
df = pd.DataFrame(students,index=['school1','school2','school3'])
print(df.T.loc['Name'])

#how to drop/delete data from series
import pandas as pd
students = [{'Name':'Alice','Class':'Physics','Score':85},{'Name':'Jack','Class':'Chemistry','Score':82},{'Name':'Helen','Class':'Biology','Score':90}]
df = pd.DataFrame(students,index=['school1','school2','school3'])
df.drop('school1')
print(df)

#Boolean Masking - so this is getting a True/False reading on whether the chance of admit column is greater than 0.7
import pandas as pd
df = pd.read_csv('Admission_Predict.csv',index_col=0)
df.columns = [x.lower().strip() for x in df.columns]
admit_mask=df['chance of admit'] > 0.7
print(admit_mask)

#in order to hide the data you do not want after doing your Boolean Masking, use this (note some of these functions may require Jupyter notebook, unsure):
import pandas as pd
df = pd.read_csv('Admission_Predict.csv',index_col=0)
df.columns = [x.lower().strip() for x in df.columns]
admit_mask=df['chance of admit'] > 0.7
print(df.where(admit_mask)).head()
#and if you add the 'dropna()' to that, it will remove all the excluded items, or at least not show them
print(df.where(admit_mask)).dropna().head()
#alternatively, you can put the criteria inside, so:
import pandas as pd
df = pd.read_csv('Admission_Predict.csv',index_col=0)
df.columns = [x.lower().strip() for x in df.columns]
print(df[df['chance of admit'] > 0.7].head())

#and if you only want to show certain columns, just call them as such:
import pandas as pd
df = pd.read_csv('Admission_Predict.csv',index_col=0)
df.columns = [x.lower().strip() for x in df.columns]
print(df[df['gre score','toefl score']].head())

#to combine Boolean masks, have to use'&':
import pandas as pd
df = pd.read_csv('Admission_Predict.csv',index_col=0)
df.columns = [x.lower().strip() for x in df.columns]
print((df['chance of admit'] > 0.7) & (df['chance of admit'] < 0.9))

#Setting an index with set_index() function, which is destruction, meaning it changes the attributes:
import pandas as pd
df = pd.read_csv('Admissions_Predict.csv',index_col=0)
print(df.head())
#in order to add/reset an index, which just produces a numbered series:
df = df.reset_index()
print(df.head())
#unique function to search datasets, columns:
import pandas as pd
df = pd.read_csv('census.csv')
print(df['SUMLEV'].unique())
#or if you wanted to filter:
df = df[df['SUMLEV'] == 50] #or
columns_to_keep = ['STNAME','CTYNAME','BIRTHS2010','BIRTHS2011']

#if you want to test if there is any missing data, and return a BooleanMask of True/False where data is missing:
import pandas as pd
df = pd.read_csv('class_grades.csv')
mask = df.isnull()
print(mask.head(10))
#and if you simply want to drop all missing data:
import pandas as pd
df = pd.read_csv('class_grades.csv')
print(df.dropna().head(10))
#if you want to replace the missing data with something(integer, string, etc.):
import pandas as pd
df = pd.read_csv('class_grades.csv')
df.fillna('Where it be?',inplace=True)
print(df.head(10))

#Sorting and filling: if you have blank or bad data that you want to change, you can fill either by what was before via ffill() or after bfill():
import pandas as pd
df = pd.read_csv('log.csv')
df = df.set_index('time') #this simply sorts by the time column
df = df.set_index()
print(df.head(20))

#Replace data
import pandas as pd
df = pd.DataFram({'A':[1,1,2,3,4],'B':[3,6,3,8,9],'C':['a','b','c','d','e']})
print(df.replace([1,3],[100,300])) #this replaces all 1s & 3s with 100s & 300s

#replace data using regex; example here is using the log file replacing everything that ends in .html:
import pandas as pd
df = pd.read_csv('log.csv')
print(df.replace(to_replace = '.*.html$',value = 'webpage', regex = True))

#Cleaning up datasets
import pandas as pd
df = pd.read_csv('presidents.csv')
df['First'] = df['President'] #this creates new column name using data from 'President' to 'First'
df['First'] = df['First'].replace('[ ].*','', regex = True) #using regex to find whitespace, followed by any character, repeated 1 or more times

#to format date/time using Pandas
import pandas as pd
df = pd.read_csv('presidents.csv')
df['Born'] = df['Born'].str.extract('([\w]{3} [\w]{1,2}, [\w]{4})') #this gets the junk out so only date/time data
df['Born'] = pd.to_datetime(df['Born'])

#Merging DataFrames
import pandas as pd
staff_df = pd.DataFrame([{'Name':'Kelly','Role':'Director of HR'},{'Name':'Sally','Role':'Course Liason'},{'Name':'James','Role':'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name':'James','School':'Business'},{'Name':'Mike','School':'Law'},{'Name':'Sally','School':'Engineering'}])
student_df = student_df.set_index('Name')
print(pd.merge(staff_df,student_df,how = 'outer',left_index = True,right_index = True)) #this is merging using an outer join, if you wanted the intersection just change to 'inner'
#similarly, if you want to join using the index selector, you change 'outer' with either 'left' or 'right'; this is called set addition
#or if you want to join by a column, use the 'on' method
pd.merge(staff_df,student_df, how = 'right', on = 'Name')

#Concatenation
import pandas as pd
df_2011 = pd.read_csv('MERGED2011_12_PP.csv', error_bad_lines = False)
df_2012 = pd.read_csv('MERGED2012_13_PP.csv', error_bad_lines = False)
df_2013 = pd.read_csv('MERGED2013_14_PP.csv', error_bad_lines = False)
frames = [df_2011, df_2012, df_2013]
print(pd.concat(frames))

#Panda Idioms
import pandas as pd
import numpy as np
import timeit
df = pd.read_csv(census.gov)
print((df.where(df['SUMLEV']==50).dropna().set_index(['STNAME','CTYNAME']).rename(columns = {'ESTIMATEBASE2010':'Estimates Base 2010'})))
#above line is a Boolean Mask where SUMLEV == 50
df = df[df['SUMLEV'] == 50]
df.set_index(['STNAME','CTYNAME'], inplace = True)
df.rename(columns = {'ESTIMATESBASE2010' : 'Estimates Base 2010'}) #note that line 392 down is the same as line 390, just not 'pandorable'

#for running time on different approaches:
import pandas as pd
import numpy as np
import timeit
def first_approach():
    global df
    return (df.where(df['SUMLEV']==50)
        .dropna()
        .set_index(['STNAME','CTYNAME'])
        .rename(columns = {'ESTIMATESBASE2010':'Estimates Base 2010'}))
df = pd.read_csv('census.csv')
print(timeit.timeit(first_approach, number=10))
#contrast the above to this:


#manipulating data
import pandas as pd
import numpy as np
df = pd.read_csv('census.csv')
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min':np.min(data),'max':np.max(data)})
print(df.apply(min_max, axis='columns').head())

#below is an example of how to add columns with min/max features to your existing dataframe
import pandas as pd
import numpy as np
df = pd.read_csv('census.csv')
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row
print(df.apply(min_max, axis='columns'))

#below is example of same things as above, but using Lambdas
import pandas as pd
import numpy as np
df = pd.read_csv('census.csv')
rows = ['POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012','POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']
print(df.apply(lambda x: np.max(x[rows]), axis=1).head())

#How to print out a list of columns names from csv
import pandas as pd
df = pd.read_csv('listings.csv')
print(list(df))

#Data aggregation
import pandas as pd
import numpy as np
df = pd.read_csv('listings.csv')
print(df.groupby('cancellation_policy').agg({'review_scores_value':np.nanmean})) #note that 'nanmean' does the same as 'average', but excludes NaN values

#and you can add more calculations inside the aggregating function, here is example of both average and standard deviation:
import pandas as pd
import numpy as np
df = pd.read_csv('listings.csv')
print(df.groupby('cancellation_policy').agg({'review_scores_value':(np.nanmean,np.nanstd),'reviews_per_month':np.nanmean}))

#transorming dataframes - ????
import pandas as pd
import numpy as np
df = pd.read_csv('listings.csv')
cols = ['cancellation_policy','review_scores_value']
transform_df = df[cols].groupby('cancellation_policy').transform(np.nanmean)
transform_df.rename({'review_scores_value':'mean_review_scores'}, axis = 'columns', inplace = True) #this is renaming the new column, as it is no longer a review score, but an average
df = df.merge(transform_df, left_index = True, right_index = True)
df['mean_diff'] = np.absolute(df['review_scores_value'] - df['mean_review_scores']) #getting the absolute value of the difference between the old review scores and average review scores
print(df.head())

#filtering dataframes
import pandas as pd
import numpy as np
df = pd.read_csv('listings.csv')
df.groupby('cancellation_policy').filter(lambda x : np.nanmean(x['review_scores_value'])>9.2)
print(df)

#Scales in pandas
import pandas as pd
df = pd.DataFrame(['A+','A','A-','B+','B','B-','C+','C','C-','D+','D'], index = ['excellent','excellent','excellent','good','good','good','ok','ok','ok','poor','poor'], columns = ['Grades'])
print(df['Grades'].astype('category').head())

#Using cut feature
import numpy as np
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg(np.average)
print(pd.cut(df,10))

#pivot tables using pandas
import pandas as pd
import numpy as np
df = pd.read_csv('cwurData.csv')
def create_category(ranking):
    if(ranking >= 1) & (ranking <=100):
        return('First Tier')
    elif(ranking >=101) & (ranking<=200):
        return('Second Tier')
    elif(ranking >=201) & (ranking<=300):
        return('Third Tier')
    else:
        return('Other')
df['Rank_Level'] = df['world_rank'].apply(lambda x : create_category(x))
print(df.pivot_table(values='score',index='country',columns='Rank_Level',aggfunc=[np.mean, np.max]).head())

#date/time function using pandas
print(pd.Timestamp('9/1/2019 10:05AM')) #or
print(pd.Timestamp(2019,9,1,10,5))
#to get numeric value for day of week
print(pd.Timestamp(2019,9,1,10,5).isoweekday()) #can also use the .weekday() function
#in order to get Period by day, month, etc.
print(pd.Period('1/2016')) #or
print(pd.Period('3/5/2016'))
#DatetimeIndex and PeriodIndex:
t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
print(t1)
#calculating diff in time:
pd.Timestamp('9/3/2016') - pd.Timestamp('9/1/2016')
#also adding time:
pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')

#Statistical Testing
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('grades.csv')
print('There are {} rows and {} columns'.format(df.shape[0], df.shape[1])) #this is helpful just to get shape/size of file
early_finishers = df[pd.to_datetime(df['assignment1_submission']) < '2016']
print(early_finishers.head())

#note that the inverse of line 533 could be expressed as follows:
late_finishers = df[~df.index.isin(early_finishers.index)] #the '~' acts as a bitwise inverse function

#to get the mean for each:
print(early_finishers['assignment1_grade'].mean())
print(late_finishers['assignment1_grade'].mean())

#in order to run t-test on outputs:
from scipy.stats import ttest_ind
ttest_ind(early_finishers['assignment1_grade'],late_finishers['assignment1_grade']) #remember that if p-value is <= alpha (usually 5%), then it IS statistically significant

#since P-values are not considered the best way to test significance, can use confidence intervals and bayesian analysis:
import pandas as pd
import numpy as np
from scipy import stats
df1 = pd.DataFrame([np.random.random(100) for x in range(100)])
df2 = pd.DataFrame([np.random.random(100) for x in range(100)])
def test_columns(alpha = 0.1):
    num_diff = 0
    for col in df1.columns:
        teststat,pval = ttest_ind(df1[col],df2[col])
        if pval <= alpha:
            print('Col {} is statistically significantly diff at alpha = {}, pval = {}'.format)col,alpha,pval))
            num_diff = num_diff + 1
    print('Total number diff was {}, which is {}%.'.format(num_diff, float(num_diff) / len(df1.columns)*100))
print(test_columns)

#Example of putting connected nodes into dictionaries:
import networkx as nx
edgelist = [['Mannheim', 'Frankfurt', 85], ['Mannheim', 'Karlsruhe', 80], ['Erfurt', 'Wurzburg', 186], ['Munchen', 'Numberg', 167], ['Munchen', 'Augsburg', 84], ['Munchen', 'Kassel', 502], ['Numberg', 'Stuttgart', 183], ['Numberg', 'Wurzburg', 103], ['Numberg', 'Munchen', 167], ['Stuttgart', 'Numberg', 183], ['Augsburg', 'Munchen', 84], ['Augsburg', 'Karlsruhe', 250], ['Kassel', 'Munchen', 502], ['Kassel', 'Frankfurt', 173], ['Frankfurt', 'Mannheim', 85], ['Frankfurt', 'Wurzburg', 217], ['Frankfurt', 'Kassel', 173], ['Wurzburg', 'Numberg', 103], ['Wurzburg', 'Erfurt', 186], ['Wurzburg', 'Frankfurt', 217], ['Karlsruhe', 'Mannheim', 80], ['Karlsruhe', 'Augsburg', 250],["Mumbai", "Delhi",400],["Delhi", "Kolkata",500],["Kolkata", "Bangalore",600],["TX", "NY",1200],["ALB", "NY",800]]
g = nx.Graph()
for edge in edgelist:
    g.add_edge(edge[0],edge[1], weight = edge[2])
for i, x in enumerate(nx.connected_components(g)):
    print('cc' + str(i) + ':', x)

#to find the shortest path between two locations/pairs:
import networkx as nx
edgelist = [['Mannheim', 'Frankfurt', 85], ['Mannheim', 'Karlsruhe', 80], ['Erfurt', 'Wurzburg', 186], ['Munchen', 'Numberg', 167], ['Munchen', 'Augsburg', 84], ['Munchen', 'Kassel', 502], ['Numberg', 'Stuttgart', 183], ['Numberg', 'Wurzburg', 103], ['Numberg', 'Munchen', 167], ['Stuttgart', 'Numberg', 183], ['Augsburg', 'Munchen', 84], ['Augsburg', 'Karlsruhe', 250], ['Kassel', 'Munchen', 502], ['Kassel', 'Frankfurt', 173], ['Frankfurt', 'Mannheim', 85], ['Frankfurt', 'Wurzburg', 217], ['Frankfurt', 'Kassel', 173], ['Wurzburg', 'Numberg', 103], ['Wurzburg', 'Erfurt', 186], ['Wurzburg', 'Frankfurt', 217], ['Karlsruhe', 'Mannheim', 80], ['Karlsruhe', 'Augsburg', 250],["Mumbai", "Delhi",400],["Delhi", "Kolkata",500],["Kolkata", "Bangalore",600],["TX", "NY",1200],["ALB", "NY",800]]
g = nx.Graph()
for edge in edgelist:
    g.add_edge(edge[0],edge[1], weight = edge[2])
print(nx.shortest_path(g, 'Stuttgart','Frankfurt', weight = 'weight'))
print(nx.shortest_path_length(g, 'Stuttgart','Frankfurt', weight = 'weight'))
#note that the code for finding the shortest path between ALL pairs is:
for x in nx.all_pairs_dijkstra_path(g, weight = 'weight'):
    print(x)

#Somehow above is diff than minimum span, but minimum spanning trees is as follows:
nx.draw_networkx(nx.minimum_spanning_tree(g))
