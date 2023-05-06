#plotting graph in Pycharm
import math
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

plt.plot(x, 2*x, label='f(x) = 2x') #just add another line like this to include another function

plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Both Graphs")
fig = plt.figure()
axes = fig.add_axes([0.5, 1, 0.5, 1])
fig.show()
plt.legend()
plt.show(block=True)
plt.interactive(False)

#plotting square root using matplotlib
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-20,20,1)
y = x**2
plt.plot(x,y)
plt.xlabel('x label')
plt.ylabel('y label')
plt.title('g(x) = sqrt(x)')
plt.show()

#to find the angle between two 3D vectors:
import numpy as np
import vg

V_1 = np.array([2,1,5])
V_2 = np.array([-1,-3,4])
Angle = round(vg.angle(V_1,V_2),0)

print('The answer is',Angle,'degrees')

#to multiply vector via dot product:
import numpy as np

A = [2,1,5]
B = [-1,-3,4]

print(np.dot(A,B))

#to multiply one matrix by the inverse of itself:
import numpy as np
A = np.array([[3,2],[-2,1]])
B = (np.linalg.inv(A))
D_P = np.dot(A,B)
print(B,D_P)

#to get magnitude and unit vector of 2d vector:
import numpy
import math

def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))
v = numpy.array([3,4])
unit_vector = (v[0]/magnitude(v),v[1]/magnitude(v))

print('Vector:', v)
print('Magnitude of the Vector:', magnitude(v))
print('The Unit Vector is:',unit_vector)

#for finding angles, just switch type in lines 1 & 3 and angle in line 3:
from mpmath import cot
import numpy as np
print(cot(np.pi/4))

#to find derivative, domain, factors, limits, and lots of other info about a polynomial (can widdle down to only include factors):
from sympy import *
from fractions import *

x = symbols('x')
f = Function('f')
f = input('type function: ')
fp = diff(f)
sol = solve(f, x)
sol_p = solve(fp, x)
print(f"f(x)={f},f'(x)={fp}")

print(f'{len(sol)}')
psol = {}
limits_at_edges = {}
df = solveset(f, x, domain=S.Reals)

for i in range(1,( len(sol) + 1)):
    psol["x" + str(i) ] = sol[i - 1].evalf()
for i in range(1, len(sol) + 1):
    limits_at_edges[f'limit x -> x{i} f(x)'] = limit(f, x, sol[i - 1])

print(f'Solution:{sol}')
print(f'Processes solution:{psol}')
print(f'Derivative solution:{sol_p}')
print(limits_at_edges)
print(f'Root(s):{df}')
pprint(f, use_unicode=True)

#or, to only find factors of polynomial:
from sympy import *

x = symbols('x')
f = Function('f')
f = input('type function: ')
df = solveset(f, x, domain=S.Reals)

print(f'Root(s):{df}')

#another way to solve a quadratic equation:
def qua(x1, x2, a,b,c):
    for i in range(x1,x2+1):
        if a*i*i+b*i+c == 0:
            print(i)

qua(-10,10,1,-8,15)

#to get geolocation of named place:
from geopy.geocoders import Nominatim

loc = Nominatim(user_agent="GetLoc")
getLoc = loc.geocode("Bedford Hills")

print(getLoc.address,"\n")
print("Latitude = ", getLoc.latitude)
print("Longitude = ", getLoc.longitude)

#or the reverse from lat/long coordinates:
from geopy.geocoders import Nominatim

geoLoc = Nominatim(user_agent="GetLoc")
locname = geoLoc.reverse("41.2366622, -73.7001868")

print(locname.address)

#to measure feet between two lat/long, can switch to .miles:
from geopy.distance import geodesic

Intercoastal = (29.447547,-81.115183)
Atlantic_Ocean = (29.448696,-81.111890)

print('It is',geodesic(Intercoastal,Atlantic_Ocean).feet,'from shore to shore')

#to measure geodesic distance between two named points (x,y) in miles:
from geopy.geocoders import Nominatim
import geopy.distance

loc = Nominatim(user_agent="GetLoc")
x = input('From where?    ')
y = input('To where?    ')
getLoc1 = loc.geocode(x)
getLoc2 = loc.geocode(y)

start_point_lat = getLoc1.latitude
start_point_lon = getLoc1.longitude
end_point_lat = getLoc2.latitude
end_point_lon = getLoc2.longitude

coords_1 = (start_point_lat,start_point_lon)
coords_2 = (end_point_lat,end_point_lon)
result = round(geopy.distance.geodesic(coords_1,coords_2).miles,2)

print('\nIt is',result,'miles between {} and {}'.format(x,y))

#to iterate values through a function with print statement:
possible_solutions = [-1,1,-1/3,1/3,-2,2,-2/3,-3,3,-6,6]
for x in possible_solutions:
    function = (3*x**3 - 4*x**2 - 13*x - 6)
    if function != 0:
        continue
    if function == 0:
        print(round(x,2),'  is a solution for the function')

#to get position in a list of ALL locations:
word_list = ['a','b','c','a','e','f','g','h','i','j','a']
def find_the_word(list_to_check,item_to_find):
    indices = []
    for idx,value in enumerate(word_list):
        if value == item_to_find:
            indices.append(idx)
    return indices
print(find_the_word(word_list,'a'))

#generating bigrams over pre-split text (add split feature above code if not already):
data = ['A','Tropical','Storm','Warning','has','been','issued','for','parts','of','Florida','as','the','tropical','system','in','the','southwestern','Gulf','of','Mexico','is','coming']

def generate_ngrams(data,ngram):
    temp = zip(*[data[i:] for i in range(0,ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans
print(generate_ngrams(data,2))

#to print off list of permutations **BE CAREFUL, WILL PRINT ALL!:
from itertools import permutations

listA = ['A','B','C']
num_of_perms = permutations(listA)
for i in list(num_of_perms):
    print(i)

#for counting the number of permutations:
from itertools import permutations

lst = ['cm1', 'cm2', 'cm3', 'cm4', 'cm5', 'cm6']

num_of_perms = permutations(lst)

x = [i for i in num_of_perms]

print(len(x))

#for printing combinations (think there is a shorter way with comb):
from itertools import combinations

listA = ['A','B','C','D','E','F','G','H','I','J','K']
comb = combinations(listA,2)
for i in list(comb):
    print(i)

#for counting combinations (need to change len(lst) to appropriate setting, will result in 1 otherwise):
from itertools import combinations

lst = ['cm1', 'cm2', 'cm3', 'cm4', 'cm5', 'cm6']

num_of_combs = combinations(lst,len(lst))

x = [i for i in num_of_combs]

print(len(x))

#but a better way to just calculate/solve if you know answer:
import math
print(math.comb(6,2) * math.comb(5,2) - 150)

#for removing all instances of an item in a list:
x = ['calculate', 'trig', 'chemistry', 'trig']
x[:] = [i for i in x if i != 'trig']
print(x)

#to calculate runtime:
import timeit
start = timeit.default_timer()
##CODE IN THE MIDDLE##
stop = timeit.default_timer()
print('\nTime: ', round(stop - start,6)) #might need to increase integer

#for dataframes after getting into pandas pd:
print(data.info())
print(data.describe())
print(data.shape)
print('There are {} rows and {} columns'.format(data.shape[0], data.shape[1]))


#to pretty print a DataFrame (usually using Pandas) into terminal:
from IPython.display import display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
display(df)

#to graph a time series with two columns, date and data:
import pandas as pd
from matplotlib import pyplot
data = pd.read_excel('CPI.xls',header=0,index_col=0,parse_dates=True)
data.plot()
pyplot.show()

#basic pandas dataframe filtering and printing results:
import pandas as pd
df = pd.read_excel('HW8 file.xlsx')

women_df = df[df.gender == 2]
denominator_w = len(women_df) * 3       #there are three columns of limitation possibilities
numerator_w = women_df[women_df.future == 1].shape[0] + women_df[women_df.old == 1].shape[0] + women_df[women_df.email == 1].shape[0]
answer_w = round((numerator_w / denominator_w)*100,2)

men_df = df[df.gender == 1]
denominator_m = len(men_df) * 3
numerator_m = men_df[men_df.future == 1].shape[0] + men_df[men_df.old == 1].shape[0] + men_df[men_df.email == 1].shape[0]
answer_m = round((numerator_m / denominator_m)*100,2)

print('\nOf the {} limitation possibilities, women chose to limit {} times, which is {}% of the time'.format(denominator_w,numerator_w,answer_w))
print('Of the {} limitation possibilities, men chose to limit {} times, which is {}% of the time'.format(denominator_m,numerator_m,answer_m))

if answer_w > answer_m:
    print('\nThus, women use more privacy settings than men')
else:
    print('\nThus, men use more privacy settings than women')

#the reverse of .head()
print(df.tail())

#another way to get all the column/variable names:
print(df.columns)

#Binomial Distribution Probability:
import math as m

n = 10
p = .2
k = 3

probability = (m.factorial(n) / (m.factorial(k)*(m.factorial(n - k)))) * p**k * ((1 - p)**(n - k))

print('\nThe probability of observing {} successes in {} independent trials is {}%'.format(k,n,round(probability*100,2)))

#Poisson Distribution Probability:
import math as m

mu = 50
k = 30
e = m.e

poisson_prob = ((mu**k)*(e**-mu)) / m.factorial(k)

print('\n**The probability that exactly {} successes will occur in a Poisson distribution is {}%'.format(k,round(poisson_prob*100,2)))

#for working with binomial distributions that are equal, greater to, or less than a null hypothesis, just put in n,p,k:
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
import math as m

n = int(input('\nWhat is sample size n?    '))
p = float(input('What is the probability p?    '))
k = int(input('How many observations k?    '))
more_or_less = str(input('If Equal To type "=", if Greater Than type ">", if Greater Than or Equal To type ">=", if Less Than type "<", if Less Than or Equal To type "<=":    '))

mu = n * p
variance = (n * p) * (1 - p)
sigma = m.sqrt(variance)
pb = binom(n, p)

if more_or_less == '=':
    probability = (m.factorial(n) / (m.factorial(k) * (m.factorial(n - k)))) * (p**k) * ((1 - p)**(n - k))
    Z_Score = (k - mu) / sigma
    print('\n**The probability of exactly {} successes in {} independent trials is {}%'.format(k,n,round(probability*100,2)))
elif more_or_less == '>':
    x = np.arange(k + 1, n + 1)
    Z_Score = (k + 0.5 - mu) / sigma
    pmf = sum(pb.pmf(x))
    print('\n**The probability of greater than {} successes in {} independent trials is {}%'.format(k,n,round(pmf*100,2)))
elif more_or_less == '>=':
    x = np.arange(k, n + 1)
    Z_Score = (k - 0.5 - mu) / sigma
    pmf = sum(pb.pmf(x))
    print('\n**The probability of greater than or equal to {} successes in {} independent trials is {}%'.format(k,n,round(pmf*100,2)))
elif more_or_less == '<':
    x = np.arange(1, k)
    Z_Score = (k - 0.5 - mu) / sigma
    pmf = sum(pb.pmf(x))
    print('\n**The probability of less than {} successes in {} independent trials is {}%'.format(k,n,round(pmf*100,2)))
else:
    x = np.arange(1, k + 1)
    Z_Score = (k + 0.5 - mu) / sigma
    pmf = sum(pb.pmf(x))
    print('\n**The probability of less than or equal to {} successes in {} independent trials is {}%'.format(k,n,round(pmf*100,2)))

dist = [binom.pmf(x,n,p) for x in x]
plt.bar(x,dist)
plt.show()

print('\nDESCRIPTIVE STATISTICS:','\nMean:    ',mu,'\nVariance:',variance,'\nSigma:   ',round(sigma,2),'\nZ_Score: ',round(Z_Score,2))

#to get distance in feet or miles from two points using lat/long:
from geopy.distance import geodesic

start = (29.448095,-81.113593)
end = (29.478459,-81.147693)
distance = geodesic(start,end).feet

if distance < 5250:
    print('\nIt is',round(distance,0),'feet from start to end')
else:
    print('\nIt is',round(distance/5250,2),'miles from start to end')

#P-Value, Z Score and probability calculator:
import scipy.stats
import math as m

raw_x = float(input('What is the x?    '))
pop_mean = float(input('What is the average or null?    '))
st_dev = float(input('Standard Deviation?    '))
sides = int(input('One Sided ("1") or Two Sided ("2")?    '))

Z_Score = (raw_x - pop_mean) / st_dev

if sides == 1:
    p_value = scipy.stats.norm.sf(abs(Z_Score))
    print('\np-value:',round(p_value*100,2),'%')
    print('\n{}% of the observations fall above {}'.format(round(p_value*100,2),raw_x))
    print('{}% of the observations fall below {}'.format(round(((1 - p_value)*100),2),raw_x))
else:
    p_value = scipy.stats.norm.sf(abs(Z_Score))*2
    print('\np_value:',round(p_value*100,2),'%')
    print('\n{}% of the observations fall above {}'.format(round(p_value*100,2),raw_x))
    print('{}% of the observations fall below {}'.format(round(((1 - p_value)*100),2),raw_x))

print('\nDESCRIPTIVE STATISTICS:','\nMean:    ',pop_mean,'\nVariance:',round(st_dev**2,2),'\nSigma:   ',round(st_dev,2),'\nZ Score: ',round(Z_Score,2))

#95% Confidence Interval, probability, Z Score et al:
import scipy.stats
import math as m

n = int(input('\nWhat is the sample size?    '))
raw_x = float(input('What is the x?    '))
pop_mean = float(input('What is the average or null?    '))
st_dev = float(input('Standard Deviation?    '))
sides = int(input('One Sided ("1") or Two Sided ("2")?    '))

st_error = st_dev / m.sqrt(n)
Z_Score = (raw_x - pop_mean) / st_error
conf_interval = [(raw_x - (1.96 * st_error)),(raw_x + (1.96 * st_error))]

if sides == 1:
    p_value = scipy.stats.norm.sf(abs(Z_Score))
    print('\np-value:',round(p_value*100,2),'%')
    print('\n{}% of the observations fall above {}'.format(round(p_value*100,2),raw_x))
    print('{}% of the observations fall below {}'.format(round(((1 - p_value)*100),2),raw_x))
else:
    p_value = scipy.stats.norm.sf(abs(Z_Score))*2
    print('\np_value:',round(p_value*100,2),'%')
    print('\n{}% of the observations fall above {}'.format(round(p_value*100,2),raw_x))
    print('{}% of the observations fall below {}'.format(round(((1 - p_value)*100),2),raw_x))

if p_value <= .05:
    print('\nBecause the p-value of {}% falls below 5%, we reject the null hypothesis'.format(round(p_value*100,2)))
else:
    print('\nBecause the p-value of {}% falls above 5%, we cannot reject the null hypothesis'.format(round(p_value*100,2)))

if pop_mean >= conf_interval[0] and pop_mean <= conf_interval[1]:
    print('\nBecause the null hypothesis is inside the confidence interval, we cannot reject the null')
else:
    print('\nBecause the null hypothesis is outside the confidence interval, we reject the null')

print('\nDESCRIPTIVE STATISTICS:','\nMean:    ',pop_mean,'\nVariance:',round(st_dev**2,2),'\nSigma:   ',round(st_dev,2),'\nZ Score: ',round(Z_Score,2),'\nS_Error: ',round(st_error,2))
print('\nThe 95% confidence interval is:',conf_interval)

#example of how to print all dataframe, not just head/tail:
import pandas as pd
df = pd.read_csv('hw9.csv')

with pd.option_context('display.max_rows',None,'display.max_columns',None,'display.precision',3) : print(df)

#initializing a Class, passing object through said Class:
class Waves:
	def __init__(wind, type):
		wind.type = type
	def check_waves(wind):
		print('The waves are', wind.type)
p1 = Waves('offshore')
p2 = Waves('onshore')
p3 = Waves('sideshore')
p1.check_waves()
p2.check_waves()
p3.check_waves()

#to get powersets:
def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
print(list(powerset([1,2,3,4])))

#to get dot product or multiply arrays with any dimension you choose:
import numpy as np

arr1 = np.array([1,2,6,4,4,-7,-3,-5,9])
arr1.0 = (3,3)
arr2 = np.array([0,7,3,9,4,-6])
arr2.shape = (3,2)

print(np.dot(arr1,arr2))

#to get identity matrix:
import numpy as np

arr1 = np.array([1,2,6,4,4,-7,-3,-5,9])
arr1.shape = (3,3)
arr2 = np.identity(3)
print(arr2)

#to transpose a matrix:
import numpy as np

arr1 = np.array([0,7,8,1,3,9,3,7])
arr1.shape = (2,4)
arr2 = arr1.transpose()
print(arr1)
print(arr2)
