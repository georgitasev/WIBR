import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import usaddress
from functools import reduce
from sklearn import tree
import pygraphviz as graphviz
import patsy
import re
from sklearn.ensemble import RandomForestClassifier
forest_cl = RandomForestClassifier(n_estimators=100, max_depth=100,random_state=0, n_jobs=-1)
tree_cl = tree.DecisionTreeClassifier(max_depth=100)

justdigits = re.compile('\d+', re.IGNORECASE)

s = pd.Series({1:"District 1: Ashanti Hamilton",
2:"District 2: Cavalier Johnson",
3:"District 3: Nicholas Kovac",
4:"District 4: Robert Bauman",
5:"District 5: James Bohl, Jr.",
6:"District 6: Milele A. Coggs",
7:"District 7: Khalif J. Rainey",
8:"District 8: Robert G. Donovan",
9:"District 9: Chantia Lewis",
10:"District 10: Michael J. Murphy",
11:"District 11: Mark A. Borkowski",
12:"District 12: José G. Pérez",
13:"District 13: Terry L. Witkowski",
14:"District 14: T. Anthony Zielinski",
15:"District 15: Russell W. Stamper, II"})

wibr = pd.read_pickle("wibr.pkl.gz",compression='gzip')
aldermanic_districts = pd.DataFrame(s,columns=['Alderperson']).reset_index().rename(columns={'index':'ALD'})
aldermanic_districts['ALD'] = aldermanic_districts.ALD.astype(float)
major_crime_types=set([ 'Arson', 'AssaultOffense', 'Burglary', 'CriminalDamage','Homicide', 'LockedVehicle', 'Robbery', 'SexOffense', 'Theft','VehicleTheft'])

def datetime_metadata(dt):
	return pd.Series({'hour':dt.hour,'dte':dt.date(),'dayname':dt.day_name(),'month':dt.month_name(), 'year':dt.year})

def foldin_date_metadata(df,dtf):
    s = df[dtf].apply(lambda dt: datetime_metadata(dt))
    return pd.concat([df,s],axis=1)

def recognizableAddress(locationString):
    if type(locationString) == str:
        try:
            usat = usaddress.tag(locationString)
        except:
            return False
        else:
            if usat[1] in ('Street Address'):
                return True
            else:
                return False
    else:
        return False
	
def decomposeAddress(locationString):
	if recognizableAddress(locationString):
		t = usaddress.tag(locationString)
		addr = t[0]
		an = addr['AddressNumber']
		attr_list = [addr[k] for k in addr.keys() if k != 'AddressNumber']
		sp = reduce(lambda x,y: x+' '+y,attr_list)
		return (an, sp)
	else:
		return None

def num_to_block(num):
	res = justdigits.match(num)
	if res is None:
		return '0'
	else:
		m = res.group()
		if len(m) < 2:
			return '0'
		else:
			return m[:-2]+'00'

df = wibr[(~wibr.ALD.isna()) & (wibr.StreetAddress==True)].sample(n=100000)
model = 'ALD ~ block_street'
y, X = patsy.dmatrices(model, df, return_type='dataframe')

if __name__ == '__main__':
	addr = '2338 E Oklahoma av, Milwaukee WI 53207'
	print(decomposeAddress(addr))