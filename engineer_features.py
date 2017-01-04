import pandas as pd
import numpy as np
import random

from imp import reload

# Engineer some features and save resulting dictionary to file

def generate():
	"""
	NOTE: Columns are
  	['Transaction Date', 'Customer ID', 'Age', \
  		'Residence Area', 'Product Subclass', 'Product ID', 'Amount', 'Asset Price?', 'Sales Price']
	"""	

	# Pick one of 'nov_00', 'dec_00', 'jan_01', 'feb_01', 'all', or 'all_periodic'
	data_period= 'all_periodic'
	print('Data set:', data_period)

	#Load data from file generated in pre-processing stage
	# df = pd.read_csv('data/' + data_period + '.csv' + '.bz2', compression='bz2')
	df = pd.read_csv('data/' + data_period + '.csv', parse_dates=['Transaction Date'])

	# Re-define as categorical, with specified order
	df['Age'] = pd.Categorical(df['Age'], categories=['<25', '25-29', '30-34', '35-39', '40-44', \
	        '45-49', '50-54', '55-59', '60-64', '>65', 'unknown'], ordered=False)
	# from closest to most distant - except "G" corresponds to "other postcode" and "H" is unknown
	df['Residence Area'] = pd.Categorical(df['Residence Area'], \
		categories=['E','F','D','A','B','C','G','H'], ordered=False)

	# Calculate individual item prices
	df['Sales Price per Item'] = df['Sales Price']/df['Amount']
	df['Asset Price per Item'] = df['Asset Price?']/df['Amount']

	"""
	 I will remove all customers with 'Age' = 'undefined'. This is not the best solution
	 ultimately, but 1) age is an important feature and 2) we can learn the age based on other factors, 
	 and retrofit. Come back to this. Corresponds to approximately 2 percent of dataset.
	"""
	df_tmp = df[['Customer ID', 'Age']].drop_duplicates()
	print('Fraction of customers with age unspecified {0:4.4f}'.format( \
		len(df_tmp[df_tmp['Age'] == 'unknown'])/len(df['Customer ID'].unique()) ) )

	print('Removing Age=unknown entries...')
	df = df[df['Age']!='unknown']

	df_tmp = df[['Customer ID', 'Age']].drop_duplicates()
	features = pd.Series(df_tmp['Age'].values, index = df_tmp['Customer ID']).to_dict()
	del df_tmp

	#FEATURE 1: Age
	for customer, age in features.items():
		tmp_list = []
		if age == '<25':
			tmp_list.append(20)
		elif age == '25-29':
			tmp_list.append(27)
		elif age == '30-34':
			tmp_list.append(32)
		elif age == '35-39':
			tmp_list.append(37)
		elif age == '40-44':
			tmp_list.append(42)
		elif age == '45-49':
			tmp_list.append(47)
		elif age == '50-54':
			tmp_list.append(52)
		elif age == '55-59':
			tmp_list.append(57)
		elif age == '60-64':
			tmp_list.append(62)
		elif age == '>65':
			tmp_list.append(70)
		features[customer] = tmp_list

	# FEATURE 2: Distance from store
	df_tmp = df[['Customer ID', 'Residence Area']].drop_duplicates()

	""" Not the best - "G" corresponds to "other postcode" and "H" is unknown, perhaps
	 this can also be learned. """
	distance_dict = {'E':1, 'F':2, 'D':3, 'A':4, 'B':5, 'C':6, 'G':8, 'H':3.5 }
	for i in df_tmp.index:
		features[df_tmp.ix[i]['Customer ID']].append(distance_dict[df_tmp.ix[i]['Residence Area']])
	del df_tmp

	# FEATURE 3: Number of visits
	customer_visits = df.groupby(['Customer ID'])['Transaction Date'].nunique()
	for customer, visits in customer_visits.iteritems():
		features[customer].append(visits)

	# FEATURE 4: Total sales
	customer_tot_sales = df.groupby(['Customer ID'])['Sales Price'].aggregate(np.sum)
	for customer, tot_sales in customer_tot_sales.iteritems():
		features[customer].append(tot_sales)

	# FEATURE 5: Total number of items bought
	customer_tot_items = df.groupby(['Customer ID'])['Amount'].aggregate(np.sum)
	for customer, tot_items in customer_tot_items.iteritems():
		features[customer].append(tot_items)

	# FEATURE 6: Number of distinct product IDs
	customer_distinct_ids = df.groupby(['Customer ID'])['Product ID'].nunique()
	for customer, id_no in customer_distinct_ids.iteritems():
		features[customer].append(id_no)

	# FEATURE 7: Number of distinct product class IDs
	customer_distinct_classes = df.groupby(['Customer ID'])['Product Subclass'].nunique()
	for customer, class_no in customer_distinct_classes.iteritems():
		features[customer].append(class_no)

	# Many more needed......

	# Features array 

	features_array = np.empty(shape=(0, len(features[random.choice(list(features.keys()))]) ) )
	for keys, values in features.items():
		features_array = np.append(features_array, [values], axis=0)

	# Save for analysis
	outfile = 'data/' + 'features_array_' + data_period
	print('Writing to file:', outfile)
	np.save(outfile, features_array)

	return features_array


if __name__ == "__main__":
    generate()

