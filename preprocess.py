"""
This script pre-processes data for a single month, (nov_00, dec_00, jan_01, feb_01), 
or for the entire four month period (all), and saves the pandas dataframe to csv.

NOTE 1:  When the 4-month period is selected an 'all_periodic.cvs' dataset is saved, 
where, in addition to dropping a number of weeks containing the above holidays, we
also make sure the data is periodic in weeks. More details below.

NOTE 1a: Weekly periodicity is a very obvious feature. Most shopping is almost always done on Sundays.

NOTE 2: It looks like "Asset Price" is the price at which the item was bought by supermarket, i.e. 
"Margin" = "Sales Price" - "Asset Price", but this isn't completely clear.

NOTE 3: Sales price is definitely the price for the total number of items bought. No need to multiply
by 'Amount' 

NOTE 4: A few products belong to more than one product subclass. This is likely an error.
See below for details.

NOTE 5: Checked: no cases of different age group and residence area entries for single customer.

"""
import pandas as pd
from scipy.stats import mode

#Pick a month, or the entire four month period
data_period= 'nov_00'
data_files = {'nov_00':'D11-02/D11', 'dec_00':'D11-02/D12', \
               'jan_01':'D11-02/D01', 'feb_01':'D11-02/D02', 'all':'select_all' }

if data_period == 'all':
  df_00_nov = pd.read_csv('D11-02/D11', sep=';',encoding = 'ISO-8859-1')
  df_00_dec = pd.read_csv('D11-02/D12', sep=';',encoding = 'ISO-8859-1')
  df_01_jan = pd.read_csv('D11-02/D01', sep=';',encoding = 'ISO-8859-1')
  df_01_feb = pd.read_csv('D11-02/D02', sep=';',encoding = 'ISO-8859-1')
  df = pd.concat([df_00_nov, df_00_dec, df_01_jan, df_01_feb])
else:
  df = pd.read_csv(data_files[data_period], sep=';', encoding = 'ISO-8859-1')

df.columns = ['Transaction Date', 'Customer ID', 'Age', \
  'Residence Area', 'Product Subclass', 'Product ID', 'Amount', 'Asset Price?', 'Sales Price']

print('Period:', data_period)
print('Size of initial data set', len(df.index))

# CLEANING UP DATA #################

df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

#Strip trailing whitespace
df['Age'] =  df['Age'].str.strip()
df['Residence Area'] =  df['Residence Area'].str.strip()

# from closest to most distant - except "G" corresponds to "other postcode" and "H" is unknown
df['Residence Area'] = pd.Categorical(df['Residence Area'], ['E','F','D','A','B','C','G','H'])

#Rename the age column. NOTE: K is not specified in readme file
age_dict = {'A':'<25', 'B':'25-29', 'C':'30-34', 'D':'35-39', 'E':'40-44', \
        'F':'45-49', 'G':'50-54', 'H':'55-59', 'I':'60-64', 'J':'>65', 'K':'unknown'}

df['Age'] = df['Age'].astype('category')
df['Age'] = df['Age'].apply(lambda x: age_dict[x])

# Some product IDs appears in more than product subclass. This is 
# very likely a mistake, usually only one digit is off, so it looks like these were perhaps 
# manually entered and were incorrectly keyed in

dodgy_IDs = df[['Product ID', 'Product Subclass']].groupby('Product ID').std()
dodgy_IDs = dodgy_IDs[dodgy_IDs['Product Subclass'] > 1].index

dropout_no = 0
for i in dodgy_IDs:
  dropout_no += len(df[df['Product ID'] == i].index)
  product_subclass_modes = df[(df['Product ID'] == i)]['Product Subclass'].mode()

  if len (product_subclass_modes) > 0: #in very rare cases there are only two items
    df.ix[df['Product ID'] == i, 'Product Subclass'] = product_subclass_modes[0]
  else: # in which case, we drop this product entirely
    df = df[df['Product ID'] != i]
    
  df = df.reset_index(drop=True) 
  
# Sort by date to avoid pitfalls- original mostly sorted but in fact not everywhere
df =  df = df.sort_values('Transaction Date');
########################################

print('Size of processed data set', len(df.index))
print("Percentage of data points with problematic product subclass", 100*dropout_no/len(df.index))

df.to_csv('data/' + data_period + '.csv', index=False)
df.to_csv('data/' + data_period + '.csv' + '.bz2', compression='bz2', index=False)

""" 
  Drop irregular looking week, and make sure data is periodic. 
  So we drop weeks surrounding Chinese New Year, and Taiwan constitution day,
  and drop the final day 2001-02-28, so now the data starts on Thu 2000-11-02 and
  ends on Wed 2001-02-28, and comprises 12 weeks altogether
"""
if data_period == 'all':
  print('Dropping a number of weeks surrounding holidays, and making sure data is periodic in weeks.')
  df_periodic =  df[('2000-12-17' >= df['Transaction Date']) | (df['Transaction Date'] >= '2001-01-08')]
  df_periodic =  df_periodic[('2001-01-14' >= df_periodic['Transaction Date']) | \
              (df_periodic['Transaction Date'] >= '2001-01-29')]
  df_periodic = df_periodic[df_periodic['Transaction Date'] != '2000-11-01']
  print('Size of processed data set', len(df_periodic.index))
  df_periodic.to_csv('data/' + 'all_periodic' + '.csv', index=False)
  df_periodic.to_csv('data/' + 'all_periodic' + '.csv' + '.bz2', compression='bz2', index=False)




