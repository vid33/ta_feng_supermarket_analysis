import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
  Basic visualisation to get a feel for the data.
  NOTE: Columns are
  ['Transaction Date', 'Customer ID', 'Age', \
  'Residence Area', 'Product Subclass', 'Product ID', 'Amount', 'Asset Price?', 'Sales Price']
"""
# Pick one of 'nov_00', 'dec_00', 'jan_01', 'feb_01', 'all', or 'all_periodic'
data_period= 'all'

#prevents crash when closing plot windows
mpl.rcParams['backend'] = "qt4agg"
mpl.rcParams['backend.qt4'] = "PySide"

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

#### PRINT SOME BASIC OBSERVATIONS 
print('Data set:', data_period)
print('Total number of data points: {:.5e}'.format(len(df.index)))
print('Total sales: {:.5e}'.format(df['Sales Price'].sum()))
print('Total value of sold assets(?): {:.5e}'.format(df['Asset Price?'].sum()))
print('Total margins: {:.5e}'.format( df['Sales Price'].sum()- df['Asset Price?'].sum() ))
print('Margin ratio: {0:4.4f}'.format( \
	(df['Sales Price'].sum()- df['Asset Price?'].sum())/df['Sales Price'].sum() ) )
print('Number of unique customers:', len(df['Customer ID'].unique()))
print('Number of customer visits:', df.groupby(['Customer ID'])['Transaction Date'].nunique().sum())
print('Number of products:', len(df['Product ID'].unique()))
print('Number of product subclasses:', len(df['Product Subclass'].unique()))
#####################################

#### PLOTS:

# Plot daily sales and margins (not sure if these really are margins, but looks like it)
df_by_date = df.groupby('Transaction Date')
daily_assets_and_sales = df_by_date['Asset Price?', 'Sales Price'].aggregate(np.sum)

fig1 = plt.figure()
plt.bar(daily_assets_and_sales.index, daily_assets_and_sales['Sales Price'], label = 'Sales Price')
plt.bar(daily_assets_and_sales.index, \
	daily_assets_and_sales['Sales Price'] - daily_assets_and_sales['Asset Price?'], color='red', \
	label = 'Margin(?)')
plt.ylabel('Total sales')
plt.title('Daily total sales and margins(?)')
plt.legend(loc = 'upper left')
fig1.autofmt_xdate() #rotates date labels so they don't overlap
fig1.show()

# POTENTIALLY SLOW(!!!) Daily sales vs. margins 
fig1a = plt.figure()
plt.scatter(df['Sales Price'], df['Sales Price'] - df['Asset Price?'])
plt.xlabel('Sales Price')
plt.ylabel('Margin (?)')
fig1a.show()

# Margin ratios (?)
fig1b = plt.figure()
plt.bar(daily_assets_and_sales.index, \
	(daily_assets_and_sales['Sales Price']-daily_assets_and_sales['Asset Price?'])/daily_assets_and_sales['Sales Price'],\
	label = 'Sales Price')
plt.ylabel('Percentage')
plt.title('Margin percentages(?)')
fig1b.autofmt_xdate() #rotates date labels so they don't overlap
fig1b.show()

# Number of daily transactions
daily_transactions = df.groupby('Transaction Date').size()

fig2 = plt.figure()
plt.bar(daily_transactions.index, daily_transactions)
plt.ylabel('Transactions')
plt.title('Number of daily transactions')
fig2.autofmt_xdate() #rotates date labels so they don't overlap
fig2.show()

# Number of items bought per shopping trip (assuming one trip per day)
# Should we also consider number of distinct items bought ?
df_by_customer_and_date = df.groupby(['Customer ID', 'Transaction Date'])
items_per_visit = df_by_customer_and_date['Amount'].aggregate(np.sum)
# Histogram of number of customers vs. items bought per visit - loads of outliers:
fig3 = plt.figure()
binwidth = 10
plt.xlabel('Items bought per visit')
plt.ylabel('Frequency')
plt.hist(items_per_visit, range(min(items_per_visit.values), max(items_per_visit.values)+binwidth, binwidth))
fig3.show()

# Numer of visits made by each customer - Most customers visit only once or twice.
df_by_customer = df.groupby(['Customer ID'])
customer_visits = df_by_customer['Transaction Date'].nunique()
fig3a = plt.figure()
binwidth = 1
plt.xlabel('Customer visits')
plt.ylabel('Frequency')
plt.hist(customer_visits, range(min(customer_visits.values), max(customer_visits.values)+binwidth, binwidth))
fig3a.show()

# Average price of item histogram
df_by_product = df.groupby(['Product ID'])
price_per_item = df_by_product['Sales Price per Item'].mean()
fig3b = plt.figure()
plt.xlabel('Average Item Price')
plt.ylabel('Frequency')
plt.hist(price_per_item.values, bins=100)
fig3b.show()

# Average price within product subclass histogram
df_by_product_class = df.groupby(['Product Subclass'])
avg_price_in_subclass = df_by_product_class['Sales Price per Item'].mean()
fig3c = plt.figure()
plt.xlabel('Average price of item in subclass')
plt.ylabel('Frequency')
plt.hist(avg_price_in_subclass.values, bins=50)
fig3c.show()

# Number of sales of each product, ordered
product_sales_no = df['Product ID'].value_counts().values
fig3d = plt.figure()
plt.bar(np.arange(2000), product_sales_no[0:2000])
plt.xlabel('Products')
plt.ylabel('Number of Sales')
plt.title('Total number of sales for each product, ordered')
fig3d.show()

"""for key, item in df_by_product:
	print(df_by_product.get_group(key)) """

# Total sales by age group
df_by_age = df.groupby('Age')
total_sales_by_age = df_by_age['Sales Price'].aggregate(np.sum)

fig4 = plt.figure()
binwidth = 10
plt.bar(np.arange(len(total_sales_by_age.index)), total_sales_by_age.values, 1, align='center')
plt.xticks(np.arange(len(total_sales_by_age.index)), total_sales_by_age.index)
plt.xlabel('Age group')
plt.ylabel('Total sales')
fig4.autofmt_xdate()
fig4.show()

# Total sales by postcode (ordered by distance); except "G"="other postcode", "H"=unknown
df_by_postcode = df.groupby('Residence Area')
total_sales_by_postcode = df_by_postcode['Sales Price'].aggregate(np.sum)

fig5 = plt.figure()
binwidth = 10
plt.bar(np.arange(len(total_sales_by_postcode.index)), total_sales_by_postcode.values, 1, align='center')
plt.xticks(np.arange(len(total_sales_by_postcode.index)), total_sales_by_postcode.index)
plt.xlabel('Postcode, by distance (G=other, H=unknown)')
plt.ylabel('Total sales')
fig5.show()





