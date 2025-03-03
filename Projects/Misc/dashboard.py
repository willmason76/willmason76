import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Generic Dashboard',
                   page_icon=':bar_chart:',
                   layout='wide')

df = pd.read_excel(
    io='Sample JobCost.xlsx',
    engine='openpyxl',
    sheet_name='Data',
    skiprows=0,
    usecols='B:Q',
    nrows=23318)

st.sidebar.header('Please Filter Here:')

category = st.sidebar.multiselect(
    'Select Category:',
    options = df['Category'].unique(),
    default = df['Category'].unique())

transaction_type = st.sidebar.multiselect(
    'Select Type:',
    options = df['TransType'].unique(),
    default = df['TransType'].unique())

df_selection = df.query('Category == @category & TransType == @transaction_type')

#st.dataframe(df) #remove the initial hashtag at begin of line if you want to see the main df
#st.dataframe(df_selection) #remove the initial hashtag at begin of line if you want to see the selection df

#MAINPAGE#
st.title(':bar_chart: Labor Cost Dashboard')
st.markdown('##')

#KPI Section
total_cost = int(df_selection['Amount'].sum())
avg_ticket = round(df_selection['Amount'].mean(),2)

left_column, middle_column, = st.columns(2)
with left_column:
    st.subheader('Total Cost:')
    st.subheader(f'US $ {total_cost:,}')
with middle_column:
    st.subheader('Average Invoice:')
    st.subheader(f'US $ {avg_ticket:,}')

st.markdown('---') #this just puts divider between the columns

#BARCHART_1
cost_by_category = (df_selection.groupby(by = ['Category']).sum()[['Amount']].sort_values(by = 'Category'))
fig_cost_by_cat = px.bar(
    cost_by_category,
    x = 'Amount',
    y = cost_by_category.index,
    orientation = 'h',
    title = '<b>Cost by Category</b>',
    color_discrete_sequence = ['#0083B8'] * len(cost_by_category), #hexadecimal code multiplied by length of dataframe
    template = 'plotly_white')

#BARCHART_2
cost_by_period = (df_selection.groupby(by = ['Year']).sum()[['Amount']])
fig_yearly_cost = px.bar(
    cost_by_period,
    x = 'Amount',
    y = cost_by_period.index,
    orientation = 'h',
    title = '<b>Cost by Year</b>',
    color_discrete_sequence = ['#0083B8'] * len(cost_by_period),
    template = 'plotly_white')

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_cost_by_cat, use_container_width = True)
right_column.plotly_chart(fig_yearly_cost, use_container_width = True)

#HIDE STREAMLIT SYTLE - this just takes away some of the visible things, doesn't really matter, can run without
hide_st_style = '''
    <style>
    #MainMenu {visibility : hidden;}
    footer {visibility : hidden}
    header {visibility : hidden}
    </style>
    '''
st.markdown(hide_st_style, unsafe_allow_html = True)

#note that the tutorial added a .TOML file to change colors, but isn't working
