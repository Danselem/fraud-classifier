import pandas as pd
import numpy as np
import datasist as ds
import datasist.project as dp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from PIL import Image
train = dp.get_data("train_proc.csv", method='csv')
#train = pd.read_csv('../../data/processed/train_proc.csv')
# loading in the model to predict on the data
pickle_in = open('rf_model.pkl', 'rb')
classifier = pickle.load(pickle_in)

fts = ['AccountId',
    'SubscriptionId',
    'CustomerId',
    'Amount',
    'Value',
    'PricingStrategy',
    'TransactionStartTime_dow',
    'TransactionStartTime_doy',
    'TransactionStartTime_dom',
    'TransactionStartTime_hr',
    'TransactionStartTime_min',
    'TransactionStartTime_is_wkd',
    'TransactionStartTime_wkoyr',
    'TransactionStartTime_mth',
    'TransactionStartTime_qtr',
    'TransactionStartTime_yr',
    'ProductId',
    'ProviderId_ProviderId_2',
    'ProviderId_ProviderId_3',
    'ProviderId_ProviderId_4',
    'ProviderId_ProviderId_5',
    'ProviderId_ProviderId_6',
    'ProductCategory_data_bundles',
    'ProductCategory_financial_services',
    'ProductCategory_movies',
    'ProductCategory_other',
    'ProductCategory_retail',
    'ProductCategory_ticket',
    'ProductCategory_transport',
    'ProductCategory_tv',
    'ProductCategory_utility_bill',
    'ChannelId_ChannelId_2',
    'ChannelId_ChannelId_3',
    'ChannelId_ChannelId_4',
    'ChannelId_ChannelId_5']

def welcome():
    return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs

def prediction(
        AccountId, SubscriptionId, CustomerId, Amount, Value, PricingStrategy, TransactionStartTime_dow,
        TransactionStartTime_doy, TransactionStartTime_dom, TransactionStartTime_hr, TransactionStartTime_min,
        TransactionStartTime_is_wkd, TransactionStartTime_wkoyr, TransactionStartTime_mth, TransactionStartTime_qtr,
        TransactionStartTime_yr, ProductId, ProviderId_ProviderId_2, ProviderId_ProviderId_3, ProviderId_ProviderId_4,
        ProviderId_ProviderId_5, ProviderId_ProviderId_6, ProductCategory_data_bundles, ProductCategory_financial_services,
        ProductCategory_movies, ProductCategory_other, ProductCategory_retail, ProductCategory_ticket, ProductCategory_transport,
        ProductCategory_tv, ProductCategory_utility_bill, ChannelId_ChannelId_2, ChannelId_ChannelId_3, ChannelId_ChannelId_4,
        ChannelId_ChannelId_5):

    prediction = classifier.predict(
        [[AccountId, SubscriptionId, CustomerId, Amount, Value, PricingStrategy, TransactionStartTime_dow,
        TransactionStartTime_doy, TransactionStartTime_dom, TransactionStartTime_hr, TransactionStartTime_min,
        TransactionStartTime_is_wkd, TransactionStartTime_wkoyr, TransactionStartTime_mth, TransactionStartTime_qtr,
        TransactionStartTime_yr, ProductId, ProviderId_ProviderId_2, ProviderId_ProviderId_3, ProviderId_ProviderId_4,
        ProviderId_ProviderId_5, ProviderId_ProviderId_6, ProductCategory_data_bundles, ProductCategory_financial_services,
        ProductCategory_movies, ProductCategory_other, ProductCategory_retail, ProductCategory_ticket, ProductCategory_transport,
        ProductCategory_tv, ProductCategory_utility_bill, ChannelId_ChannelId_2, ChannelId_ChannelId_3, ChannelId_ChannelId_4,
        ChannelId_ChannelId_5]])
    print(prediction)
    return prediction


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Xente Fraud Classifier")
    
    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Xente Fraud Classifier ML App </h1>
    </div>
    """

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
    st.header("About the data")
    st.markdown('''
    The data has the following features:
    - __acousticness__: A measure from 0 to 1 to determine the track is acoustic or not.
    - __available_markets__: Country codes where the track is a available.
    - __danceability__: Rythmic score from 0-1 which determines how easier is it to dance.
    - __energy__: Amount of energy in the track from 0-1.
    - __instrumentalness__: Instrumentalness of the track.
    - __key__: Track key value.
    - __liveness__: Quality of track from 0-1.
    - __loudness__: Higher the value louder the track.
    - __name__: Name of the track.
    - __popularity__: Track popularity score.
    - __preview_url__: Preview link of the track.
    - __speechiness__: Amount of vocals in the track.
    - __tempo__: Tempo of the track.
    - __time_signature__: Length of the track in minutes.
    - __valence__: Positivity score of the track.''')
    st.subheader('Dataset Summary')
    st.subheader('First five data points')
    st.write(train.head())
    st.subheader('Random five data points')
    st.write(train.sample(5))
    st.subheader('Last five data points')
    st.write(train.tail())
    st.subheader('Shape of  data set')
    st.write(train.shape)
    st.subheader('Size of  data set')
    st.write(train.size)
    cols = ['ProductId', 'ProductCategory_tv', 'ProductCategory_ticket']
    #fig = plt.figure()
    for col in cols:
        fig = plt.figure(figsize=(8,6))
        g = sns.countplot(x=col, data = train)
        plt.xticks(rotation =90)
        st.write(fig)

    
    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    #for i in fts:
    #    vars()[i] = st.text_input(i, "Type Here")
    # d = {}
    # for name in fts:
    # #     #d[name] = st.number_input("Enter " + name + " value", key = name)
    #     vars()[name] = st.number_input("Enter " + name + " value", key = name)
    AccountId = st.number_input("Enter {} value between {} and {}".format(fts[0], train.iloc[:, 0].min(), train.iloc[:, 0].max()), train.iloc[:, 0].min(), train.iloc[:, 0].max(), key = fts[0],) 
    SubscriptionId = st.number_input("Enter {} value between {} and {}".format(fts[1], train.iloc[:, 1].min(), train.iloc[:, 1].max()), train.iloc[:, 1].min(), train.iloc[:, 1].max(), key = fts[1],)
    CustomerId = st.number_input("Enter {} value between {} and {}".format(fts[2], train.iloc[:, 2].min(), train.iloc[:, 2].max()), train.iloc[:, 2].min(), train.iloc[:, 2].max(), key = fts[2],) 
    Amount = st.number_input("Enter {} value between {} and {}".format(fts[3], train.iloc[:, 3].min(), train.iloc[:, 3].max()), train.iloc[:, 3].min(), train.iloc[:, 3].max(), key = fts[3]) 
    Value = st.number_input("Enter {} value between {} and {}".format(fts[4], train.iloc[:, 4].min(), train.iloc[:, 4].max()), train.iloc[:, 4].min(), train.iloc[:, 4].max(), key = fts[4]) 
    PricingStrategy = st.number_input("Enter {} value between {} and {}".format(fts[5], train.iloc[:, 5].min(), train.iloc[:, 5].max()), train.iloc[:, 5].min(), train.iloc[:, 5].max(), key = fts[5]) 
    TransactionStartTime_dow = st.number_input("Enter {} value between {} and {}".format(fts[6], train.iloc[:, 6].min(), train.iloc[:, 6].max()), train.iloc[:, 6].min(), train.iloc[:, 6].max(), key = fts[6])
    TransactionStartTime_doy = st.number_input("Enter {} value between {} and {}".format(fts[7], train.iloc[:, 7].min(), train.iloc[:, 7].max()), train.iloc[:, 7].min(), train.iloc[:, 7].max(), key = fts[7]) 
    TransactionStartTime_dom = st.number_input("Enter {} value between {} and {}".format(fts[8], train.iloc[:, 8].min(), train.iloc[:, 8].max()), train.iloc[:, 8].min(), train.iloc[:, 8].max(), key = fts[8]) 
    TransactionStartTime_hr = st.number_input("Enter {} value between {} and {}".format(fts[9], train.iloc[:, 9].min(), train.iloc[:, 9].max()), train.iloc[:, 9].min(), train.iloc[:, 9].max(), key = fts[9])
    TransactionStartTime_min = st.number_input("Enter {} value between {} and {}".format(fts[10], train.iloc[:, 10].min(), train.iloc[:, 10].max()), train.iloc[:, 10].min(), train.iloc[:, 10].max(), key = fts[10])
    TransactionStartTime_is_wkd = st.number_input("Enter {} value between {} and {}".format(fts[11], train.iloc[:, 11].min(), train.iloc[:, 11].max()), train.iloc[:, 11].min(), train.iloc[:, 11].max(), key = fts[11])
    TransactionStartTime_wkoyr = st.number_input("Enter {} value between {} and {}".format(fts[12], train.iloc[:, 12].min(), train.iloc[:, 12].max()), train.iloc[:, 12].min(), train.iloc[:, 12].max(), key = fts[12])
    TransactionStartTime_mth = st.number_input("Enter {} value between {} and {}".format(fts[13], train.iloc[:, 13].min(), train.iloc[:, 13].max()), train.iloc[:, 13].min(), train.iloc[:, 13].max(), key = fts[13])
    TransactionStartTime_qtr = st.number_input("Enter {} value between {} and {}".format(fts[14], train.iloc[:, 14].min(), train.iloc[:, 14].max()), train.iloc[:, 14].min(), train.iloc[:, 14].max(), key = fts[14])
    TransactionStartTime_yr = st.number_input("Enter {} value between {} and {}".format(fts[15], train.iloc[:, 15].min(), train.iloc[:, 15].max()), train.iloc[:, 15].min(), train.iloc[:, 15].max(), key = fts[15]) 
    ProductId = st.number_input("Enter {} value between {} and {}".format(fts[16], train.iloc[:, 16].min(), train.iloc[:, 16].max()), train.iloc[:, 16].min(), train.iloc[:, 16].max(), key = fts[16])
    ProviderId_ProviderId_2 = st.number_input("Enter {} value between {} and {}".format(fts[17], train.iloc[:, 17].min(), train.iloc[:, 17].max()), train.iloc[:, 17].min(), train.iloc[:, 17].max(), key = fts[17])
    ProviderId_ProviderId_3 = st.number_input("Enter {} value between {} and {}".format(fts[18], train.iloc[:, 18].min(), train.iloc[:, 18].max()), train.iloc[:, 18].min(), train.iloc[:, 18].max(), key = fts[18])
    ProviderId_ProviderId_4 = st.number_input("Enter {} value between {} and {}".format(fts[19], train.iloc[:, 19].min(), train.iloc[:, 19].max()), train.iloc[:, 19].min(), train.iloc[:, 19].max(), key = fts[19])
    ProviderId_ProviderId_5 = st.number_input("Enter {} value between {} and {}".format(fts[20], train.iloc[:, 20].min(), train.iloc[:, 20].max()), train.iloc[:, 20].min(), train.iloc[:, 20].max(), key = fts[20]) 
    ProviderId_ProviderId_6 = st.number_input("Enter {} value between {} and {}".format(fts[21], train.iloc[:, 21].min(), train.iloc[:, 21].max()), train.iloc[:, 21].min(), train.iloc[:, 21].max(), key = fts[21]) 
    ProductCategory_data_bundles = st.number_input("Enter {} value between {} and {}".format(fts[22], train.iloc[:, 22].min(), train.iloc[:, 22].max()), train.iloc[:, 22].min(), train.iloc[:, 22].max(), key = fts[22])
    ProductCategory_financial_services = st.number_input("Enter {} value between {} and {}".format(fts[23], train.iloc[:, 23].min(), train.iloc[:, 23].max()), train.iloc[:, 23].min(), train.iloc[:, 23].max(), key = fts[23])
    ProductCategory_movies = st.number_input("Enter {} value between {} and {}".format(fts[24], train.iloc[:, 24].min(), train.iloc[:, 24].max()), train.iloc[:, 24].min(), train.iloc[:, 24].max(), key = fts[24])
    ProductCategory_other = st.number_input("Enter {} value between {} and {}".format(fts[25], train.iloc[:, 25].min(), train.iloc[:, 25].max()), train.iloc[:, 25].min(), train.iloc[:, 25].max(), key = fts[25])
    ProductCategory_retail = st.number_input("Enter {} value between {} and {}".format(fts[26], train.iloc[:, 26].min(), train.iloc[:, 26].max()), train.iloc[:, 26].min(), train.iloc[:, 26].max(), key = fts[26]) 
    ProductCategory_ticket = st.number_input("Enter {} value between {} and {}".format(fts[27], train.iloc[:, 27].min(), train.iloc[:, 27].max()), train.iloc[:, 27].min(), train.iloc[:, 27].max(), key = fts[27])
    ProductCategory_transport = st.number_input("Enter {} value between {} and {}".format(fts[28], train.iloc[:, 28].min(), train.iloc[:, 28].max()), train.iloc[:, 28].min(), train.iloc[:, 28].max(), key = fts[28])
    ProductCategory_tv = st.number_input("Enter {} value between {} and {}".format(fts[29], train.iloc[:, 29].min(), train.iloc[:, 29].max()), train.iloc[:, 29].min(), train.iloc[:, 29].max(), key = fts[29]) 
    ProductCategory_utility_bill = st.number_input("Enter {} value between {} and {}".format(fts[30], train.iloc[:, 30].min(), train.iloc[:, 30].max()), train.iloc[:, 30].min(), train.iloc[:, 30].max(), key = fts[30])
    ChannelId_ChannelId_2 = st.number_input("Enter {} value between {} and {}".format(fts[31], train.iloc[:, 31].min(), train.iloc[:, 31].max()), train.iloc[:, 31].min(), train.iloc[:, 31].max(), key = fts[31])
    ChannelId_ChannelId_3 = st.number_input("Enter {} value between {} and {}".format(fts[32], train.iloc[:, 32].min(), train.iloc[:, 32].max()), train.iloc[:, 32].min(), train.iloc[:, 32].max(), key = fts[32])
    ChannelId_ChannelId_4 = st.number_input("Enter {} value between {} and {}".format(fts[33], train.iloc[:, 33].min(), train.iloc[:, 33].max()), train.iloc[:, 33].min(), train.iloc[:, 33].max(), key = fts[33])
    ChannelId_ChannelId_5 = st.number_input("Enter {} value between {} and {}".format(fts[34], train.iloc[:, 34].min(), train.iloc[:, 34].max()), train.iloc[:, 34].min(), train.iloc[:, 34].max(), key = fts[34])
    #sepal_length = st.text_input("Sepal Length", "Type Here")
    #sepal_width = st.text_input("Sepal Width", "Type Here")
    #petal_length = st.text_input("Petal Length", "Type Here")
    #petal_width = st.text_input("Petal Width", "Type Here")
    
    result =""
    
    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(
        AccountId, SubscriptionId, CustomerId, Amount, Value, PricingStrategy, TransactionStartTime_dow,
        TransactionStartTime_doy, TransactionStartTime_dom, TransactionStartTime_hr, TransactionStartTime_min,
        TransactionStartTime_is_wkd, TransactionStartTime_wkoyr, TransactionStartTime_mth, TransactionStartTime_qtr,
        TransactionStartTime_yr, ProductId, ProviderId_ProviderId_2, ProviderId_ProviderId_3, ProviderId_ProviderId_4,
        ProviderId_ProviderId_5, ProviderId_ProviderId_6, ProductCategory_data_bundles, ProductCategory_financial_services,
        ProductCategory_movies, ProductCategory_other, ProductCategory_retail, ProductCategory_ticket, ProductCategory_transport,
        ProductCategory_tv, ProductCategory_utility_bill, ChannelId_ChannelId_2, ChannelId_ChannelId_3, ChannelId_ChannelId_4,
        ChannelId_ChannelId_5)
    st.success('The output is {}'.format(result))
    
if __name__=='__main__':
    main()