import pytwitter_tools as tools
import twitter_user as tu

import tweepy as tw
import pandas as pd
import twitter
import json
import requests
import json

consumer_key='X8y5NRfbKvdbfJoeqH5wcwKf9'
consumer_key_secret='zcoUREqyclv3SePHVJCaiePUzneCSyGf5tZG91n0mVRJdmjlv0'
access_token='1443757769824165889-cdJIn7XoLOLJuCEgMJ0yKbIMYVvxxh'
acess_token_secret='T18YIqFWG50SowDLZWB0BOOfUKR750YYVBntdU00Smt16'

auth = tw.OAuthHandler(consumer_key,consumer_key_secret)
auth.set_access_token(access_token,acess_token_secret)
api = tw.API(auth,wait_on_rate_limit=True)

anchorID = tools.phrase_search("i have depression",api)

anchorUser = User(userID=anchorID,
                  depression_status=True)

friends_list = tools.get_friends_list(anchorUser,api)