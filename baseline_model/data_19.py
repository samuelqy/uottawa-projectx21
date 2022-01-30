import pandas as pd
import csv
import sys
import os
import json
import requests

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
        

def getstuff(filename = "User_Timeline_Label.csv"):
    with open(filename, "r") as csvfile:

        datareader = csv.reader(csvfile)
        yield next(datareader)

        for row in datareader:
            yield row


def extractImg(target):

    user_dir = target
    text_dir = os.path.join(user_dir, 'timeline.txt')

    with open(text_dir) as f:

        tweets = f.readlines()
        
        for tweet in tweets:

            j_tweet = json.loads(tweet)
            entities = j_tweet['entities']

            if "media" in entities:

                if entities['media'][0]['type'] == "photo":

                    link = entities['media'][0]['media_url']
                    img_data = requests.get(link).content
                    image_dir = os.path.join(user_dir, j_tweet['id_str']+'.jpg')

                    with open(image_dir, 'wb') as handler:
                        handler.write(img_data)

            else:
                pass

def fillUser(row):

   current_dir = os.getcwd()
   pos_dir = os.path.join(current_dir, r'positive')
   neg_dir = os.path.join(current_dir, r'negative')
   user_id = row[1]
   row_tweets = row[2]
   depression = row[3]
   
   target = os.path.join(pos_dir if depression == "True" else neg_dir, user_id)

   if not os.path.exists(target):
      os.makedirs(target)

   file_dir = os.path.join(target, r'timeline.txt')
   file = open(file_dir, "w")
   file.write(row_tweets)
   file.close()
   extractImg(target)



c = 0
for row in getstuff():


    if c != 0:
        fillUser(row)

    if c%10 == 0:
        print("{}th user".format(c))
    c += 1

    