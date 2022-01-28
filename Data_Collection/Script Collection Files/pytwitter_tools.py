"""
Make sure all imports are installed

Use your own keys
"""

import tweepy as tw
import pandas as pd
import json
import matplotlib.pyplot as plt

import networkx as nx

import twitter_user as tu

depression_phrases = ["i have depression",
                      "i am depressed",
                      "i'm depressed",
                      "im depressed",
                      "i am feeling depressed",
                      "im feeling depressed",
                      "i'm feeling depressed",
                      "i have clinical depression",
                      "i am clinically depressed",
                      "im clinically depressed",
                      "i'm clinically depressed",
                      "i am diagnosed with depression",
                      "im diagnosed with depression",
                      "i'm diagnosed with depression",
                      "my depression",
                      ]


def authorize():
    consumer_key = 'c7kjCYQ0gIY60zBP2Ng4kPC7K'  # gpric024 account
    consumer_key_secret = 'O4vSqiZpb0ccsdEPxub6AUoJovR7eXmHPreE4fUfniMMfzhtAj'
    access_token = '1437574559121776643-dLFJLlQUi4O72PTfBJlFk7GzudwXAF'
    acess_token_secret = 'wGDxMnGILcwtdnkdqw7o4AhbtxJr9W8DaOmuVlpCrVecC'
    auth = tw.OAuthHandler(consumer_key, consumer_key_secret)
    auth.set_access_token(access_token, acess_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    return api

# def authorize():
#     consumer_key = 'X8y5NRfbKvdbfJoeqH5wcwKf9' #UNI or Fil account
#     consumer_key_secret = 'zcoUREqyclv3SePHVJCaiePUzneCSyGf5tZG91n0mVRJdmjlv0'
#     access_token = '1443757769824165889-cdJIn7XoLOLJuCEgMJ0yKbIMYVvxxh'
#     acess_token_secret = 'T18YIqFWG50SowDLZWB0BOOfUKR750YYVBntdU00Smt16'
#     auth = tw.OAuthHandler(consumer_key, consumer_key_secret)
#     auth.set_access_token(access_token, acess_token_secret)
#     api = tw.API(auth, wait_on_rate_limit=True)
#     return api


def phrase_search(api):
    '''
    Returns user of the last person who tweeted a specific phrase, and the list of fetched tweets
    '''
    fetched_tweets = api.search_tweets("\"i have depression\"", count=1, result_type="recent", lang="en")
    print(fetched_tweets[0]._json['text'])
    anchor_id = fetched_tweets[0]._json['user']['id']

    return anchor_id


def get_friends_list(api, user):
    '''
    Takes an object of class User as input
    Returns a list of the users friends ids as strings
    '''
    anchor_friends_ids = []
    try:

        anchor_friends_raw = api.get_friends(user_id=user.get_UserID(), count=200)
    except Exception as e:
        print(e)
        return anchor_friends_ids

    for i in anchor_friends_raw:
        anchor_friends_ids.append(i._json['id'])

    return anchor_friends_ids


# def check_protected(api, userID):
#     '''
#     Check if a twitter acount is set to private by their UserID
#     '''
#     try:
#         user = api.get_user(user_id=userID)
#         if user._json['protected'] == 'True':
#             return True
#     except Exception as e:
#         print(e)
#         return False
#     return False


def get_tweets_by_id(api, userID):
    '''
    Get all tweets from a user by their UserID
    '''
    try:
        all_tweets = api.user_timeline(user_id=userID,
                                       # 200 is the maximum allowed count
                                       # try setting it to 1000
                                       count=5000,
                                       include_rts=False,
                                       # Necessary to keep full_text
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode='extended'
                                       )
        return all_tweets

    except Exception as e:
        return []


def depression_check_user(api, userID):
    """
    Takes UserID as input
    Returns boolean to identicate wether or not they are depressed
    """
    tweets = get_tweets_by_id(api, userID)
    depression_status = False

    for info in tweets:
        for depression_phrase in depression_phrases:
            if depression_phrase in info.full_text.lower():
                # print(info.full_text)
                depression_status = True

    return depression_status


def create_Set_And_Graph(api, user, G, allUsers):

    # lvl1_friends is a list of USER IDS
    lvl1_friends = get_friends_list(api, user)

    # lvl1_users is a list of USER OBJECTS which are friends of USER
    lvl1_users = []

    for newUserID in lvl1_friends:
        print(newUserID)
        success = False
        # Checking if the new user is protected before instantiating their user id as an object of class User
        try:
            newUser = tu.User(userID=newUserID,
                              depression_status=depression_check_user(api, newUserID))
            success = True
        except Exception as e:
            print(e)
            success = False
        if success:
            lvl1_users.append(newUser)
            if newUserID not in allUsers:
                allUsers.add(newUserID)
                G.add_node(newUser)
                G.add_edge(newUser, user)
        if len(allUsers) >= 1000:
            break #EVEN HARDER CAP 1000
    user.set_followers(lvl1_users)
    return user, G, allUsers


def get_Outside_Edges(api, G):
    lst_users = []
    for node in G:
        if G.degree(node) == 1:
            lst_users.append(node)
    return lst_users


def expand_Graph(api, G, allUsers, maxLayers, maxUsers):
    counter = 0
    while counter < maxLayers and len(allUsers) < maxUsers:
        print(len(allUsers))
        outside_nodes = get_Outside_Edges(api, G)
        for user in outside_nodes:
            temp, G, allUsers = create_Set_And_Graph(api, user, G, allUsers)
        counter += 1


def pull_timeline(api, userID):
    tweet_history = ""
    try:
        tweets = get_tweets_by_id(api, userID)
        for tweet in tweets:
            tweet_history += (json.dumps(tweet._json) + "\n")
        return tweet_history
    except Exception as e:
        return tweet_history

def findNode(userID, G):
    for n in G:
        if userID == n.get_UserID():
            return n
    return None


def add_Edges(api, node, G):
    lst = node.get_followers()
    cnt = 1
    total = len(lst)
    for n in lst:
        print("Friend " + str(cnt) + "/" + str(total))
        cnt += 1
        # friendIds = get_friends_list(api, n)
        friend = findNode(n.get_UserID(), G)
        if friend is not None:
            if not G.has_edge(node, friend):
                G.add_edge(node, friend)
                print("Connection Added")
        # for iD in friendIds:
        #     tmp = findNode(iD, G)
        #     if tmp != False:
        #         if not G.has_edge(friend, tmp):
        #             G.add_edge(friend, tmp)
        #             print("Connection Added")
        #         print("Connection already added")


def main():
    api = authorize()
    print(type(api))

    #anchor_id = phrase_search(api)
    anchor_id = 288390095
    print(anchor_id)

    anchorUser = tu.User(userID=anchor_id,
                         depression_status=True)

    allUsers = set()
    G = nx.Graph()
    G.add_node(anchorUser)
    allUsers.add(anchorUser.get_UserID())

    temp, G, allUsers = create_Set_And_Graph(api, anchorUser, G, allUsers)

    # Maximum amount of users we want to reach or max Layer depth. The cap is not hard capped. It will most likely acheive way more than 1000 users.
    expand_Graph(api, G, allUsers, 1, 1000)

    df = pd.DataFrame(columns=['UserID', 'TweetHistory', 'Depression_Status'])
    counter = 1
    graphSize = G.size()
    for node in G.copy():
        print(str(counter) + "/" + str(graphSize) + "|| Adding Edges to Node: " + str(node))
        add_Edges(api, node, G)
        my_id = node.get_UserID()
        timeline = pull_timeline(api, my_id)
        depression_stat = node.get_depression_status()
        df2 = pd.DataFrame([[my_id, timeline, depression_stat]],
                           columns=['UserID', 'TweetHistory', 'Depression_Status'])
        df = df.append(df2, ignore_index=True)
        counter += 1

    counter = 1
    #Adding outside edges
    lst = get_Outside_Edges(api,G)
    for node in lst:
        print(str(counter)+"/"+str(len(lst)) +" Outside edge")
        counter+=1
        try:
            friendIds = get_friends_list(api, node)
        except Exception as e:
            friendIds = []
        for id in friendIds:
            friend = findNode(id, G)
            if friend is not None:
                if not G.has_edge(node, friend):
                    G.add_edge(node, friend)
                    print("Connection Added to outside edge")

    nx.write_edgelist(G, path="GeneratedGraph.edgelist")
    df.to_csv("User_Timeline_Label.csv")

    # print(anchorUser.get_depression_status())
    color_map = ['red' if node.get_depression_status() == True else 'green' for node in G]
    nx.draw(G, node_color=color_map)
    plt.show()


if __name__ == "__main__":
    main()
