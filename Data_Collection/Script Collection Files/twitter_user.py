class User():
    
    def __init__(self,userID,depression_status=None,json_dump=None,followers=[],user_bio=""):
        self._userID = userID
        self._depression_status = depression_status
        self._json_dump = json_dump
        self._followers = followers
        self._user_bio = user_bio
        
    # getter method
    def get_UserID(self):
        return self._userID
    # getter method
    def get_depression_status(self):
        return self._depression_status
    # getter method
    def get_json_info(self):
        return self._json_info
    # getter method
    def get_followers(self):
        return self._followers
    # setter method
    def set_followers(self,follower_list):
        self._followers = follower_list
    # setter method
    def set_depression_status(self,new_status):
        self._depression_status = new_status
    # setter method
    def set_json_dump(self,json):
        self._json_dump= json
            
    def __str__(self):
        return 'UserID='+str(self._userID)+'\nDepression Status='+str(self._depression_status)