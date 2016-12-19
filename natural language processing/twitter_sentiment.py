from twython import Twython, TwythonStreamer
import json
import sentiment_mod as s

data_path = 'C:/Users/rodri/OneDrive/Documentos/python/data/nlp_data/'

with open('C:/Users/rodri/OneDrive/Documentos/twitter_credentials.json') as f:
	twitter_auth_info = json.load(f)

CON_KEY = twitter_auth_info['con_key']
CON_SECRET = twitter_auth_info['con_secret']
ACC_TOKEN = twitter_auth_info['access_token']
ACC_SECRET = twitter_auth_info['access_secret']

tweets = []

class MyStreamer(TwythonStreamer):

	def on_success(self, data):

		try:

			tweet = data['text']#.encode('utf-8')
			#print(tweet)
			sentiment_value, confidence = s.sentiment(tweet)
			tweets.append(tweet)

			print(data['text'].encode('utf-8'), sentiment_value, confidence)

			if confidence*100 >= 80:
				output = open(data_path+'twitter-out.txt', 'a')
				output.write(sentiment_value)
				output.write('\n')
				output.close()
		except:
			return True
		#if len(tweets) >= 10:
			#self.disconnect()

	def on_error(self, status_code, data):
		print(status_code, data)
		self.disconnect()

stream = MyStreamer(CON_KEY, CON_SECRET, ACC_TOKEN, ACC_SECRET)
stream.statuses.filter(track=['happy'])
