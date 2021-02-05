# botyspoty
This project detects bots in internet on the basis of tweets.

1.Dataset1(Trolls).ipynb uses 3 million Russian troll tweets dataset https://www.kaggle.com/fivethirtyeight/russian-troll-tweets and gives bot1.csv as output. The objective of this file is to combine all 9 files and extract only English language tweets.

2.Dataset2(Humans).ipynb uses humansraw.txt. O/P is human.csv. This code converts txt file --> csv file.

3.FinalDataset.ipynb uses the o/p from 1 and 2 and gives bot_or_human.csv as o/p. The file labels the dataset as bot=1 and human=0.

4.Bot_in_Net1.ipynb uses bot_or_human.csv and performs a text processing. o/p is clean.csv. 
  Text processing includes:converting to lower case it,removing everything apart from a-zA-Z,tokenizing it,removing stopwords and finally lemmetizing it.

5.Model.ipynb is where magic happens. It use clean.csv. Words are converted into feature vectors using count vector, TFIDF and word embeddings then these feature vectors are trained using LR,NB,SVM,RF and LSTM.

6.Modeldevpickel.py and Model for Pickel.ipynb pickels the model for deployment. o/p model.pkl and cv.pkl.

7.app.py is used for Deployment purpose.

8.templates folder contains html files and static folder contains css files.

9.Final Testing.ipynb used for testing model.pkl and cv.pkl files.

10.Procfile is used for determining where the app start from.


Deployment Link: https://botyspoty.herokuapp.com/

