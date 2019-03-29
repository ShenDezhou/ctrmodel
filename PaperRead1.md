Prediction of Movie Success using Sentiment Analysis of Tweets
1.	数据：使用斯坦福SNAP的2009Twitter数据集，以及使用Tweepy接口并且使用电影相关关键词来采集Twitter数据。
Message、who、content、timestamp
2.	处理：60G的2009数据集、1G的新采集数据，使用HPCC集群，利用PBS命令访问。包含电影的tweets、过滤2周前4周后微博，处理30部电影。使用https://github.com/shuyo/ldig.过滤非英语微博，过滤广告等噪声，去重。
3.	使用Lingpipe进行情感分析，基于8-gram的语言模型，首先标注数据，将数据分成positive/negative/neutral/irrelevant四部分。使用2009年数据中的24部电影的200个微博作为训练集，6部电影作为测试集。2012年的8部电影作为另一测试集。
4.	使用数据训练lingpipe情感分析器，又利用nltk训练bayes分类器。
5.	预测：定义电影收入为三类hit, flop, and average，Profit-budget:[ (>=20M),<0, 0<= <=20M]
6.	定义了PT-NT ratio，PT ratio is the percent of positive tweets, and NT ratio is the percent of negative tweets.
7.	定义profit ratio = (revenue-budget)/budget.
8.	定义PT-NT ratio在hit, flop, and average（>=5, 1.5< <5, <1.5）.
9.	评价：Lingpipe情感分析准确率64.4%， PT-NT ratio预测profit ratio，正确率5/6。Miami Connection无法判断。
