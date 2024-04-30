# Text PreProcessing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("C:/Users/PC/Desktop/amazon_reviews.csv", sep=",")
df.head()
df.info()
df['reviewText']


########################
# Normalizing Case Folding
########################
# string ifade yer aldığı için bütün satırlar belirli bir standarta koyuldu ve büyük-küçük harf dönüşümü gerçekleşti.
df['reviewText'] = df['reviewText'].str.lower()


#########################
# Punctuations
#########################

# regular expression
# Metinde herhangi bir noktalama işareti görüldüğünde boşluk ile değiştir.
df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')


######################
# Numbers
#####################

# ilgili text içerisindeki sayıları yakala sonra boşluk ile değiştir.
df['reviewText'] = df['reviewText'].str.replace('\d', '')


######################
# Stopwords
#####################

# metinlerde herhangi bir anlamı olmayan-barınan yaygın kullanılan kelimeleri at.

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
sw = stopwords.words('english')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

####################
# Rarewords
####################

# Nadir geçen kelimelerin örüntü oluşturamayacağını varsayarak onları çıkartma işlemi.
# bir kelime ne kadar sıklıkta geçiyor.

import pandas as pd
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
drops = temp_df[temp_df <= 1]
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))



#####################
# Tokenization
#####################
# cümleleri parçalamak birimleştirmek.
nltk.download('punkt')
from textblob import TextBlob
df['reviewText'].apply(lambda x: TextBlob(x).words).head()



######################
# Lemmatization
#####################

# kelimeleri köklerine indirgemek
# (stemming) ayrıca  buda bir köklerine ayırma işlemidir.
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
df['reviewText'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))




###########################
# 2. Text Visualization
###########################
# Metinler numerik formata değil nasıl görselleştirebiliriz?
# Terim frekanslarının hesaplanması

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)


###############################
# Bar Plot ( Sütun Grafik )
##############################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()


###########################
# Word Cloud
##########################
# İlgili metindeki kelimelerin geçme frekanslarına göre bir bulut şeklinde görsel oluşturulması işidir.
# Öncelikle (reviewText) değişkeninde gezerek oradaki tüm satırları tek bir cümleymiş gibi bir araya getirmek gerekir

text = " ".join(i for i in df.reviewText)

from wordcloud import WordCloud
# Text dosyasından worldcloud oluşturuldu.
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# Frekans açısından en yüksek kelimeler daha büyük görünür.
# Kelime bulutundaki kelimelerin büyüklüğü kelimelerin frekanslarına göre şekilleniyor.
plt.show()


# 1.Daha açık renkli bir grafik oluşturalım
# 2.Font boyutu 50 olsun
# 3.Bu grafiğe yansıtılan maksimum kelime sayısı 100 olsun

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# İlgili wordcloud cıktısını kaydetmek için.
wordcloud.to_file("wordcloud.png")




##########################
# Şablonlara göre Word Cloud
##########################
# Kelime bulutu istenilen bir şablona göre görselleştirilebilir.

from PIL import Image
tr_mask = np.array(Image.open("tr.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick")
wc.generate(text)
plt.figure(figsize=[10,10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
