import requests
import pandas as pd
import easygui
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
import re 
import string
from nltk.corpus import stopwords
from string import punctuation

# JJ = easygui.fileopenbox()
# df = pd.read_csv(JJ)
df = pd.read_excel('Input.xlsx')

print(df.head())

df['positive_score'] = ''                #a1
df['negative_score'] = ''                #a2
df['polarity_score'] = ''                #a3
df['subjectivity_score'] = ''            #a4
df['avg_sentence_length'] = ''           #a5
df['Percentage_of_Complex_words'] = ''   #a6
df['Fog_Index'] = ''                     #a7
df['complex_word_count'] = ''            #a8
df['word_count'] = ''                    #a9
df['avg_syllable_word_count'] = ''       #a10
df['pp_count'] = ''                      #a11
df['average_word_length'] = ''           #a12

print(df.head())

stopwords_dir = "StopWords"
sentment_dir = "MasterDictionary"

stop_words = set()
for files in os.listdir(stopwords_dir):
  with open(os.path.join(stopwords_dir,files),'r',encoding='ISO-8859-1') as f:
    stop_words.update(set(f.read().splitlines()))

# print(stop_words)
pos=set()
neg=set()

for files in os.listdir(sentment_dir):
  if files =='positive-words.txt':
    with open(os.path.join(sentment_dir,files),'r',encoding='ISO-8859-1') as f:
      pos.update(f.read().splitlines())
  else:
    with open(os.path.join(sentment_dir,files),'r',encoding='ISO-8859-1') as f:
      neg.update(f.read().splitlines())

# print(pos)
# print(neg)

start = 0
end = len(df)

for a in range(start , end):
    
    url_id = df.loc[a, 'URL_ID']
    url = df.loc[a, 'URL']
    page = requests.get(url)
    print(a,url_id,url,page)

    soup = BeautifulSoup(page.content, 'html.parser')
    # print(page.content)
    try:
        title = soup.find('h1').get_text() 
        print(title)
    except:
        print("can't get title of {}".format(url_id))
        continue
    article = ""
    for p in soup.find_all('p'):
        article += p.get_text()
        # print(article)

    text_file = title + '\n' + article
    
    # print(file_file)
    docs = []

    # text_file = re.sub(r'[^\w\s.]','',text_file)

    #tokenize the given text file
    text_split = text_file.split()
    text_split

    filtered_text = [word for word in text_split if word.lower() not in stop_words]

    docs.append(filtered_text)

    # 1	Sentimental Analysis
    # Sentimental analysis is the process of determining whether a piece of writing is positive, negative, or neutral. The below Algorithm is designed for use in Financial Texts. It consists of steps:
    # 1.1	Cleaning using Stop Words Lists
    # The Stop Words Lists (found in the folder StopWords) are used to clean the text so that Sentiment Analysis can be performed by excluding the words found in Stop Words List. 
    # 1.2	Creating a dictionary of Positive and Negative words
    # The Master Dictionary (found in the folder MasterDictionary) is used for creating a dictionary of Positive and Negative words. We add only those words in the dictionary if they are not found in the Stop Words Lists. 
    # 1.3	Extracting Derived variables.
    # We convert the text into a list of tokens using the nltk tokenize module and use these tokens to calculate the 4 variables described below:
    # Positive Score: This score is calculated by assigning the value of +1 for each word if found in the Positive Dictionary and then adding up all the values.
    # Negative Score: This score is calculated by assigning the value of -1 for each word if found in the Negative Dictionary and then adding up all the values. We multiply the score with -1 so that the score is a positive number.
    # Polarity Score: This is the score that determines if a given text is positive or negative in nature. It is calculated by using the formula: 
    # Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
    # Range is from -1 to +1
    # Subjectivity Score: This is the score that determines if a given text is objective or subjective. It is calculated by using the formula: 
    # Subjectivity Score = (Positive Score + Negative Score)/ ((Total Words after cleaning) + 0.000001)
    # Range is from 0 to +1

    positive_words = []
    Negative_words = []
    positive_score = []
    negative_score = []
    polarity_score = []
    subjectivity_score = []

    for i in range(len(docs)):
        positive_words.append([word for word in docs[i] if word.lower() in pos])
        positive_score.append(len(positive_words[i]))
        Negative_words.append([word for word in docs[i] if word.lower() in neg])
        negative_score.append(len(Negative_words[i]))
        polarity_score.append((positive_score[i] - negative_score[i]) / ((positive_score[i] + negative_score[i]) + 0.000001))
        subjectivity_score.append((positive_score[i] + negative_score[i]) / ((len(docs[i])) + 0.000001))

    print( 
    'positive_score:',positive_score ,
    'negative_score:' ,negative_score ,
    'polarity_score:',polarity_score,
    'subjectivity_score:',subjectivity_score )

    a1 = positive_score[i]
    a2 = negative_score[i]
    a3 = polarity_score[i]
    a4 = subjectivity_score[i] 

    # 2	Analysis of Readability
    # Analysis of Readability is calculated using the Gunning Fox index formula described below.
    # Average Sentence Length = the number of words / the number of sentences
    # Percentage of Complex words = the number of complex words / the number of words 
    # Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
    # 3	Average Number of Words Per Sentence
    # The formula for calculating is:
    # Average Number of Words Per Sentence = the total number of words / the total number of sentences



    avg_sentence_length = []
    avg_syllable_word_count =[]
    Fog_Index = []

    sentences = text_file.split('.')

    total_words = len(text_file.split())

    total_sentences = len(sentences)

    average_sentence_length = total_words / total_sentences

    print("Average Sentence Length:", average_sentence_length ,"Total_words:",total_words ,"total_sentences:",total_sentences)
    a5 = average_sentence_length





    # 4	Complex Word Count
    # Complex words are words in the text that contain more than two syllables.

    complex_word_count = 0

    for word in text_split:
        vowel_count = 0
        for char in word.lower():
            if char in 'aeiou':
                vowel_count += 1
        if vowel_count > 2:
            complex_word_count += 1

    print("Complex Word Count:", complex_word_count)

    a8 = complex_word_count

    Percentage_of_Complex_words  = complex_word_count/total_words
    percentage_of_complex_words = float(Percentage_of_Complex_words)
    print("Percentage_of_Complex_words:", Percentage_of_Complex_words)

    a6 = Percentage_of_Complex_words

    Fog_Index = 0.4*(average_sentence_length + Percentage_of_Complex_words)
    print("Fog_Index:", Fog_Index)  

    a7 = Fog_Index




    # 5	Word Count
    # We count the total cleaned words present in the text by 
    # 1.	removing the stop words (using stopwords class of nltk package).
    # 2.	removing any punctuations like ? ! , . from the word before counting.

    stopword = set(stopwords.words('english'))
    word_tokens = word_tokenize(text_file)

    filtered_sentence = [w for w in word_tokens if not w.lower() in stopword]

    filtered_sentence = []
    
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    word_count = len(filtered_sentence)
    
    print('word_count:', word_count)

    a9 = word_count




    # 6	Syllable Count Per Word
    # We count the number of Syllables in each word of the text by counting the vowels present in each word. We also handle some exceptions like words ending with "es","ed" by not counting them as a syllable.

    syllable_count = 0
    syllable_words =[]
    for word in text_split:
        if word.endswith('es'):
            word = word[:-2]
        elif word.endswith('ed'):
            word = word[:-2]
        vowels = 'aeiou'
        syllable_count_word = sum( 1 for letter in word if letter.lower() in vowels)
        if syllable_count_word >= 1:
            syllable_words.append(word)
            syllable_count += syllable_count_word
            
    avg_syllable_word_count = syllable_count / len(syllable_words)
    print('avg_syllable_word_count:', avg_syllable_word_count)

    a10 = avg_syllable_word_count


    # 7	Personal Pronouns
    # To calculate Personal Pronouns mentioned in the text, we use regex to find the counts of the words - “I,” “we,” “my,” “ours,” and “us”. Special care is taken so that the country name US is not included in the list.

    personal_pronouns = ['I','we', 'my','ours', 'us']

    pronoun_counts = {pronoun: 0 for pronoun in personal_pronouns}
    print(pronoun_counts)
    
    for pronoun in personal_pronouns:
        pattern = rf'\b{pronoun}\b'  
        count = len(re.findall(pattern,text_file, re.IGNORECASE))
        pronoun_counts[pronoun] += count
        
    for pronoun, count in pronoun_counts.items():
        print(f"'{pronoun}' Count:", count)
        
    pp_counts = sum(pronoun_counts.values())
    print("pp_counts:",pp_counts)

    a11 = pp_counts




    # 8	Average Word Length
    # Average Word Length is calculated by the formula:
    # Sum of the total number of characters in each word/Total number of words

    average_word_length = len(text_file) / total_words 
    print("average_word_length:",average_word_length)


    a12 = average_word_length
        


    df.loc[a, 'positive_score'] = a1                #a1
    df.loc[a, 'negative_score'] = a2                #a2
    df.loc[a, 'polarity_score'] = a3                #a3
    df.loc[a, 'subjectivity_score'] = a4            #a4
    df.loc[a, 'avg_sentence_length'] = a5           #a5
    df.loc[a, 'Percentage_of_Complex_words'] = a6   #a6
    df.loc[a, 'Fog_Index'] = a7                     #a7
    df.loc[a, 'complex_word_count'] = a8            #a8
    df.loc[a, 'word_count'] = a9                    #a9
    df.loc[a, 'avg_syllable_word_count'] = a10      #a10
    df.loc[a, 'pp_count'] = a11                     #a11
    df.loc[a, 'average_word_length'] = a12           #a12
    # print(df.head())

    print(a)
    # j = i

print(df.head())

df.to_csv('output_data.csv', index=False)

print("Api-Hit Done. Output file created.")