
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from gensim import corpora
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')
reviews = pd.read_excel(r"C:\reviews_project\turbo_final.xlsx")

#check
#print (reviews.head())

#data prep
reviews.dropna(subset=['content_translated', 'rating'], inplace=True)
reviews['content_translated'] = reviews['content_translated'].str.lower()
reviews = reviews[reviews['content_translated'].str.split().str.len() >= 3] 
lemmatizer = WordNetLemmatizer()
# Preprocessing function
def preprocess(text):
    # Define custom stop words
    custom_stop_words = ['like', 'nice', 'good', 'bad', 'app', 'dick', 'vpn','internet','stars','cc','thanks','1111']
    
    # Combine NLTK stop words with custom stop words
    all_stop_words = set(stopwords.words('english')).union(set(custom_stop_words))
    
    # Tokenize and remove stop words
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in all_stop_words]
    
    return tokens

# Apply preprocessing to the review_text column
reviews['tokens'] = reviews['content_translated'].apply(preprocess)

def categorize_rating(rating):
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'
#run func
reviews['sentiment_category'] = reviews['rating'].apply(categorize_rating)

# Filter positive and negative reviews
positive_reviews = reviews[reviews['sentiment_category'] == 'positive']
negative_reviews = reviews[reviews['sentiment_category'] == 'negative']

# Apply preprocessing to the review_text column of both DataFrames
positive_reviews['tokens'] = positive_reviews['content_translated'].apply(preprocess)
negative_reviews['tokens'] = negative_reviews['content_translated'].apply(preprocess)

# Create a dictionary representation of the documents.
# Create dictionary and corpus for positive reviews
positive_dictionary = corpora.Dictionary(positive_reviews['tokens'])
positive_corpus = [positive_dictionary.doc2bow(tokens) for tokens in positive_reviews['tokens']]

# Create dictionary and corpus for negative reviews
negative_dictionary = corpora.Dictionary(negative_reviews['tokens'])
negative_corpus = [negative_dictionary.doc2bow(tokens) for tokens in negative_reviews['tokens']]

from gensim.models import LdaModel

# Set parameters for LDA
# Set parameters for LDA model
num_topics = 3  # Adjust based on your needs

# Build LDA model with alpha and beta tuning
lda_positive_model = LdaModel(
    corpus=positive_corpus,
    num_topics=num_topics,
    id2word=positive_dictionary,
    passes=15,
    alpha='auto',  # Automatically learn the alpha parameter
    eta='auto'     # Automatically learn the beta (eta) parameter
)

lda_negative_model = LdaModel(
    corpus=negative_corpus,
    num_topics=num_topics,
    id2word=negative_dictionary,
    passes=15,
    alpha='auto',  # Automatically learn the alpha parameter
    eta='auto'     # Automatically learn the beta (eta) parameter
)

print("Topics in Positive Reviews:")
for idx, topic in lda_positive_model.print_topics(-1):
    print(f"Topic {idx + 1}: {topic}")

print("\nTopics in Negative Reviews:")
for idx, topic in lda_negative_model.print_topics(-1):
    print(f"Topic {idx + 1}: {topic}")

from gensim.models import CoherenceModel

# Calculate coherence score
coherence_model_lda = CoherenceModel(model=lda_positive_model, texts=reviews['tokens'], dictionary=positive_dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f'Coherence Score: {coherence_lda}')

coherence_model_lda = CoherenceModel(model=lda_negative_model, texts=reviews['tokens'], dictionary=negative_dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f'Coherence Score: {coherence_lda}')

# Assign topics to each document
def get_topic_distribution(lda_model, corpus):
    topic_distribution = []
    for doc in corpus:
        topic_distribution.append(lda_model.get_document_topics(doc))
    return topic_distribution

# Get topic distribution for positive reviews
positive_reviews['topic_distribution'] = get_topic_distribution(lda_positive_model, positive_corpus)

# Assign the dominant topic to each positive review
positive_reviews['dominant_topic'] = positive_reviews['topic_distribution'].apply(
    lambda x: sorted(x, key=lambda y: y[1], reverse=True)[0][0]
)

# Get topic distribution for negative reviews
negative_reviews['topic_distribution'] = get_topic_distribution(lda_negative_model, negative_corpus)

# Assign the dominant topic to each negative review
negative_reviews['dominant_topic'] = negative_reviews['topic_distribution'].apply(
    lambda x: sorted(x, key=lambda y: y[1], reverse=True)[0][0]
)

# Combine results back into the original DataFrame
reviews['dominant_topic'] = None

for index, row in positive_reviews.iterrows():
    reviews.loc[index, 'dominant_topic'] = row['dominant_topic']

for index, row in negative_reviews.iterrows():
    reviews.loc[index, 'dominant_topic'] = row['dominant_topic']


print(reviews[['content_translated', 'rating', 'sentiment_category', 'dominant_topic']].head())


# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Function to plot topic distribution for positive and negative reviews
def plot_topic_distribution(reviews_df, sentiment):
    # Filter reviews based on sentiment
    filtered_reviews = reviews_df[reviews_df['sentiment_category'] == sentiment]
    
    # Count the occurrences of each dominant topic
    topic_counts = filtered_reviews['dominant_topic'].value_counts().sort_index()
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='viridis')
    plt.title(f'Distribution of Dominant Topics in {sentiment.capitalize()} Reviews')
    plt.xlabel('Topic')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)
    plt.show()

# Plot for positive reviews
plot_topic_distribution(reviews, 'positive')

# Plot for negative reviews
plot_topic_distribution(reviews, 'negative')

def extract_influential_words(lda_model, num_words=10):
    influential_words = {}
    for idx, topic in lda_model.print_topics(num_words=num_words):
        # Split the topic string into individual word-weight pairs
        word_weight_pairs = topic.split('+')
        
        # Create a dictionary of word and its weight
        word_weights = {}
        for pair in word_weight_pairs:
            # Split each pair by '*'
            parts = pair.split('*')
            if len(parts) == 2:  # Ensure there are exactly two parts
                weight = float(parts[0])
                word = parts[1].strip().replace('"', '')
                word_weights[word] = weight
        
        influential_words[idx] = word_weights
    return influential_words

def create_word_clouds(influential_words, title_prefix):
    for topic_idx, words in influential_words.items():
        # Create a word cloud from the influential words
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
        
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{title_prefix} Topic {topic_idx + 1}')
        plt.show()

# Extract influential words for positive and negative models
positive_influential_words = extract_influential_words(lda_positive_model)
negative_influential_words = extract_influential_words(lda_negative_model)

# Create word clouds for positive topics
create_word_clouds(positive_influential_words, "Positive Review")

# Create word clouds for negative topics
create_word_clouds(negative_influential_words, "Negative Review")

# Count the number of reviews for each rating
rating_counts = reviews['rating'].value_counts().sort_index()

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis')
plt.title('Number of Reviews by Rating')
plt.xlabel('Rating (1-5 Stars)')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.grid(axis='y')

# Show the plot
plt.show()
