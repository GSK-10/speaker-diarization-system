from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print("Started")

def categorize_sentiment(text):

    # Create a SentimentIntensityAnalyzer object / instance
    analyzer = SentimentIntensityAnalyzer()
    
    # Get the sentiment scores
    # This method returns a dictionary containing positive, negative, neutral, and compound scores.
    sentiment_scores = analyzer.polarity_scores(text)
    print("Score:", sentiment_scores)
    
    # Categorize sentiment based on scores
    sentiment_category = 'Neutral' # default sentiment category
        
    if sentiment_scores['compound'] >= 0.5:
        sentiment_category = 'Happy'
    elif sentiment_scores['compound'] <= -0.5:
        sentiment_category = 'Angry'
    elif sentiment_scores['compound'] <= -0.25:
        sentiment_category = 'Fear'
    elif sentiment_scores['compound'] >= 0.25:
        sentiment_category = 'Surprise'
    
    return sentiment_category, sentiment_scores


if __name__ == "__main__":
    # Example text for analysis
    text = "Someone might kill me. I am not understanding what to do. Someone help me."

    # Perform sentiment analysis and categorization
    sentiment_category = categorize_sentiment(text)

    # Print the result
    print("Sentiment Category:", sentiment_category)

