import sys

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Prepare data
def get_booking_list():
    booking_data = requests.get("http://localhost:8080/api/booking/getBookingList?filter_user=&page=&size=").json()[
        "data"]
    booking_df = pd.DataFrame(booking_data)
    return booking_df


# Create TF-IDF vectors for the lessons
def generate_TF_IDF_vectors(booking_df):
    tfidf = TfidfVectorizer(stop_words='english')
    lesson_vectors = tfidf.fit_transform(booking_df['lessonName'])
    return lesson_vectors


# Compute the pairwise cosine similarity between the lesson vectors
def compute_cosine_similarity(lesson_vectors):
    similarity_matrix = cosine_similarity(lesson_vectors)
    return similarity_matrix


# Generate recommendations for a user
# target_user = 'admin2'
def get_recommended_lesson(target_user, booking_df, similarity_matrix):
    user_lessons = booking_df[booking_df['bookingUser'] == target_user]['lessonName'].unique()
    recommended_lessons = []
    for lesson in user_lessons:
        lesson_index = booking_df[booking_df['lessonName'] == lesson].index[0]
        similar_lesson_indices = similarity_matrix[lesson_index].argsort()[::-1][1:]
        for idx in similar_lesson_indices:
            similar_lesson = booking_df.loc[idx, 'lessonName']
            if similar_lesson not in user_lessons and similar_lesson not in recommended_lessons:
                recommended_lessons.append(similar_lesson)
                if len(recommended_lessons) == 3:
                    break
        if len(recommended_lessons) == 3:
            break
    return recommended_lessons

def single_quote_to_double_quote(string):
    return string.replace("'", '"')

def main(user_name):
    booking_df = get_booking_list()
    lesson_vectors = generate_TF_IDF_vectors(booking_df)
    similarity_matrix = compute_cosine_similarity(lesson_vectors)
    # target_user = 'admin2'
    recommended_lessons = get_recommended_lesson(user_name, booking_df, similarity_matrix)
    return single_quote_to_double_quote(str(recommended_lessons))

if __name__ == '__main__':
    user_name = sys.argv[1]
    print(main(user_name))