

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

# Establish the database connection
engine = create_engine('mysql+pymysql://root:sunny106133@localhost/project')

# Fetch data into Pandas DataFrames
products_df = pd.read_sql("SELECT * FROM products", con=engine)
users_df = pd.read_sql("SELECT * FROM users", con=engine)
reviews_df = pd.read_sql("SELECT * FROM product_reviews", con=engine)

missing_users = set(reviews_df['user_id'].unique()) - set(users_df['user_id'].unique())
missing_products = set(reviews_df['product_id'].unique()) - set(products_df['product_id'].unique())

merged_1 = pd.merge(reviews_df, products_df, how='outer', on='product_id')
print("Shape after first merge (reviews + products):", merged_1.shape)

merged_2 = pd.merge(merged_1, users_df, how='outer', on='user_id')
print("Shape after second merge (reviews + products + users):", merged_2.shape)


# Merge DataFrames
merged_df = pd.merge(reviews_df, products_df, how='outer', on='product_id')
merged_df = pd.merge(merged_df, users_df, how='outer', on='user_id')
# print(merged_df.shape)

merged_df.fillna(value=np.nan, inplace=True)


# Now merged_df contains your merged data
df = merged_df
mask_all_nan_except_credentials = df.drop(columns=['user_id', 'user_name', 'password']).isna().all(axis=1)

# Create a mask for rows where rating and rating_count are NOT NaN.
mask_valid_ratings = ~(df['rating'].isna() | df['rating_count'].isna())

# Combine both conditions using OR (|)
final_mask = mask_all_nan_except_credentials | mask_valid_ratings

# Filter the dataframe using the combined mask
df = df[final_mask]

df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%','').astype(float)/100
df['rating'] = df['rating'].astype(str)
count = df['rating'].str.contains('\|').sum()
print(f"Total de linhas com '|' na coluna 'rating': {count}")
df = df[df['rating'].apply(lambda x: '|' not in str(x))]
count = df['rating'].str.contains('\|').sum()
print(f"Total de linhas com '|' na coluna 'rating': {count}")

df['rating'] = df['rating'].astype(str).str.replace(',', '').astype(float)
df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)

le = LabelEncoder()
df['user_id_encoded'] = le.fit_transform(df['user_id'])

# Calculate the mean rating across all products
C = df['rating'].mean()

# Calculate the 90th percentile of the number of ratings
m = df['rating_count'].quantile(0.9)

# Filter out movies that have a rating count less than m
qualified_products = df[df['rating_count'] >= m]

# Compute the weighted rating for each qualified product
def weighted_rating(x, m=m, C=C):
    v = x['rating_count']
    R = x['rating']
    return (v / (v + m) * R) + (m / (v + m) * C)

# Apply the function to the DataFrame
qualified_products['weighted_rating'] = qualified_products.apply(weighted_rating, axis=1)

# Sort products based on score
qualified_products = qualified_products.sort_values('weighted_rating', ascending=False)

# Print the top recommendations
print(qualified_products[['product_name', 'rating', 'rating_count', 'weighted_rating']])

# DATA SPLIT

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def content_based_filtering(df):
    df['combined_text'] = df['about_product'].fillna('') + ' ' + df['review_title'].fillna('') + ' ' + df['review_content'].fillna('')
    index_mapping = {index: i for i, index in enumerate(df.index)}
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    user_profiles = {}

    # def create_tfidf_matrix(df_inner):
    #     tfidf = TfidfVectorizer(stop_words='english')
    #     df_inner['about_product'] = df_inner['about_product'].fillna('')
    #     tfidf_matrix = tfidf.fit_transform(df_inner['about_product'])
    #     return tfidf_matrix, tfidf

    # tfidf_matrix, tfidf = create_tfidf_matrix(df)

    # user_profiles = {}

    for user_id_encoded in df['user_id_encoded'].unique():
        user_data = df[df['user_id_encoded'] == user_id_encoded]
        tfidf_indices = [index_mapping[idx] for idx in user_data.index.tolist()]

        user_vector = np.sum(tfidf_matrix[tfidf_indices], axis=0)
        user_vector = np.asarray(user_vector).reshape(1, -1)

        if np.linalg.norm(user_vector) != 0:
            user_vector_norm = user_vector / np.linalg.norm(user_vector)
            user_profiles[user_id_encoded] = user_vector_norm
        else:
            print(f"Zero norm detected for user_id_encoded: {user_id_encoded}")
    

    def recommend_products_with_profiles(user_id_encoded):
        user_vector = user_profiles.get(user_id_encoded, None)
        if user_vector is None:
            return qualified_products

        cosine_sim_user = cosine_similarity(user_vector, tfidf_matrix)
    
        # Getting products already interacted with by the user
        interacted_products = df.loc[df['user_id_encoded'] == user_id_encoded]['product_id'].tolist()

        # Sorting the similarity scores
        similarity_scores = sorted(list(enumerate(cosine_sim_user[0])), key=lambda x: x[1], reverse=True)

        all_top_products = []
        for idx, score in similarity_scores:
            if len(all_top_products) == 5:  # break if we already have 5 recommendations
                break
            product_id = df.iloc[idx]['product_id']
            if product_id not in interacted_products and product_id not in [prod['product_id'] for prod in all_top_products]:
                all_top_products.append(df.iloc[idx])

        recommendation_response = []
        for product in all_top_products:
            model_response = {}
            model_response['product_id'] = product['product_id']
            model_response['product_name'] = product['product_name']
            model_response['img_link'] = product['img_link']
            model_response['actual_price'] = product['actual_price']
            model_response['discounted_price'] = product['discounted_price']
            model_response['discount_percentage'] = product['discount_percentage']
            recommendation_response.append(model_response)

        valid_scores = [score for idx, score in similarity_scores if df.iloc[idx]['product_id'] in [prod['product_id'] for prod in all_top_products]][:5]

        results_df = pd.DataFrame({
            'Id_Encoded': [user_id_encoded] * len(recommendation_response),
            'recommended_product': recommendation_response,
            'score_recommendation': valid_scores
        })

        return results_df
    return recommend_products_with_profiles, tfidf_matrix

recommender_function, tfidf_matrix_out = content_based_filtering(df)
recommendations = recommender_function(3)
print(recommendations)
# OG
# def prerequisites_collaborative():
#     x = df.groupby('user_id_encoded').count()['rating'] > 2
#     users_rated = x[x].index
#     filtered_df = df[df['user_id_encoded'].isin(users_rated)]

#     y = filtered_df.groupby('product_name').count()['rating'] > 0
#     # filtered_out_users = set(df['user_id_encoded']) - set(filtered_df['user_id_encoded'])
#     # print("Users filtered out:", filtered_out_users)
#     high_rated_products = y[y].index
#     final_rating = filtered_df[filtered_df['product_name'].isin(high_rated_products)]

#     # Create a user-item matrix
#     pt = final_rating.pivot_table(index='user_id_encoded', columns='product_name', values='rating')
#     print(3 in pt.index)

#     pt.fillna(0, inplace=True)
#     return pt
def prerequisites_collaborative():
    # Filter users who've rated at least 1 product
    x = df.groupby('user_id_encoded').count()['rating'] > 1
    users_rated = x[x].index
    filtered_df = df[df['user_id_encoded'].isin(users_rated)]

    # Consider all products that have been rated at least once
    y = filtered_df.groupby('product_name').count()['rating'] > 1
    high_rated_products = y[y].index
    final_rating = filtered_df[filtered_df['product_name'].isin(high_rated_products)]

    # Create a user-item matrix
    pt = final_rating.pivot_table(index='user_id_encoded', columns='product_name', values='rating')
    print(3 in pt.index)

    pt.fillna(0, inplace=True)
    return pt


pt = prerequisites_collaborative()


# OGG
from sklearn.metrics.pairwise import cosine_similarity
similarity_score = cosine_similarity(pt)
# OG COLLAB
# def get_recommendations(user_id_encoded):
#     """Return a list of recommended products for a given user."""
#     try:
#         index = np.where(pt.index == user_id_encoded)[0][0]
#         # print(f"Index for user_id_encoded: {user_id_encoded}: {index}")

#         # Find similar users
#         similar_users = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
#         # print(f"Similar users: {similar_users}")

#         # Get the items that these users have interacted with
#         recommended_products = []
#         for i in similar_users:
#             user_id = pt.index[i[0]]
#             rated_products = pt.columns[(pt.loc[user_id] > 0)].tolist()
#             recommended_products.extend(rated_products)

#         # Remove duplicates and return
#         recommended_products = list(set(recommended_products))
#         # print(f"Recommended products: {recommended_products}")
#     except IndexError:
#         print(f"Error: user_id_encoded {user_id_encoded} not found in index.")
#         recommended_products = qualified_products[['product_name', 'rating', 'rating_count', 'weighted_rating']]

#     return recommended_products

def get_recommendations(user_id_encoded):
    """Return a list of recommended products for a given user."""
    if user_id_encoded not in pt.index:
        print(f"Error: user_id_encoded {user_id_encoded} not found in index.")
        return qualified_products[['product_name', 'rating', 'rating_count', 'weighted_rating']]

    # Find similar users
    index = pt.index.get_loc(user_id_encoded)
    similar_users = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]

    # Get the items that these users have interacted with
    recommended_products = {}
    for i in similar_users:
        user_id = pt.index[i[0]]
        rated_products = pt.columns[(pt.loc[user_id] > 0)].tolist()
        for product in rated_products:
            if product not in recommended_products:
                recommended_products[product] = i[1]
            else:
                recommended_products[product] += i[1]

    # Filter and format the top products
    top_products = sorted(recommended_products.items(), key=lambda x: x[1], reverse=True)[:5]

    results_df = pd.DataFrame({
        'Id_Encoded': [user_id_encoded] * len(top_products),
        'recommended_product': [{'product_name': prod[0]} for prod in top_products],
        'score_recommendation': [prod[1] for prod in top_products]
    })

    return results_df



recommendations_collab = get_recommendations(3)
print("collab : " , recommendations_collab)

# OG HYBRID
# def hybrid_recommendation(user_id_encoded, alpha=0.5):
#     """
#     alpha: Determines the weightage to be given to content-based vs collaborative filtering.
#     An alpha of 0.5 means both will have equal weightage.
#     """
#     # 1. Get recommendations from Content-based Recommendation System
#     recommender_function, tfidf_matrix_out = content_based_filtering(df)
#     content_based_recommendations = recommender_function(user_id_encoded)

#     # 2. Get recommendations from Collaborative Filtering System (user-based CF)
#     collaborative_recommendations = get_recommendations(user_id_encoded)[:5]

#     # Truncate lists for consistency
#     min_length = min(len(content_based_recommendations), len(collaborative_recommendations))
#     content_based_list = content_based_recommendations['recommended_product'].tolist()[:min_length]
#     collaborative_list = collaborative_recommendations[:min_length]
    
#     # 3. Combine the scores 
#     content_based_recommendations['score_recommendation'] *= alpha
#     collaborative_scores = [(1 - idx * 0.1) * (1 - alpha) for idx in range(min_length)]
    
#     try:
#         combined_products = content_based_list + collaborative_list
#         combined_scores = content_based_recommendations['score_recommendation'].tolist()[:min_length] + collaborative_scores
        
#         # Deduplicate
#         seen = set()
#         final_recommendations = []
#         final_scores = []
#         print("seen",seen)
#         for prod, score in zip(combined_products, combined_scores):
#             if isinstance(prod, dict):
#                 product_id = prod['product_id']
#             else:  # If prod is a string
#                 product_id = prod

#             if product_id not in seen:
#                 final_recommendations.append(prod)
#                 final_scores.append(score)
#                 seen.add(product_id)
                
#     except Exception as e:
#         raise ValueError(f"Error during combination: {e}, content_based_list length: {len(content_based_list)}, collaborative_list length: {len(collaborative_list)}")

#     return pd.DataFrame({'Product': final_recommendations, 'Score': final_scores})

# working in a weird way
# def hybrid_recommendation(user_id_encoded, alpha=0.5):
#     """
#     Combines recommendations from content-based and collaborative recommendation systems.
#     alpha: Determines the weightage to be given to content-based vs collaborative filtering.
#     An alpha of 0.5 means both will have equal weightage.
#     """

#     # Get recommendations from Content-based Recommendation System
#     recommender_function, tfidf_matrix_out = content_based_filtering(df)
#     content_based_recommendations = recommender_function(user_id_encoded)

#     # Get recommendations from Collaborative Filtering System
#     collaborative_recommendations = get_recommendations(user_id_encoded)[:5]

#     # Prepare dictionaries for easy score lookup
#     content_based_scores = dict(zip(content_based_recommendations['recommended_product'].apply(lambda x: x['product_id']),
#                                    content_based_recommendations['score_recommendation']))
    
#     collaborative_scores = dict(zip(collaborative_recommendations, [(1 - idx * 0.1) for idx in range(len(collaborative_recommendations))]))
    
#     # Combine product ids from both methods
#     all_product_ids = set(content_based_scores.keys()) | set(collaborative_scores.keys())
    
#     final_recommendations = []
#     for product_id in all_product_ids:
#         content_score = content_based_scores.get(product_id, 0)
#         collab_score = collaborative_scores.get(product_id, 0)
        
#         combined_score = alpha * content_score + (1 - alpha) * collab_score
#         final_recommendations.append({'Product': product_id, 'Score': combined_score})
    
#     # Sort the final recommendations by score in descending order and convert to DataFrame
#     final_recommendations = sorted(final_recommendations, key=lambda x: x['Score'], reverse=True)
#     return pd.DataFrame(final_recommendations)


# idk what was this 
def hybrid_recommendation(user_id_encoded, alpha=0.2, N=50):
    """ Get hybrid recommendations. """

    # 1. Get top N recommendations from both systems
    recommender_function, _ = content_based_filtering(df)
    content_based_recommendations = recommender_function(user_id_encoded)[:N]
    
    collaborative_recommendations = get_recommendations(user_id_encoded)[:N]

    # Normalize scores from both systems to [0,1] using Min-Max scaling
    max_content_score = content_based_recommendations['score_recommendation'].max()
    max_collab_score = collaborative_recommendations['score_recommendation'].max()

    content_based_recommendations['normalized_score'] = content_based_recommendations['score_recommendation'] / max_content_score
    collaborative_recommendations['normalized_score'] = collaborative_recommendations['score_recommendation'] / max_collab_score

    # 2. Normalize and combine scores for common products
    combined_scores = {}
    
    for _, row in content_based_recommendations.iterrows():
        prod_id = row['recommended_product']['product_id']
        content_score = row['normalized_score']

        collab_row = collaborative_recommendations[collaborative_recommendations['recommended_product'].apply(lambda x: x['product_name']) == prod_id]
        if not collab_row.empty:
            collab_score = collab_row['normalized_score'].values[0]
        else:
            collab_score = 0

        combined_score = alpha * content_score + (1 - alpha) * collab_score
        combined_scores[prod_id] = combined_score

    # 3. Sort and get top 5
    sorted_products = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    final_recommendations = [{'product_id': prod[0]} for prod in sorted_products]
    final_scores = [prod[1] for prod in sorted_products]

    return pd.DataFrame({'Product': final_recommendations, 'Score': final_scores})


recommendations = hybrid_recommendation(3)
print("Hybrid :" ,recommendations)








