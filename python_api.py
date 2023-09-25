import sys
from flask import Flask , request, jsonify
import subprocess
from flask_cors import CORS
import pandas as pd

sys.path.insert(0, 'C:/Users/swapn/Desktop/final rs/')

app = Flask(__name__)

CORS(app)
@app.route("/recommend", methods=['GET'])
def recommend():
    userid = request.args.get('userid')

    # Ensure the Python script is in your system path
    if userid:
        
        # Import the function
        from collaborative import popular_products
        from content_based_rs2 import recommend_products_with_profiles 
        from hybrid import get_recommendations, hybrid_recommendation_v2 , userIdEncoder
        
        # Get recommendations
        recommendations = popular_products()
        userIdEncoded = userIdEncoder(userid)
        # print(userIdEncoded)
        # recommendations2 = recommend_products_with_profiles(userIdEncoded)
        recommendations2 = recommend_products_with_profiles(userIdEncoded)
        recommendations2 = pd.DataFrame(recommendations2)

        # Convert DataFrame to a dictionary
        recommendations_dict = recommendations.to_dict(orient='records')
        recommendations_dict_2 = recommendations2.to_dict(orient='records')
        # Return recommendations as a JSON
        return jsonify({'Popular_products': recommendations_dict, 'content_based_products': recommendations_dict_2})
    else:
        from collaborative import popular_products
        from content_based_rs2 import recommend_products_with_profiles , userIdEncoder
        
        recommendations = popular_products()
        recommendations_dict = recommendations.to_dict(orient='records')
        recommendations2 = popular_products()
        recommendations2 = pd.DataFrame(recommendations2)
        recommendations_dict_2 = recommendations2.to_dict(orient='records')
        return jsonify({'Popular_products': recommendations_dict, 'content_based_products': recommendations_dict_2})



@app.route("/cart" , methods=['GET'])
def get_collab_recommendations():
    userid = request.args.get('userid')


    from hybrid import get_recommendations , userIdEncoder
    
    userIdEncoded = userIdEncoder(userid)
    recommendation = get_recommendations(userIdEncoded)
    recommendations_dict = recommendation.to_dict(orient='records')
    return jsonify({'Collab_products': recommendations_dict})

@app.route("/hybrid" , methods=['GET'])
def get_hybrid_recommendations():
    userid = request.args.get('userid')


    from hybrid import hybrid_recommendation_v2 , userIdEncoder
    
    userIdEncoded = userIdEncoder(userid)
    recommendation = hybrid_recommendation_v2(userIdEncoded)
    recommendations_dict = recommendation.to_dict(orient='records')
    return jsonify({'hybrid_products': recommendations_dict})

@app.route("/categories")
def get_categories():
    sys.path.insert(0, 'C:/Users/swapn/Desktop/final rs/')
    from content_based_rs2 import recommend_products_with_profiles, get_all_categories
    all_categories = get_all_categories()
    return jsonify({'categories': all_categories})


if __name__ == '__main__':
    app.run(debug=True)