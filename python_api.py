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
    print(userid)
    print(request.args)
    # Ensure the Python script is in your system path
    if userid:
        
        # Import the function
        from collaborative import popular_products
        from content_based_rs2 import recommend_products_with_profiles , userIdEncoder
        # Get recommendations
        recommendations = popular_products()
        userIdEncoded = userIdEncoder(userid)
        print(userIdEncoded)
        recommendations2 = recommend_products_with_profiles(userIdEncoded)
        recommendations2 = recommend_products_with_profiles(4773)
        # recommendations2 = pd.DataFrame(recommendations2)

        # Convert DataFrame to a dictionary
        recommendations_dict = recommendations.to_dict(orient='records')
        recommendations_dict_2 = recommendations2.to_dict(orient='records')
        # Return recommendations as a JSON
        return jsonify({'first_set': recommendations_dict, 'second_set': recommendations_dict_2})
    else:
        from collaborative import popular_products
        from content_based_rs2 import recommend_products_with_profiles , userIdEncoder
        
        recommendations = popular_products()
        recommendations_dict = recommendations.to_dict(orient='records')
        recommendations2 = recommend_products_with_profiles(4773)
        recommendations2 = pd.DataFrame(recommendations2)
        recommendations_dict_2 = recommendations2.to_dict(orient='records')
        return jsonify({'first_set': recommendations_dict, 'second_set': recommendations_dict_2})
        # sys.path.insert(0, 'C:/Users/swapn/Desktop/final rs/')
        # print(product_name)
        # from collaborative import get_recommendations
        # from content_based_rs2 import recommend_products_with_profiles
        # recommended_products_cf = get_recommendations(product_name)
        # recommended_products_cb = recommend_products_with_profiles(4773)
        # if recommended_products_cf==[]:
        #     print("hi")
        # # recommendations = recommended_products
        # response = {'collaborative':recommended_products_cf, 'contentBased':recommended_products_cb}
        # return jsonify(response)


@app.route("/categories")
def get_categories():
    sys.path.insert(0, 'C:/Users/swapn/Desktop/final rs/')
    from content_based_rs2 import recommend_products_with_profiles, get_all_categories
    all_categories = get_all_categories()
    return jsonify({'categories': all_categories})


if __name__ == '__main__':
    app.run(debug=True)