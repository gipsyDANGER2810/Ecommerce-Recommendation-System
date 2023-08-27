import sys
from flask import Flask , request, jsonify
import subprocess
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route("/recommend")
def recommend_popular():



        # Ensure the Python script is in your system path
    sys.path.insert(0, 'C:/Users/swapn/Desktop/final rs/')

        # Import the function
    from collaborative import popular_products
    
    recommendations = popular_products()
    recommendations_dict = recommendations.to_dict(orient='records')

        # Get recommendations
        # Return recommendations as a JSON
    return jsonify(recommendations_dict)




if __name__ == '__main__':
    app.run(debug=True)