import sys
from flask import Flask , request, jsonify
import subprocess




app = Flask(__name__)

@app.route("/recommend")
def recommend():
    product_name = request.args.get('product_name')

    if product_name:
        # Ensure the Python script is in your system path
        sys.path.insert(0, 'C:/Users/swapn/Desktop/dataset/Rsystem.py')

        # Import the function
        from Rsystem import show_recommendations

        # Get recommendations
        recommendations = show_recommendations(product_name)

        print(f"Received product_name: {product_name}")

        # Return recommendations as a JSON
        return jsonify(recommendations)
    else:
        return jsonify({"error": "No product name provided"})



if __name__ == '__main__':
    app.run(debug=True)