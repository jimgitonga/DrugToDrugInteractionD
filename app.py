
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_compress import Compress
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import json
import redis
import logging
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from drugData import DDIPredictor

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
Compress(app)  # Enable compression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure rate limiting - using newer Flask-Limiter initialization
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["500 per day", "100 per hour"]
)
limiter.init_app(app)

# Initialize Redis client for caching
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()  # Test connection
    USE_REDIS = True
    logger.info("Redis connected successfully, using Redis for caching")
except (redis.ConnectionError, redis.exceptions.ConnectionError):
    USE_REDIS = False
    logger.warning("Redis connection failed, falling back to in-memory cache")

# Cache settings
CACHE_EXPIRY = 60 * 60 * 24  # 24 hours in seconds
PREDICTION_CACHE = {}  # In-memory cache fallback

def get_from_cache(key):
    """Get prediction from cache (Redis or in-memory)"""
    if USE_REDIS:
        cached_data = redis_client.get(key)
        if cached_data:
            return json.loads(cached_data)
    else:
        if key in PREDICTION_CACHE:
            return PREDICTION_CACHE[key]
    return None

def set_in_cache(key, value):
    """Store prediction in cache (Redis or in-memory)"""
    if USE_REDIS:
        redis_client.setex(key, CACHE_EXPIRY, json.dumps(value))
    else:
        PREDICTION_CACHE[key] = value

def get_latest_model_timestamp():
    """Get the most recent model timestamp from models directory"""
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.startswith("ddi_model_")]
    if not model_files:
        raise ValueError("No models found in models directory")
    
    # Get the most recent timestamp
    latest_model = max(model_files)
    timestamp = latest_model.replace("ddi_model_", "").replace(".pt", "")
    logger.info(f"Found latest model with timestamp: {timestamp}")
    return timestamp

# Initialize predictor and load the latest model
predictor = DDIPredictor()
latest_timestamp = get_latest_model_timestamp()
predictor.load_model(latest_timestamp)
logger.info("Model loaded successfully")

# List of common drugs to pre-warm the cache
COMMON_DRUGS = [
    "CHEMBL25", "CHEMBL2", "CHEMBL3", "CHEMBL4", "CHEMBL5",
    "CHEMBL6", "CHEMBL7", "CHEMBL8", "CHEMBL9", "CHEMBL10"
]  # Update with your most commonly queried drugs

def warm_prediction_cache():
    """Pre-warm the cache with predictions for common drugs"""
    logger.info("Pre-warming prediction cache for common drugs")
    
    drug_pairs = []
    for i, drug1 in enumerate(COMMON_DRUGS):
        for drug2 in COMMON_DRUGS[i+1:]:
            drug_pairs.append((drug1, drug2))
    
    # Use ThreadPoolExecutor to speed up the pre-warming
    with ThreadPoolExecutor(max_workers=min(10, len(drug_pairs))) as executor:
        executor.map(process_drug_pair, drug_pairs)
    
    logger.info(f"Cache pre-warmed with predictions for common drugs")

def process_drug_pair(pair):
    """Process a single drug pair prediction with caching"""
    drug1, drug2 = pair
    cache_key = f"{drug1}_{drug2}"
    reverse_key = f"{drug2}_{drug1}"
    
    # Check cache first
    result = get_from_cache(cache_key)
    if result is not None:
        return result
    
    result = get_from_cache(reverse_key)
    if result is not None:
        return result
    
    # Not in cache, perform prediction
    try:
        result = predictor.predict(drug1, drug2, return_details=True)
        
        # Cache the result if it's valid
        if isinstance(result, dict) and 'error' not in result:
            set_in_cache(cache_key, result)
            # Don't need to cache reverse key separately, as we check both directions
        
        return result
    except Exception as e:
        logger.error(f"Error predicting for {drug1}-{drug2}: {str(e)}")
        return {"error": str(e)}

# Pre-warm cache with common drug predictions
warm_prediction_cache()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("60 per minute")
def predict():
    try:
        data = request.get_json()
        drug1 = data['drug1']
        drug2 = data['drug2']
        
        logger.info(f"Received prediction request for {drug1} and {drug2}")
        
        # Process the drug pair
        result = process_drug_pair((drug1, drug2))
        
        response = {
            'success': True,
            'prediction': result
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/batch_predict', methods=['POST'])
@limiter.limit("20 per minute")
def batch_predict():
    try:
        data = request.get_json()
        drug_pairs = data['drug_pairs']
        
        logger.info(f"Received batch prediction request for {len(drug_pairs)} pairs")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(10, len(drug_pairs))) as executor:
            results = list(executor.map(process_drug_pair, drug_pairs))
        
        response = {
            'success': True,
            'predictions': results
        }
        
        logger.info(f"Batch prediction completed")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/cache_status', methods=['GET'])
@limiter.exempt
def cache_status():
    """Endpoint to check cache status"""
    try:
        if USE_REDIS:
            cache_size = redis_client.dbsize()
            cache_type = "Redis"
        else:
            cache_size = len(PREDICTION_CACHE)
            cache_type = "In-memory"
        
        return jsonify({
            'success': True,
            'cache_type': cache_type,
            'cache_size': cache_size
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/clear_cache', methods=['POST'])
@limiter.limit("10 per hour")
def clear_cache():
    """Endpoint to clear prediction cache"""
    try:
        if USE_REDIS:
            redis_client.flushdb()
        else:
            PREDICTION_CACHE.clear()
        
        logger.info("Prediction cache cleared")
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_timestamp': latest_timestamp
    })

@app.errorhandler(429)
def ratelimit_handler(e):
    """Custom response for rate limit errors"""
    return jsonify({
        'success': False,
        'error': 'Rate limit exceeded',
        'message': str(e.description)
    }), 429

if __name__ == '__main__':
    # Set variables for server
    HOST = '0.0.0.0'  # Use your computer's local IP when deploying
    PORT = 5000
    
    if os.environ.get('FLASK_ENV') == 'production':
        # In production, use a production WSGI server
        try:
            from waitress import serve
            logger.info(f"Starting production server on {HOST}:{PORT}")
            serve(app, host=HOST, port=PORT, threads=8)
        except ImportError:
            logger.warning("Waitress not installed, falling back to Flask development server")
            app.run(host=HOST, port=PORT, debug=False, threaded=True)
    else:
        # For development
        logger.info(f"Starting development server on {HOST}:{PORT}")
        app.run(host=HOST, port=PORT, debug=True)