from pymongo import MongoClient

# Replace with your VPS IP address and MongoDB port (default: 27017)
MONGO_URI = "mongodb://82.112.231.98:27017/"

# Connect to the MongoDB server on your VPS
client = MongoClient(MONGO_URI)

# Use the database you created on your VPS
db = client["face_database"]  # Replace with the name of your database

# Collections
visitor_data = db["registered_faces"]  # Collection for student data
