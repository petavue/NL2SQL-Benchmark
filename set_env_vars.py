import os

ANY_SCALE_API_KEY_PETAVUE = "esecret_***************"
ANY_SCALE_API_KEY_PERSONAL = "esecret_***************"
PETAVUE_OPENAI_KEY = "sk-**********************"
AWS_ACCESS_KEY_ID = "*************"
AWS_SECRET_ACCESS_KEY = "*******************************"
HUGGING_FACE_TOKEN="hf_***************"

# Set environment variables
os.environ['ANY_SCALE_API_KEY'] = ANY_SCALE_API_KEY_PETAVUE
os.environ['OPENAI_KEY'] = PETAVUE_OPENAI_KEY
os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
os.environ['HUGGING_FACE_TOKEN'] = HUGGING_FACE_TOKEN
