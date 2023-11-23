import os
from os.path import join, dirname
from dotenv import load_dotenv

load_dotenv(verbose=True)

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)

openai_api_type = os.environ["OPENAI_API_TYPE"]
openai_api_key = os.environ["OPENAI_API_KEY"]
openai_api_base = os.environ["OPENAI_API_BASE"]
openai_api_version = os.environ["OPENAI_API_VERSION"]
deployment_name = os.environ["OPENAI_DEPLOYMENT"]
model_name = os.environ["OPENAI_MODEL_NAME"]
