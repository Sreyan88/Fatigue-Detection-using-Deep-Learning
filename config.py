from dotenv import load_dotenv
import os
dotenv_path = os.getcwd()+'/.env'
dotenv_path=dotenv_path.replace('\\',"//")
load_dotenv(dotenv_path)