from dotenv import load_dotenv
import os

load_dotenv()  # Muat file .env

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB
