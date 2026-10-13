"""Run offline without a .env, production database, or real API credentials."""
import os

os.environ["OPENAI_API_KEY"] = "offline-test-key"
os.environ["DATABASE_URL"] = "sqlite://"