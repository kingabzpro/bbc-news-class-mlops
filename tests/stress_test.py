import os
import random

from dotenv import load_dotenv
from locust import HttpUser, between, task

load_dotenv()

API_KEY = os.getenv("API_KEY")


class NewsClassifierUser(HttpUser):
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)

    # Sample news titles for testing
    sample_titles = [
        "Breaking: Major Tech Company Announces Revolutionary AI Breakthrough",
        "Sports: Local Team Wins Championship After Dramatic Final",
        "Politics: New Bill Passes Senate with Bipartisan Support",
        "Business: Stock Market Reaches New All-Time High",
        "Technology: New Smartphone Features Revolutionary Battery Life",
        "Entertainment: Popular TV Show Announces Final Season",
        "Science: Researchers Discover New Species in Amazon Rainforest",
        "Health: New Study Reveals Benefits of Mediterranean Diet",
        "Education: University Introduces New AI-Focused Curriculum",
        "Environment: Global Climate Summit Reaches Historic Agreement",
    ]

    def on_start(self):
        """Set up the API key for authentication"""
        self.headers = {"X-API-Key": API_KEY}  # Replace with your actual API key

    @task(2)
    def get_info(self):
        """Test the /info endpoint"""
        self.client.get("/info", headers=self.headers)

    @task(8)
    def predict_news(self):
        """Test the /predict endpoint with random news titles"""
        title = random.choice(self.sample_titles)
        self.client.post("/predict", json={"title": title}, headers=self.headers)
