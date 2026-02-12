"""
Twitter/X Lead Generation Monitor for supyagent.com

This script monitors Twitter for posts about AI agents, LangChain, AutoGPT,
and related topics. It identifies engagement opportunities and influential
accounts to follow.

Setup:
1. Install: pip install tweepy
2. Get Twitter API access: https://developer.twitter.com/
3. Set environment variables: TWITTER_API_KEY, TWITTER_API_SECRET, 
   TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
4. Run: python twitter_monitor.py

Note: Twitter API v2 requires elevated access for search. Consider using
the free tier strategically or Twitter's web interface for initial research.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict
import tweepy

# Configuration
KEYWORDS = [
    "ai agent",
    "langchain",
    "autogpt", 
    "autonomous agent",
    "multi-agent system",
    "llm tools",
    "agent framework",
    "building agents"
]

HASHTAGS = [
    "#AIAgents",
    "#LangChain",
    "#AutoGPT",
    "#LLM",
    "#BuildInPublic",
    "#AIEngineering",
    "#MachineLearning"
]

# Influencers to monitor
KEY_ACCOUNTS = [
    "langchainai",
    "hwchase17",  # Harrison Chase (LangChain)
    "yoheinakajima",  # BabyAGI creator
    "AndrewYNg",
    "karpathy",
    "OpenAI",
    "ClaudeAI"
]

# Minimum follower count for engagement
MIN_FOLLOWERS = 50
MIN_ENGAGEMENT_SCORE = 2  # likes + retweets

class TwitterMonitor:
    def __init__(self):
        """Initialize Twitter API client"""
        # Twitter API v2
        self.client = tweepy.Client(
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_SECRET")
        )
        
        self.seen_tweets = self.load_seen_tweets()
    
    def load_seen_tweets(self) -> set:
        """Load previously seen tweet IDs"""
        try:
            with open("automation/seen_tweets.json", "r") as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
    
    def save_seen_tweets(self):
        """Save seen tweet IDs"""
        with open("automation/seen_tweets.json", "w") as f:
            json.dump(list(self.seen_tweets), f)
    
    def search_keyword(self, keyword: str, max_results: int = 20) -> List[Dict]:
        """Search for tweets containing keyword"""
        opportunities = []
        
        try:
            # Search recent tweets
            tweets = self.client.search_recent_tweets(
                query=f"{keyword} -is:retweet lang:en",
                max_results=max_results,
                tweet_fields=["created_at", "public_metrics", "author_id"],
                user_fields=["username", "public_metrics"],
                expansions=["author_id"]
            )
            
            if not tweets.data:
                return opportunities
            
            # Create user lookup
            users = {user.id: user for user in tweets.includes.get("users", [])}
            
            for tweet in tweets.data:
                if tweet.id in self.seen_tweets:
                    continue
                
                author = users.get(tweet.author_id)
                if not author:
                    continue
                
                # Filter by follower count
                if author.public_metrics["followers_count"] < MIN_FOLLOWERS:
                    continue
                
                # Calculate engagement score
                metrics = tweet.public_metrics
                engagement = metrics["like_count"] + metrics["retweet_count"]
                
                if engagement < MIN_ENGAGEMENT_SCORE:
                    continue
                
                # Categorize tweet
                category = self.categorize_tweet(tweet.text)
                
                opportunity = {
                    "tweet_id": tweet.id,
                    "author": author.username,
                    "author_followers": author.public_metrics["followers_count"],
                    "text": tweet.text,
                    "url": f"https://twitter.com/{author.username}/status/{tweet.id}",
                    "likes": metrics["like_count"],
                    "retweets": metrics["retweet_count"],
                    "replies": metrics["reply_count"],
                    "created": tweet.created_at.isoformat(),
                    "keyword": keyword,
                    "category": category
                }
                
                opportunities.append(opportunity)
                self.seen_tweets.add(tweet.id)
            
        except Exception as e:
            print(f"Error searching '{keyword}': {e}")
        
        return opportunities
    
    def categorize_tweet(self, text: str) -> str:
        """Categorize tweet by type"""
        text_lower = text.lower()
        
        # Question
        if "?" in text or any(q in text_lower for q in ["how to", "how do i", "help", "anyone know"]):
            return "QUESTION"
        
        # Showcase
        if any(s in text_lower for s in ["built", "made", "created", "launched", "shipped"]):
            return "SHOWCASE"
        
        # Looking for recommendations
        if any(r in text_lower for r in ["recommend", "suggestions", "what's the best", "looking for"]):
            return "RECOMMENDATION"
        
        # Tutorial/educational
        if any(t in text_lower for t in ["tutorial", "guide", "how to", "learn", "thread"]):
            return "EDUCATIONAL"
        
        return "DISCUSSION"
    
    def monitor_influencer(self, username: str) -> List[Dict]:
        """Monitor tweets from a specific influencer"""
        opportunities = []
        
        try:
            # Get user
            user = self.client.get_user(username=username)
            if not user.data:
                return opportunities
            
            # Get recent tweets
            tweets = self.client.get_users_tweets(
                user.data.id,
                max_results=10,
                tweet_fields=["created_at", "public_metrics"]
            )
            
            if not tweets.data:
                return opportunities
            
            for tweet in tweets.data:
                if tweet.id in self.seen_tweets:
                    continue
                
                # Check if relevant to AI agents
                if not any(kw in tweet.text.lower() for kw in ["agent", "langchain", "llm", "ai"]):
                    continue
                
                opportunity = {
                    "tweet_id": tweet.id,
                    "author": username,
                    "type": "INFLUENCER",
                    "text": tweet.text,
                    "url": f"https://twitter.com/{username}/status/{tweet.id}",
                    "likes": tweet.public_metrics["like_count"],
                    "retweets": tweet.public_metrics["retweet_count"],
                    "created": tweet.created_at.isoformat(),
                    "category": "INFLUENCER_POST"
                }
                
                opportunities.append(opportunity)
                self.seen_tweets.add(tweet.id)
        
        except Exception as e:
            print(f"Error monitoring @{username}: {e}")
        
        return opportunities
    
    def monitor_all(self) -> List[Dict]:
        """Monitor all keywords and influencers"""
        all_opportunities = []
        
        # Search keywords
        for keyword in KEYWORDS[:3]:  # Limit to avoid rate limits
            print(f"Searching: {keyword}")
            opportunities = self.search_keyword(keyword)
            all_opportunities.extend(opportunities)
            time.sleep(2)  # Rate limiting
        
        # Search hashtags
        for hashtag in HASHTAGS[:2]:
            print(f"Searching: {hashtag}")
            opportunities = self.search_keyword(hashtag)
            all_opportunities.extend(opportunities)
            time.sleep(2)
        
        # Monitor influencers
        for account in KEY_ACCOUNTS[:3]:
            print(f"Monitoring: @{account}")
            opportunities = self.monitor_influencer(account)
            all_opportunities.extend(opportunities)
            time.sleep(2)
        
        return all_opportunities
    
    def save_opportunities(self, opportunities: List[Dict]):
        """Save opportunities to file"""
        if not opportunities:
            return
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"automation/twitter_leads_{date_str}.json"
        
        existing = []
        try:
            with open(filename, "r") as f:
                existing = json.load(f)
        except FileNotFoundError:
            pass
        
        existing.extend(opportunities)
        
        with open(filename, "w") as f:
            json.dump(existing, f, indent=2)
        
        print(f"\n✅ Saved {len(opportunities)} opportunities to {filename}")
    
    def print_summary(self, opportunities: List[Dict]):
        """Print summary of opportunities"""
        if not opportunities:
            print("\nNo new opportunities found this run.")
            return
        
        print(f"\n{'='*80}")
        print(f"Found {len(opportunities)} engagement opportunities on Twitter!")
        print(f"{'='*80}\n")
        
        # Sort by engagement
        sorted_opps = sorted(
            opportunities,
            key=lambda x: x.get("likes", 0) + x.get("retweets", 0),
            reverse=True
        )
        
        # Group by category
        by_category = {}
        for opp in sorted_opps:
            cat = opp["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(opp)
        
        for category, opps in sorted(by_category.items()):
            print(f"\n{category} ({len(opps)} tweets):")
            print("-" * 80)
            
            for opp in opps[:3]:
                print(f"\n  👤 @{opp['author']} ({opp.get('author_followers', 'N/A')} followers)")
                print(f"  💬 {opp['text'][:100]}...")
                print(f"  🔗 {opp['url']}")
                print(f"  📊 {opp['likes']} likes | {opp['retweets']} retweets")
        
        print(f"\n{'='*80}\n")

def main():
    """Main monitoring loop"""
    print("🐦 Starting Twitter Lead Monitor for supyagent.com")
    print(f"Monitoring {len(KEYWORDS)} keywords, {len(HASHTAGS)} hashtags, {len(KEY_ACCOUNTS)} accounts\n")
    
    monitor = TwitterMonitor()
    
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scan...")
            
            opportunities = monitor.monitor_all()
            monitor.save_opportunities(opportunities)
            monitor.print_summary(opportunities)
            monitor.save_seen_tweets()
            
            print(f"\nNext scan in 1 hour...")
            time.sleep(3600)  # 1 hour (to respect rate limits)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        monitor.save_seen_tweets()

if __name__ == "__main__":
    # Check for credentials
    required_vars = [
        "TWITTER_BEARER_TOKEN",
        "TWITTER_API_KEY",
        "TWITTER_API_SECRET",
        "TWITTER_ACCESS_TOKEN",
        "TWITTER_ACCESS_SECRET"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Error: Missing Twitter API credentials!")
        print(f"\nPlease set environment variables: {', '.join(missing)}")
        print("\nGet credentials at: https://developer.twitter.com/")
        exit(1)
    
    main()
