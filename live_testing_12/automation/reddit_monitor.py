"""
Reddit Lead Generation Monitor for supyagent.com

This script monitors relevant subreddits for posts about AI agents, LangChain,
AutoGPT, and related topics. It identifies high-value engagement opportunities
and saves them for manual review.

Setup:
1. Install: pip install praw
2. Create Reddit app: https://www.reddit.com/prefs/apps
3. Set environment variables: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
4. Run: python reddit_monitor.py
"""

import praw
import os
import json
import time
from datetime import datetime
from typing import List, Dict

# Configuration
KEYWORDS = [
    "ai agent", "langchain", "autogpt", "building agent",
    "agent framework", "llm automation", "autonomous agent",
    "multi-agent", "agent orchestration", "llm tools",
    "openai function calling", "tool calling", "agent memory"
]

SUBREDDITS = [
    "LocalLLaMA",
    "OpenAI", 
    "LangChain",
    "AutoGPT",
    "MachineLearning",
    "ChatGPTCoding",
    "artificial",
    "SideProject",
    "programming",
    "learnmachinelearning"
]

# Minimum thresholds for "high-value" posts
MIN_SCORE = 10
MIN_COMMENTS = 3

class RedditMonitor:
    def __init__(self):
        """Initialize Reddit API client"""
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="supyagent-lead-monitor/1.0"
        )
        self.seen_posts = self.load_seen_posts()
        
    def load_seen_posts(self) -> set:
        """Load previously seen post IDs"""
        try:
            with open("automation/seen_posts.json", "r") as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
    
    def save_seen_posts(self):
        """Save seen post IDs to avoid duplicates"""
        with open("automation/seen_posts.json", "w") as f:
            json.dump(list(self.seen_posts), f)
    
    def check_keywords(self, text: str) -> List[str]:
        """Check if text contains any keywords"""
        text_lower = text.lower()
        return [kw for kw in KEYWORDS if kw in text_lower]
    
    def is_high_value(self, post) -> bool:
        """Determine if post is high-value for engagement"""
        # Check score and comments
        if post.score < MIN_SCORE and post.num_comments < MIN_COMMENTS:
            return False
        
        # Avoid very old posts
        post_age_hours = (time.time() - post.created_utc) / 3600
        if post_age_hours > 48:  # Older than 2 days
            return False
        
        return True
    
    def categorize_post(self, post, keywords: List[str]) -> str:
        """Categorize post by type of engagement opportunity"""
        title_lower = post.title.lower()
        text_lower = f"{post.title} {post.selftext}".lower()
        
        # Question posts - highest priority
        if any(q in title_lower for q in ["how to", "how do i", "help with", "question", "?"]):
            return "QUESTION"
        
        # Show and tell - good for engagement
        if any(s in title_lower for s in ["built", "made", "created", "show", "my project"]):
            return "SHOWCASE"
        
        # Looking for tools/recommendations
        if any(t in text_lower for t in ["looking for", "recommend", "what tool", "best way"]):
            return "RECOMMENDATION"
        
        # Discussion/opinion
        return "DISCUSSION"
    
    def monitor_subreddit(self, subreddit_name: str) -> List[Dict]:
        """Monitor a single subreddit for relevant posts"""
        opportunities = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Check both new and hot posts
            posts = list(subreddit.new(limit=50)) + list(subreddit.hot(limit=25))
            
            for post in posts:
                # Skip if already seen
                if post.id in self.seen_posts:
                    continue
                
                # Check for keywords
                post_text = f"{post.title} {post.selftext}"
                matched_keywords = self.check_keywords(post_text)
                
                if matched_keywords and self.is_high_value(post):
                    category = self.categorize_post(post, matched_keywords)
                    
                    opportunity = {
                        "subreddit": subreddit_name,
                        "title": post.title,
                        "url": f"https://reddit.com{post.permalink}",
                        "score": post.score,
                        "comments": post.num_comments,
                        "keywords": matched_keywords,
                        "category": category,
                        "created": datetime.fromtimestamp(post.created_utc).isoformat(),
                        "author": str(post.author)
                    }
                    
                    opportunities.append(opportunity)
                    self.seen_posts.add(post.id)
                    
        except Exception as e:
            print(f"Error monitoring r/{subreddit_name}: {e}")
        
        return opportunities
    
    def monitor_all(self) -> List[Dict]:
        """Monitor all configured subreddits"""
        all_opportunities = []
        
        for subreddit in SUBREDDITS:
            print(f"Scanning r/{subreddit}...")
            opportunities = self.monitor_subreddit(subreddit)
            all_opportunities.extend(opportunities)
            time.sleep(2)  # Rate limiting
        
        return all_opportunities
    
    def save_opportunities(self, opportunities: List[Dict]):
        """Save opportunities to file for review"""
        if not opportunities:
            return
        
        # Append to daily log
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"automation/reddit_leads_{date_str}.json"
        
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
        """Print summary of opportunities found"""
        if not opportunities:
            print("\nNo new opportunities found this run.")
            return
        
        print(f"\n{'='*80}")
        print(f"Found {len(opportunities)} engagement opportunities!")
        print(f"{'='*80}\n")
        
        # Group by category
        by_category = {}
        for opp in opportunities:
            cat = opp["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(opp)
        
        for category, opps in sorted(by_category.items()):
            print(f"\n{category} ({len(opps)} posts):")
            print("-" * 80)
            
            for opp in opps[:3]:  # Show top 3 per category
                print(f"\n  📍 r/{opp['subreddit']}")
                print(f"  📝 {opp['title'][:70]}...")
                print(f"  🔗 {opp['url']}")
                print(f"  📊 {opp['score']} upvotes | {opp['comments']} comments")
                print(f"  🏷️  Keywords: {', '.join(opp['keywords'][:3])}")
        
        print(f"\n{'='*80}\n")

def main():
    """Main monitoring loop"""
    print("🤖 Starting Reddit Lead Monitor for supyagent.com")
    print(f"Monitoring {len(SUBREDDITS)} subreddits for {len(KEYWORDS)} keywords\n")
    
    monitor = RedditMonitor()
    
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scan...")
            
            opportunities = monitor.monitor_all()
            monitor.save_opportunities(opportunities)
            monitor.print_summary(opportunities)
            monitor.save_seen_posts()
            
            print(f"\nNext scan in 30 minutes...")
            time.sleep(1800)  # 30 minutes
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        monitor.save_seen_posts()

if __name__ == "__main__":
    # Check for credentials
    if not os.getenv("REDDIT_CLIENT_ID") or not os.getenv("REDDIT_CLIENT_SECRET"):
        print("❌ Error: Reddit API credentials not found!")
        print("\nPlease set environment variables:")
        print("  export REDDIT_CLIENT_ID='your_client_id'")
        print("  export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("\nGet credentials at: https://www.reddit.com/prefs/apps")
        exit(1)
    
    main()
