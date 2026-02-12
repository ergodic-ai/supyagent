"""
GitHub Lead Generation Monitor for supyagent.com

This script monitors GitHub repositories related to AI agents for:
- New issues that need help
- Discussions about agent development
- New repositories in the space
- Contributors who might be interested in supyagent

Setup:
1. Install: pip install PyGithub
2. Create GitHub personal access token: https://github.com/settings/tokens
3. Set environment variable: GITHUB_TOKEN
4. Run: python github_monitor.py
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict
from github import Github, GithubException

# Configuration
TARGET_REPOS = [
    "langchain-ai/langchain",
    "Significant-Gravitas/AutoGPT",
    "microsoft/semantic-kernel",
    "hwchase17/langchain",
    "yoheinakajima/babyagi",
    "TransformerOptimus/SuperAGI",
    "reworkd/AgentGPT",
    "e2b-dev/e2b",
    "microsoft/autogen"
]

TOPIC_KEYWORDS = [
    "ai-agents",
    "llm",
    "langchain",
    "autonomous-agents",
    "agent-framework",
    "llm-agents"
]

# Labels that indicate good engagement opportunities
HELPFUL_LABELS = [
    "help wanted",
    "good first issue",
    "question",
    "bug",
    "enhancement",
    "documentation"
]

class GitHubMonitor:
    def __init__(self):
        """Initialize GitHub API client"""
        token = os.getenv("GITHUB_TOKEN")
        self.gh = Github(token)
        self.seen_items = self.load_seen_items()
    
    def load_seen_items(self) -> set:
        """Load previously seen issue/discussion IDs"""
        try:
            with open("automation/seen_github.json", "r") as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
    
    def save_seen_items(self):
        """Save seen item IDs"""
        with open("automation/seen_github.json", "w") as f:
            json.dump(list(self.seen_items), f)
    
    def monitor_repo_issues(self, repo_name: str) -> List[Dict]:
        """Monitor issues in a repository"""
        opportunities = []
        
        try:
            repo = self.gh.get_repo(repo_name)
            
            # Get recent open issues
            issues = repo.get_issues(state="open", sort="created", direction="desc")
            
            for issue in issues[:20]:  # Check last 20 issues
                item_id = f"issue_{issue.id}"
                
                if item_id in self.seen_items:
                    continue
                
                # Skip pull requests
                if issue.pull_request:
                    continue
                
                # Check if issue is recent (last 7 days)
                age = datetime.now(issue.created_at.tzinfo) - issue.created_at
                if age.days > 7:
                    continue
                
                # Get labels
                labels = [label.name for label in issue.labels]
                
                # Check if it has helpful labels
                has_helpful_label = any(
                    label.lower() in [hl.lower() for hl in HELPFUL_LABELS]
                    for label in labels
                )
                
                # Check for keywords in title/body
                text = f"{issue.title} {issue.body or ''}".lower()
                has_keywords = any(kw in text for kw in [
                    "agent", "langchain", "tool", "integration", 
                    "automation", "workflow"
                ])
                
                if has_helpful_label or has_keywords:
                    opportunity = {
                        "type": "ISSUE",
                        "repo": repo_name,
                        "title": issue.title,
                        "url": issue.html_url,
                        "number": issue.number,
                        "labels": labels,
                        "comments": issue.comments,
                        "created": issue.created_at.isoformat(),
                        "author": issue.user.login,
                        "category": self.categorize_issue(issue, labels)
                    }
                    
                    opportunities.append(opportunity)
                    self.seen_items.add(item_id)
            
        except GithubException as e:
            print(f"Error monitoring {repo_name}: {e}")
        
        return opportunities
    
    def categorize_issue(self, issue, labels: List[str]) -> str:
        """Categorize issue by type"""
        labels_lower = [l.lower() for l in labels]
        
        if "good first issue" in labels_lower:
            return "GOOD_FIRST_ISSUE"
        elif "help wanted" in labels_lower:
            return "HELP_WANTED"
        elif "question" in labels_lower:
            return "QUESTION"
        elif "bug" in labels_lower:
            return "BUG"
        elif "enhancement" in labels_lower or "feature" in labels_lower:
            return "FEATURE_REQUEST"
        elif "documentation" in labels_lower:
            return "DOCUMENTATION"
        else:
            return "OTHER"
    
    def monitor_repo_discussions(self, repo_name: str) -> List[Dict]:
        """Monitor discussions in a repository"""
        opportunities = []
        
        try:
            repo = self.gh.get_repo(repo_name)
            
            # Note: PyGithub doesn't have full discussions support yet
            # This would require GraphQL API
            # For now, we'll skip this and focus on issues
            
        except Exception as e:
            print(f"Error monitoring discussions in {repo_name}: {e}")
        
        return opportunities
    
    def find_new_repos(self) -> List[Dict]:
        """Find new repositories about AI agents"""
        opportunities = []
        
        try:
            # Search for recently created repos
            for keyword in TOPIC_KEYWORDS[:2]:  # Limit to avoid rate limits
                query = f"{keyword} created:>{(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}"
                repos = self.gh.search_repositories(query, sort="stars", order="desc")
                
                for repo in repos[:5]:  # Top 5 per keyword
                    item_id = f"repo_{repo.id}"
                    
                    if item_id in self.seen_items:
                        continue
                    
                    # Filter by minimum stars
                    if repo.stargazers_count < 10:
                        continue
                    
                    opportunity = {
                        "type": "NEW_REPO",
                        "name": repo.full_name,
                        "description": repo.description,
                        "url": repo.html_url,
                        "stars": repo.stargazers_count,
                        "language": repo.language,
                        "created": repo.created_at.isoformat(),
                        "keyword": keyword,
                        "category": "NEW_REPOSITORY"
                    }
                    
                    opportunities.append(opportunity)
                    self.seen_items.add(item_id)
                
                time.sleep(2)  # Rate limiting
        
        except GithubException as e:
            print(f"Error searching for new repos: {e}")
        
        return opportunities
    
    def find_active_contributors(self, repo_name: str) -> List[Dict]:
        """Find active contributors who might be interested"""
        contributors = []
        
        try:
            repo = self.gh.get_repo(repo_name)
            
            # Get recent contributors
            for contributor in repo.get_contributors()[:10]:
                item_id = f"contributor_{contributor.id}"
                
                if item_id in self.seen_items:
                    continue
                
                # Get user details
                user = self.gh.get_user(contributor.login)
                
                # Check if they have email public
                email = user.email if user.email else "Not public"
                
                contributor_info = {
                    "type": "CONTRIBUTOR",
                    "username": contributor.login,
                    "name": user.name,
                    "email": email,
                    "bio": user.bio,
                    "company": user.company,
                    "contributions": contributor.contributions,
                    "followers": user.followers,
                    "repo": repo_name,
                    "category": "ACTIVE_CONTRIBUTOR"
                }
                
                contributors.append(contributor_info)
                self.seen_items.add(item_id)
        
        except GithubException as e:
            print(f"Error finding contributors in {repo_name}: {e}")
        
        return contributors
    
    def monitor_all(self) -> List[Dict]:
        """Monitor all repositories and search for new ones"""
        all_opportunities = []
        
        # Monitor target repositories
        for repo in TARGET_REPOS:
            print(f"Scanning {repo}...")
            
            # Check issues
            issues = self.monitor_repo_issues(repo)
            all_opportunities.extend(issues)
            
            time.sleep(1)  # Rate limiting
        
        # Find new repositories
        print("Searching for new repositories...")
        new_repos = self.find_new_repos()
        all_opportunities.extend(new_repos)
        
        return all_opportunities
    
    def save_opportunities(self, opportunities: List[Dict]):
        """Save opportunities to file"""
        if not opportunities:
            return
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"automation/github_leads_{date_str}.json"
        
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
        print(f"Found {len(opportunities)} engagement opportunities on GitHub!")
        print(f"{'='*80}\n")
        
        # Group by category
        by_category = {}
        for opp in opportunities:
            cat = opp["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(opp)
        
        for category, opps in sorted(by_category.items()):
            print(f"\n{category} ({len(opps)} items):")
            print("-" * 80)
            
            for opp in opps[:3]:
                if opp["type"] == "ISSUE":
                    print(f"\n  📦 {opp['repo']}")
                    print(f"  🎯 Issue #{opp['number']}: {opp['title'][:60]}...")
                    print(f"  🔗 {opp['url']}")
                    print(f"  🏷️  Labels: {', '.join(opp['labels'][:3])}")
                    print(f"  💬 {opp['comments']} comments")
                elif opp["type"] == "NEW_REPO":
                    print(f"\n  ⭐ {opp['name']} ({opp['stars']} stars)")
                    print(f"  📝 {opp['description'][:60]}...")
                    print(f"  🔗 {opp['url']}")
                    print(f"  💻 Language: {opp['language']}")
        
        print(f"\n{'='*80}\n")

def main():
    """Main monitoring loop"""
    print("🐙 Starting GitHub Lead Monitor for supyagent.com")
    print(f"Monitoring {len(TARGET_REPOS)} repositories\n")
    
    monitor = GitHubMonitor()
    
    try:
        while True:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scan...")
            
            opportunities = monitor.monitor_all()
            monitor.save_opportunities(opportunities)
            monitor.print_summary(opportunities)
            monitor.save_seen_items()
            
            print(f"\nNext scan in 2 hours...")
            time.sleep(7200)  # 2 hours
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        monitor.save_seen_items()

if __name__ == "__main__":
    # Check for credentials
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ Error: GitHub token not found!")
        print("\nPlease set environment variable:")
        print("  export GITHUB_TOKEN='your_github_token'")
        print("\nCreate token at: https://github.com/settings/tokens")
        print("Required scopes: repo, read:user")
        exit(1)
    
    main()
