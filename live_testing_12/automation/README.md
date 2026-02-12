# Lead Generation Automation for supyagent.com

Automated monitoring and lead generation tools for finding and engaging with AI agent developers across Reddit, Twitter/X, and GitHub.

## Overview

This automation suite helps you:
- **Discover** conversations about AI agents, LangChain, AutoGPT, and related topics
- **Identify** high-value engagement opportunities (questions, showcases, discussions)
- **Track** influential developers and new projects in the space
- **Organize** leads for manual, personalized follow-up

**Important**: These tools are designed to assist with finding opportunities, not to spam. All engagement should be authentic, helpful, and value-driven.

## Setup

### 1. Install Dependencies

```bash
cd automation
pip install -r requirements.txt
```

### 2. Get API Credentials

#### Reddit
1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Select "script" type
4. Note your `client_id` and `client_secret`

#### Twitter/X
1. Apply for developer access at https://developer.twitter.com/
2. Create a new app
3. Generate API keys and access tokens
4. Note: Free tier has limited search capabilities

#### GitHub
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo`, `read:user`
4. Copy the generated token

### 3. Set Environment Variables

Create a `.env` file in the automation directory:

```bash
# Reddit
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret

# Twitter/X
TWITTER_BEARER_TOKEN=your_bearer_token
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret

# GitHub
GITHUB_TOKEN=your_github_token
```

Or export them in your shell:

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export GITHUB_TOKEN="your_github_token"
# etc...
```

## Usage

### Reddit Monitor

Monitors subreddits for posts about AI agents and related topics.

```bash
python reddit_monitor.py
```

**What it does:**
- Scans 10 relevant subreddits every 30 minutes
- Identifies posts with keywords like "ai agent", "langchain", "autogpt"
- Filters for high-value posts (minimum upvotes/comments)
- Categorizes posts: QUESTION, SHOWCASE, RECOMMENDATION, DISCUSSION
- Saves opportunities to `reddit_leads_YYYY-MM-DD.json`

**Output:** Daily JSON files with engagement opportunities

### Twitter Monitor

Monitors Twitter for tweets about AI agents and tracks influencers.

```bash
python twitter_monitor.py
```

**What it does:**
- Searches keywords and hashtags every hour
- Monitors tweets from key influencers in the AI agent space
- Filters by minimum follower count and engagement
- Categorizes: QUESTION, SHOWCASE, RECOMMENDATION, EDUCATIONAL, DISCUSSION
- Saves opportunities to `twitter_leads_YYYY-MM-DD.json`

**Output:** Daily JSON files with engagement opportunities

### GitHub Monitor

Monitors GitHub repositories for issues, discussions, and new projects.

```bash
python github_monitor.py
```

**What it does:**
- Scans major AI agent repositories every 2 hours
- Identifies issues with labels like "help wanted", "good first issue"
- Finds new repositories about AI agents
- Categorizes: GOOD_FIRST_ISSUE, HELP_WANTED, QUESTION, BUG, FEATURE_REQUEST
- Saves opportunities to `github_leads_YYYY-MM-DD.json`

**Output:** Daily JSON files with engagement opportunities

## Monitoring All Platforms

To run all monitors simultaneously, use a process manager or separate terminal windows:

```bash
# Terminal 1
python reddit_monitor.py

# Terminal 2
python twitter_monitor.py

# Terminal 3
python github_monitor.py
```

Or use a tool like `tmux` or `screen` to run them in the background.

## Output Files

All monitors create daily JSON files with this structure:

```json
[
  {
    "title": "How to build a multi-agent system?",
    "url": "https://reddit.com/...",
    "category": "QUESTION",
    "score": 45,
    "comments": 12,
    "keywords": ["multi-agent", "building agent"],
    "created": "2024-01-15T10:30:00"
  }
]
```

## Daily Workflow

1. **Morning (9 AM)**: Review overnight leads
   - Check `*_leads_YYYY-MM-DD.json` files
   - Identify top 5-10 opportunities to engage with

2. **Engagement**: Respond authentically
   - Provide helpful, detailed answers
   - Share relevant resources or code examples
   - Mention supyagent only when truly relevant

3. **Evening (6 PM)**: Quick check
   - Review new leads from the day
   - Plan tomorrow's content based on trending topics

## Best Practices

### DO:
✅ Provide genuine value in every interaction
✅ Be helpful first, promotional second (90/10 rule)
✅ Personalize all outreach based on the person's specific needs
✅ Share code examples, tutorials, and resources
✅ Build relationships over time

### DON'T:
❌ Spam or over-promote
❌ Use generic copy-paste responses
❌ Engage with every single post (be selective)
❌ Violate platform terms of service
❌ Auto-reply without human review

## Customization

### Adjust Keywords

Edit the `KEYWORDS` list in each script to match your target audience:

```python
KEYWORDS = [
    "your custom keyword",
    "another keyword",
    # ...
]
```

### Change Monitoring Frequency

Adjust the `time.sleep()` values:

```python
time.sleep(1800)  # 30 minutes
time.sleep(3600)  # 1 hour
time.sleep(7200)  # 2 hours
```

### Filter Criteria

Modify minimum thresholds:

```python
MIN_SCORE = 10        # Minimum upvotes/likes
MIN_COMMENTS = 3      # Minimum comments
MIN_FOLLOWERS = 50    # Minimum follower count
```

## Troubleshooting

### Rate Limiting
If you hit API rate limits:
- Increase sleep intervals between requests
- Reduce the number of keywords/subreddits monitored
- Use authenticated requests (higher limits)

### No Results
If monitors aren't finding opportunities:
- Check that keywords are relevant
- Lower minimum thresholds (score, followers)
- Verify API credentials are correct
- Check platform API status

### Authentication Errors
- Verify environment variables are set correctly
- Check that API tokens haven't expired
- Ensure you have necessary API access levels

## Analytics

Track your engagement effectiveness:

1. **Response Rate**: How many opportunities you engage with
2. **Conversion Rate**: Leads → Website visits → Sign-ups
3. **Platform Performance**: Which platform drives best leads
4. **Content Performance**: Which topics get most engagement

## Next Steps

1. **Start with one platform**: Begin with Reddit or GitHub (easier setup)
2. **Run for 1 week**: Gather data on what works
3. **Refine keywords**: Adjust based on results
4. **Scale gradually**: Add more platforms as you optimize
5. **Build templates**: Create response templates for common questions
6. **Measure ROI**: Track which leads convert to customers

## Support

For issues or questions about these automation tools:
- Check the main strategy document: `lead_generation_strategy.md`
- Review API documentation for each platform
- Ensure you're following platform terms of service

---

**Remember**: Automation is a tool to help you find opportunities, but genuine human engagement is what builds relationships and trust. Use these tools to work smarter, not to spam harder.
