"""
Email Outreach Script for supyagent.com

This script helps you send personalized emails to GitHub contributors
and other AI agent developers. It includes templates and personalization
based on their work.

IMPORTANT: This is a HELPER script, not an auto-spammer. Always:
1. Review each email before sending
2. Personalize based on recipient's work
3. Only send to people who would genuinely benefit
4. Respect CAN-SPAM and GDPR regulations

Setup:
1. Install: pip install resend  # or your email service
2. Set environment variable: EMAIL_API_KEY
3. Create a CSV with leads: leads.csv
4. Review and customize emails before sending
"""

import os
import csv
import json
import time
from datetime import datetime
from typing import List, Dict

# Email templates
TEMPLATES = {
    "github_contributor": {
        "subject": "Loved your work on {project}",
        "body": """Hi {name},

I came across your {contribution_type} to {project} and was really impressed by {specific_detail}.

I'm working on supyagent.com, which helps developers build production-ready AI agents. Given your experience with {technology}, I thought you might find {specific_feature} useful for {use_case}.

{optional_tip}

Would love to hear your thoughts! Also happy to answer any questions about {relevant_topic}.

Best,
{sender_name}

P.S. {personal_touch}
"""
    },
    
    "value_first": {
        "subject": "{resource_type} for {their_use_case}",
        "body": """Hi {name},

I saw you're working on {their_project}. I recently published a guide on {relevant_topic} that might save you some time:

{resource_link}

Key points:
• {benefit_1}
• {benefit_2}
• {benefit_3}

{optional_offer}

Hope this helps!

{sender_name}
"""
    },
    
    "follow_up": {
        "subject": "Re: {original_subject}",
        "body": """Hi {name},

Following up on my previous email. I know inboxes get busy!

I wanted to share {new_resource} that's specifically relevant to {their_work}:

{resource_link}

{brief_description}

No pressure to respond - just thought it might be helpful for {their_specific_use_case}.

Cheers,
{sender_name}
"""
    }
}

class EmailOutreach:
    def __init__(self, dry_run=True):
        """
        Initialize email outreach system
        
        Args:
            dry_run: If True, don't actually send emails (just preview)
        """
        self.dry_run = dry_run
        self.sent_log = self.load_sent_log()
        
        # Initialize email service (example with Resend)
        # Replace with your preferred service
        if not dry_run:
            self.api_key = os.getenv("EMAIL_API_KEY")
            if not self.api_key:
                raise ValueError("EMAIL_API_KEY not set")
    
    def load_sent_log(self) -> set:
        """Load log of emails already sent"""
        try:
            with open("automation/sent_emails.json", "r") as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
    
    def save_sent_log(self):
        """Save log of sent emails"""
        with open("automation/sent_emails.json", "w") as f:
            json.dump(list(self.sent_log), f)
    
    def load_leads_from_csv(self, filename: str) -> List[Dict]:
        """
        Load leads from CSV file
        
        Expected columns:
        - email
        - name
        - project
        - contribution_type
        - technology
        - notes
        """
        leads = []
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                leads.append(row)
        return leads
    
    def personalize_email(self, template_name: str, lead: Dict) -> Dict:
        """
        Personalize email template for a specific lead
        
        Args:
            template_name: Name of template to use
            lead: Dictionary with lead information
            
        Returns:
            Dictionary with subject and body
        """
        template = TEMPLATES[template_name]
        
        # You would customize this based on the lead's information
        # This is a simplified example
        subject = template["subject"].format(**lead)
        body = template["body"].format(**lead)
        
        return {
            "subject": subject,
            "body": body,
            "to": lead["email"]
        }
    
    def preview_email(self, email: Dict):
        """Preview email before sending"""
        print("\n" + "="*80)
        print(f"TO: {email['to']}")
        print(f"SUBJECT: {email['subject']}")
        print("-"*80)
        print(email['body'])
        print("="*80 + "\n")
    
    def send_email(self, email: Dict) -> bool:
        """
        Send email via your email service
        
        Args:
            email: Dictionary with to, subject, body
            
        Returns:
            True if sent successfully
        """
        if self.dry_run:
            self.preview_email(email)
            print("DRY RUN: Email not actually sent\n")
            return True
        
        try:
            # Example with Resend API
            # Replace with your email service
            import requests
            
            response = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": "your-email@supyagent.com",
                    "to": email["to"],
                    "subject": email["subject"],
                    "text": email["body"]
                }
            )
            
            if response.status_code == 200:
                print(f"✅ Sent email to {email['to']}")
                return True
            else:
                print(f"❌ Failed to send to {email['to']}: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending to {email['to']}: {e}")
            return False
    
    def run_campaign(self, leads: List[Dict], template_name: str, 
                     max_per_day: int = 20, delay_seconds: int = 60):
        """
        Run email outreach campaign
        
        Args:
            leads: List of lead dictionaries
            template_name: Which template to use
            max_per_day: Maximum emails to send per day
            delay_seconds: Delay between emails (rate limiting)
        """
        sent_today = 0
        
        print(f"\n🚀 Starting email campaign")
        print(f"Template: {template_name}")
        print(f"Total leads: {len(leads)}")
        print(f"Max per day: {max_per_day}")
        print(f"Dry run: {self.dry_run}\n")
        
        for lead in leads:
            # Skip if already contacted
            if lead["email"] in self.sent_log:
                print(f"⏭️  Skipping {lead['email']} (already contacted)")
                continue
            
            # Check daily limit
            if sent_today >= max_per_day:
                print(f"\n⏸️  Reached daily limit of {max_per_day} emails")
                break
            
            # Personalize email
            email = self.personalize_email(template_name, lead)
            
            # Send email
            if self.send_email(email):
                self.sent_log.add(lead["email"])
                sent_today += 1
                
                # Save progress
                self.save_sent_log()
                
                # Rate limiting
                if not self.dry_run and sent_today < max_per_day:
                    print(f"⏳ Waiting {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
        
        print(f"\n✅ Campaign complete!")
        print(f"Sent: {sent_today} emails")
        print(f"Remaining: {len([l for l in leads if l['email'] not in self.sent_log])} leads")

def create_sample_leads_csv():
    """Create a sample leads.csv file"""
    sample_leads = [
        {
            "email": "example@example.com",
            "name": "John Doe",
            "project": "langchain",
            "contribution_type": "PR #123",
            "specific_detail": "your implementation of custom tools",
            "technology": "LangChain and custom agents",
            "specific_feature": "our agent orchestration system",
            "use_case": "complex multi-agent workflows",
            "optional_tip": "I noticed you were working on agent memory. Here's a quick approach that might help: [link]",
            "sender_name": "Your Name",
            "personal_touch": "Saw you're also interested in [hobby from their profile]!"
        }
    ]
    
    with open("automation/leads_sample.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_leads[0].keys())
        writer.writeheader()
        writer.writerows(sample_leads)
    
    print("✅ Created automation/leads_sample.csv")
    print("Customize this file with your actual leads before sending!")

def main():
    """Main function with examples"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Email Outreach Helper for supyagent.com              ║
╚══════════════════════════════════════════════════════════════╝

IMPORTANT: This is a HELPER tool, not an auto-spammer!

Best practices:
1. Always run in dry_run mode first
2. Review each email before sending
3. Personalize based on recipient's work
4. Only contact people who would genuinely benefit
5. Respect daily limits (20 max recommended)
6. Follow CAN-SPAM and GDPR regulations

""")
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python email_outreach.py create-sample    # Create sample CSV")
        print("  python email_outreach.py preview leads.csv # Preview emails")
        print("  python email_outreach.py send leads.csv    # Send emails (for real)")
        return
    
    command = sys.argv[1]
    
    if command == "create-sample":
        create_sample_leads_csv()
        
    elif command == "preview":
        if len(sys.argv) < 3:
            print("Error: Please specify CSV file")
            return
        
        csv_file = sys.argv[2]
        outreach = EmailOutreach(dry_run=True)
        leads = outreach.load_leads_from_csv(csv_file)
        outreach.run_campaign(leads, "github_contributor", max_per_day=5)
        
    elif command == "send":
        if len(sys.argv) < 3:
            print("Error: Please specify CSV file")
            return
        
        csv_file = sys.argv[2]
        
        print("⚠️  WARNING: This will send REAL emails!")
        confirm = input("Are you sure? Type 'yes' to continue: ")
        
        if confirm.lower() != "yes":
            print("Cancelled.")
            return
        
        outreach = EmailOutreach(dry_run=False)
        leads = outreach.load_leads_from_csv(csv_file)
        outreach.run_campaign(leads, "github_contributor", max_per_day=20)
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
