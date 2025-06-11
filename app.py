
import random
from app.services.email_client import get_mock_emails_by_date

statuses = ["To Do", "In Progress", "Done"]
key_actions_templates = [
    "Initial analysis is pending.",
    "Engineering team is actively working on this item.",
    "Issue completed and deployed to production.",
    "Currently under review by QA team.",
    "Blocked by dependency on another team."
]
assignees = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Martin", "TBD"]
reporters = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Martin", "TBD"]

def summarize_text(jira_ids: list) -> list:
    results = []
    for jira_id in jira_ids:
        story = f"""### {jira_id}

**Summary:** This is a mocked summary for {jira_id}.  
**Status:** {random.choice(statuses)}  
**Key Actions:** {random.choice(key_actions_templates)}  
**Assignee:** {random.choice(assignees)}  
**Reporter:** {random.choice(reporters)}  

---
"""
        results.append(story)
    return results

def extract_meeting_details(text: str) -> dict:
    return {
        "attendees": "John Doe, Jane Smith",
        "title": "Team Sync Meeting",
        "date": "2025-06-13",
        "time": "10:00 AM",
        "duration": "60 minutes"
    }

def summarize_emails_by_date(date: str, user: str = "vishal.kr@example.com") -> list:
    emails = get_mock_emails_by_date(date)

    categories = {
        "@mentioned": [],
        "Important": [],
        "Prod P1": []
    }
    todo_list = []

    for email in emails:
        cat = email.get("category", "Uncategorized")
        if cat in categories:
            categories[cat].append(email)
        if cat in ["Important", "Prod P1"]:
            todo_list.append(f"Follow up on '{email['subject']}'")

    story_parts = []

     # Greeting
    story_parts.append(f"👋 Hello, {user}!\nToday’s email summary for **{date}**:\n---\n")

    for cat, emails_in_cat in categories.items():
        if emails_in_cat:
            story_parts.append(f"### {cat} Emails\n\n")
            for email in emails_in_cat:
                story_parts.append(f"- **{email['subject']}**\n  \"{email['summary']}\"\n")
            story_parts.append("---\n")

    # TODO list
    if todo_list:
        story_parts.append("### TODO List for today\n")
        for task in todo_list:
            story_parts.append(f"- {task}\n")
        story_parts.append("---\n")

    return story_parts
