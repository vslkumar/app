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
