from __future__ import annotations
import json, os, requests
from dataclasses import dataclass

@dataclass
class Ticket:
    """Данные тикета от пользователя."""
    name: str
    email: str
    title: str
    description: str

class Ticketing:
    """Jira/Trello/GitHub с локальным fallback (tickets.jsonl)."""
    def __init__(self) -> None:
        pass

    def create(self, t: Ticket) -> str:
        """Создаёт тикет и возвращает человеко‑читаемый идентификатор/ссылку."""
        try:
            if os.environ.get("JIRA_BASE_URL"):
                return self._create_jira(t)
        except Exception as e:
            print("Jira create failed:", e)
        try:
            if os.environ.get("TRELLO_KEY"):
                return self._create_trello(t)
        except Exception as e:
            print("Trello create failed:", e)
        try:
            if os.environ.get("GITHUB_TOKEN"):
                return self._create_github(t)
        except Exception as e:
            print("GitHub create failed:", e)
        return self._create_local(t)

    def _create_jira(self, t: Ticket) -> str:
        base = os.environ["JIRA_BASE_URL"].rstrip("/")
        email = os.environ["JIRA_EMAIL"]
        token = os.environ["JIRA_API_TOKEN"]
        project = os.environ["JIRA_PROJECT_KEY"]
        url = f"{base}/rest/api/3/issue"
        payload = {
            "fields": {
                "project": {"key": project},
                "summary": t.title,
                "description": f"Reporter: {t.name} <{t.email}>\n\n{t.description}",
                "issuetype": {"name": "Task"}
            }
        }
        resp = requests.post(url, json=payload, auth=(email, token),
                             headers={"Accept": "application/json", "Content-Type": "application/json"})
        resp.raise_for_status()
        key = resp.json().get("key", "JIRA-?")
        return f"Jira issue {key}"

    def _create_trello(self, t: Ticket) -> str:
        key = os.environ["TRELLO_KEY"]
        token = os.environ["TRELLO_TOKEN"]
        list_id = os.environ["TRELLO_LIST_ID"]
        url = "https://api.trello.com/1/cards"
        params = {
            "idList": list_id,
            "name": t.title,
            "desc": f"Reporter: {t.name} <{t.email}>\n\n{t.description}",
            "key": key,
            "token": token,
        }
        resp = requests.post(url, params=params)
        resp.raise_for_status()
        card = resp.json()
        return f"Trello card {card.get('shortLink')}"


    def _create_github(self, t: Ticket) -> str:
        repo = os.environ["GITHUB_REPO"]
        token = os.environ["GITHUB_TOKEN"]
        url = f"https://api.github.com/repos/{repo}/issues"
        payload = {
            "title": t.title,
            "body": f"Reporter: {t.name} <{t.email}>\n\n{t.description}"
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        issue = resp.json()
        # Вернём и номер, и URL — Streamlit покажет как текст, кликнуть можно по URL
        number = issue.get("number")
        html_url = issue.get("html_url") or f"https://github.com/{repo}/issues/{number}"
        return f"GitHub issue #{number}: {html_url}"

    def _create_local(self, t: Ticket) -> str:
        os.makedirs("tickets", exist_ok=True)
        path = os.path.join("tickets", "tickets.jsonl")
        rec = {"name": t.name, "email": t.email, "title": t.title, "description": t.description}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return "local-ticket"
