import json
import os
import uuid
import copy
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from rag_engine import query_rag

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
LLM_MODEL = os.environ.get("AZURE_OPENAI_LLM_MODEL", "gpt-4o")

REPORT_TEMPLATE = {
    "activity": {
        "task_type": {
            "label": "Task Type",
            "description": "The specific task being performed (e.g., framing, concrete pour, scaffolding), task descriptions and progress status (start/end).",
            "value": None,
        },
        "workforce": {
            "label": "Workforce",
            "description": "Crew composition: headcount, roles (workers, students, TAs, instructors), and any absent personnel.",
            "value": None,
        },
        "materials": {
            "label": "Materials",
            "description": "Material types, specifications, and installation methods (e.g., 2x4 lumber, plywood, nails, 16\" spacing).",
            "value": None,
        },
        "equipment_and_tools": {
            "label": "Equipment & Tools",
            "description": "Hand tools (hammers, squares) and power tools (table saw, circular saw, miter box) used on site.",
            "value": None,
        },
    },
    "safety": {
        "hazard": {
            "label": "Hazard",
            "description": "Safety hazards or risks observed, potential injury sources, incidents, and PPE usage (e.g., pinch points, saw blade hazards, gloves).",
            "value": None,
        },
    },
}

SYSTEM_PROMPT = """\
You are a friendly and professional Daily Report Assistant for construction sites and educational building labs. \
Your job is to help workers and students create their daily activity report through natural conversation.

## Report Sections to Fill

**Activity:**
1. **Task Type** – What tasks were performed, descriptions, and progress status (started/completed?)
2. **Workforce** – Who was working: crew size, roles (workers, students, TAs, instructors), absences
3. **Materials** – What materials were used: types, specs, installation details
4. **Equipment & Tools** – What tools and machinery were used

**Safety:**
5. **Hazard** – Safety hazards observed, risks identified, incidents, PPE usage

## Conversation Guidelines

- Be conversational and natural. Do not list all questions at once.
- Start by asking about today's main activity/task.
- Follow up naturally based on their responses, one topic at a time.
- When safety-related topics come up, weave in relevant educational information from the provided reference documents.
- After covering one section, move on to the next unfilled one.
- Acknowledge their input before asking the next question.
- If they provide info for multiple fields at once, acknowledge all of it.
- When all or most fields are filled, summarize what you have and let the user know they can download the report.
- Keep responses concise (2-4 sentences usually).
- Prefer plain sentences and avoid em dashes.
- Avoid bullet-heavy style in chat replies.

## Current Report Status
{report_status}

## Reference Documents (for safety/construction education)
{rag_context}
"""

EXTRACTION_PROMPT = """\
Extract daily report information from this conversation. Return a JSON object with these fields.
Use null for any field not mentioned by the user. Only extract what the USER explicitly stated.

Fields:
- task_type: Task descriptions and progress
- workforce: Crew size, roles, absences
- materials: Materials used with specs
- equipment_and_tools: Tools and machinery used
- hazard: Safety hazards, risks, incidents, PPE

Conversation:
{conversation}

Return ONLY valid JSON, nothing else:"""

SUMMARY_PROMPT = """\
Based on this conversation about a day's work, write a short explanation in 2 to 3 sentences.
Describe what was done and any notable points (tasks, who was there, tools, safety). \
Do not give definitions. Write only from what the user said. Use plain English.

Conversation:
{conversation}

Short explanation:"""

ANALYSIS_PROMPT = """\
You are a construction safety and quality advisor reviewing a daily activity \
report from a construction site or educational building lab. Using the report \
details and the reference documents (building codes, OSHA guidance) below, \
produce a brief, practical analysis with actionable recommendations.

Guidelines:
- Ground every recommendation in the report content or the reference documents. Do not invent facts or cite codes that are not in the references.
- When a recommendation is supported by a reference document, put that document's name in "source". Otherwise use an empty string.
- Be specific and practical. Focus on safety, code compliance, and work quality.
- If important information is missing from the report (e.g., crew size, PPE), you may recommend recording or addressing it.
- Provide 3 to 5 recommendations.

Report details:
{report_status}

Reference documents:
{rag_context}

Return ONLY valid JSON in this exact shape, nothing else:
{{
  "analysis": "2-3 sentence overall assessment of the day's work and safety posture",
  "recommendations": [
    {{"category": "Safety", "text": "specific actionable recommendation", "source": "document name or empty string"}}
  ]
}}
Allowed categories: Safety, Code Compliance, Quality, Documentation."""


class Session:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.conversation_history = []
        self.report_data = copy.deepcopy(REPORT_TEMPLATE)

    def get_report_status_text(self):
        lines = []
        for category, fields in self.report_data.items():
            for field_key, field in fields.items():
                filled = field["value"] is not None
                marker = "FILLED" if filled else "EMPTY"
                val = field["value"] or "Not yet collected"
                lines.append(f"[{marker}] {field['label']}: {val}")
        return "\n".join(lines)

    def get_completion_ratio(self):
        total = 0
        filled = 0
        for category, fields in self.report_data.items():
            for field_key, field in fields.items():
                total += 1
                if field["value"]:
                    filled += 1
        return filled / total if total > 0 else 0

    def get_report_summary(self):
        summary = {}
        for category, fields in self.report_data.items():
            summary[category] = {}
            for field_key, field in fields.items():
                summary[category][field_key] = {
                    "label": field["label"],
                    "value": field["value"],
                    "filled": field["value"] is not None,
                }
        return summary


class ConversationManager:
    def __init__(self, rag_retriever):
        self.retriever = rag_retriever
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
        )
        self.sessions = {}

    def create_session(self):
        session = Session()
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _chat(self, messages, temperature=0.7):
        completion = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content

    def _extract_report_data(self, session):
        conv_text = "\n".join(
            [
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in session.conversation_history
            ]
        )
        prompt = EXTRACTION_PROMPT.format(conversation=conv_text)

        try:
            result = self._chat(
                [{"role": "user", "content": prompt}], temperature=0
            )

            json_str = result.strip()
            if "```" in json_str:
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            data = json.loads(json_str)

            field_mapping = {
                "task_type": ("activity", "task_type"),
                "workforce": ("activity", "workforce"),
                "materials": ("activity", "materials"),
                "equipment_and_tools": ("activity", "equipment_and_tools"),
                "hazard": ("safety", "hazard"),
            }

            for field_key, (category, key) in field_mapping.items():
                if field_key in data and data[field_key]:
                    session.report_data[category][key]["value"] = data[field_key]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    def get_initial_message(self, session_id):
        session = self.get_session(session_id)
        if not session:
            return None

        initial_msg = (
            "Hi there! I'm your Daily Report Assistant. I'll help you put together "
            "today's daily activity report through a quick conversation.\n\n"
            "Let's get started. What task or activity did you work on today?"
        )
        session.conversation_history.append(
            {"role": "assistant", "content": initial_msg}
        )

        return {
            "response": initial_msg,
            "report_data": session.get_report_summary(),
            "completion": 0,
        }

    def process_message(self, session_id, user_message):
        session = self.get_session(session_id)
        if not session:
            return None

        session.conversation_history.append({"role": "user", "content": user_message})

        rag_result = query_rag(self.retriever, user_message)
        rag_context = rag_result["context"]
        sources = rag_result["sources"]

        system_prompt = SYSTEM_PROMPT.format(
            report_status=session.get_report_status_text(),
            rag_context=rag_context if rag_context else "No relevant documents found.",
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session.conversation_history)

        response = self._chat(messages)

        session.conversation_history.append(
            {"role": "assistant", "content": response}
        )

        self._extract_report_data(session)

        return {
            "response": response,
            "report_data": session.get_report_summary(),
            "completion": session.get_completion_ratio(),
            "sources": sources,
        }

    def get_conversation_summary(self, session_id: str) -> str | None:
        session = self.get_session(session_id)
        if not session or not session.conversation_history:
            return None
        conv_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in session.conversation_history
        )
        prompt = SUMMARY_PROMPT.format(conversation=conv_text)
        try:
            return self._chat([{"role": "user", "content": prompt}], temperature=0.3)
        except Exception:
            return None

    def get_report_analysis(self, session_id: str) -> dict | None:
        """Generate a RAG-grounded analysis with actionable recommendations.

        Retrieves relevant code/safety passages based on the collected report,
        then asks the LLM to produce a short assessment plus recommendations
        that cite the reference documents.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        # Build a retrieval query from the filled report fields.
        filled_parts = []
        for _category, fields in session.report_data.items():
            for _key, field in fields.items():
                if field["value"]:
                    val = self._field_value_text(field["value"])
                    filled_parts.append(f"{field['label']}: {val}")
        if not filled_parts:
            return None

        query = " ".join(filled_parts)
        rag_result = query_rag(self.retriever, query)

        labeled = []
        for src in rag_result.get("sources", []):
            name = self._clean_source_name(src.get("file_name"))
            page = src.get("page_label")
            header = f"[Source: {name}{', p.' + str(page) if page else ''}]"
            labeled.append(f"{header}\n{src.get('text', '')}")
        rag_context = "\n\n---\n\n".join(labeled) if labeled else "No reference documents found."

        prompt = ANALYSIS_PROMPT.format(
            report_status=session.get_report_status_text(),
            rag_context=rag_context,
        )

        try:
            result = self._chat([{"role": "user", "content": prompt}], temperature=0.2)
            json_str = result.strip()
            if "```" in json_str:
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None

            recommendations = []
            for rec in data.get("recommendations", []):
                if isinstance(rec, dict) and rec.get("text"):
                    recommendations.append({
                        "category": rec.get("category") or "General",
                        "text": rec["text"],
                        "source": rec.get("source") or "",
                    })

            if not recommendations and not data.get("analysis"):
                return None

            return {
                "analysis": data.get("analysis", ""),
                "recommendations": recommendations,
            }
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    @staticmethod
    def _field_value_text(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return ", ".join(ConversationManager._field_value_text(v) for v in value)
        if isinstance(value, dict):
            return "; ".join(
                f"{k}: {ConversationManager._field_value_text(v)}" for k, v in value.items()
            )
        return str(value)

    @staticmethod
    def _clean_source_name(file_name) -> str:
        if not file_name:
            return "Reference"
        name = str(file_name)
        if "." in name:
            name = name.rsplit(".", 1)[0]
        return name.replace("_", " ")
