import copy
import json
import uuid
from datetime import datetime, timezone

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from rag_engine import get_retrieval_config, query_rag

LLM_MODEL = "gpt-4o"


def get_collection_config() -> dict:
    """Stable collection-time config saved with each exported session."""
    return {
        "chat_model": LLM_MODEL,
        "retrieval": get_retrieval_config(),
    }

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
1. **Task Type** - What tasks were performed, descriptions, and progress status (started/completed?)
2. **Workforce** - Who was working: crew size, roles (workers, students, TAs, instructors), absences
3. **Materials** - What materials were used: types, specs, installation details
4. **Equipment & Tools** - What tools and machinery were used

**Safety:**
5. **Hazard** - Safety hazards observed, risks identified, incidents, PPE usage

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
        self.turn_records = []
        self.report_data = copy.deepcopy(REPORT_TEMPLATE)
        self.metrics_history = []
        self.aggregated_metrics = {}
        self.evaluation_state = {
            "status": "not_run",
            "evaluated_at": None,
            "config": None,
            "turns_evaluated": 0,
            "artifact_source": None,
        }
        self.turn_counter = 0

    @staticmethod
    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def append_message(self, role, content, *, turn_index=None, phase=None, source_count=None):
        """Store messages with enough metadata for later round reconstruction."""
        entry = {
            "message_id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": self._now_utc_iso(),
            "turn_index": turn_index,
            "phase": phase,
        }
        if source_count is not None:
            entry["source_count"] = source_count
        self.conversation_history.append(entry)
        return entry

    def begin_turn(self, user_message):
        """Create one round record before retrieval and answer generation."""
        self.turn_counter += 1
        turn_index = self.turn_counter
        user_entry = self.append_message(
            "user",
            user_message,
            turn_index=turn_index,
            phase="user_message",
        )
        turn_record = {
            "round_id": f"{self.id}-turn-{turn_index:04d}",
            "turn_index": turn_index,
            "user_message": user_message,
            "user_timestamp": user_entry["timestamp"],
            "assistant_response": None,
            "assistant_timestamp": None,
            "retrieval": {
                "query": user_message,
                "candidate_count": 0,
                "accepted_source_count": 0,
                "sources": [],
                "evaluation_snapshot": {
                    "context": "",
                    "candidates": [],
                    "relevance": [],
                },
            },
            "evaluation_status": "not_run",
            "evaluation_completed_at": None,
            "evaluation_failed_reason": None,
            "metrics": None,
            "report_completion_after_turn": None,
        }
        self.turn_records.append(turn_record)
        return turn_record

    @staticmethod
    def _serialize_retrieval_items(items):
        """Compact retrieval records saved for later offline evaluation."""
        return [
            {
                "file_name": item.get("file_name"),
                "page_label": item.get("page_label"),
                "score": item.get("score"),
                "text": item.get("text", ""),
            }
            for item in items
        ]

    def finalize_turn(self, turn_record, response, rag_result):
        """Complete one round record after answer generation.

        The saved retrieval snapshot is the contract between collection and
        later offline evaluation. Future judge reruns use this stored context,
        candidate pool, and relevance mask instead of touching live chat.
        """
        sources = rag_result.get("sources", [])
        assistant_entry = self.append_message(
            "assistant",
            response,
            turn_index=turn_record["turn_index"],
            phase="assistant_response",
            source_count=len(sources),
        )
        turn_record["assistant_response"] = response
        turn_record["assistant_timestamp"] = assistant_entry["timestamp"]
        serialized_sources = self._serialize_retrieval_items(sources)
        turn_record["retrieval"] = {
            "query": rag_result.get("query", turn_record["user_message"]),
            "candidate_count": len(rag_result.get("candidates", [])),
            "accepted_source_count": len(sources),
            "source_file_names": [source.get("file_name") for source in sources],
            "sources": serialized_sources,
            "evaluation_snapshot": {
                "context": rag_result.get("context", ""),
                "candidates": self._serialize_retrieval_items(
                    rag_result.get("candidates", [])
                ),
                # Key parameter: this relevance mask is saved so the retrieval
                # proxy can be recomputed later without rerunning live retrieval.
                "relevance": [bool(flag) for flag in rag_result.get("relevance", [])],
            },
        }
        return assistant_entry

    def chat_messages(self):
        """Project rich stored messages down to API-safe role/content items."""
        return [
            {"role": message["role"], "content": message["content"]}
            for message in self.conversation_history
        ]

    def get_report_status_text(self):
        lines = []
        for _category, fields in self.report_data.items():
            for _field_key, field in fields.items():
                filled = field["value"] is not None
                marker = "FILLED" if filled else "EMPTY"
                val = field["value"] or "Not yet collected"
                lines.append(f"[{marker}] {field['label']}: {val}")
        return "\n".join(lines)

    def get_completion_ratio(self):
        total = 0
        filled = 0
        for _category, fields in self.report_data.items():
            for _field_key, field in fields.items():
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

    def get_aggregated_metrics(self):
        return self.aggregated_metrics

    def get_ordered_turn_records(self):
        """Stable round ordering for export and later analysis."""
        return sorted(self.turn_records, key=lambda record: record["turn_index"])

    def get_evaluation_state(self):
        return dict(self.evaluation_state)


class ConversationManager:
    def __init__(self, rag_retriever):
        self.retriever = rag_retriever
        self.client = OpenAI()
        self.sessions = {}

    def create_session(self):
        session = Session()
        self.sessions[session.id] = session
        return session

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def get_collection_config(self) -> dict:
        """Expose the collection-time knobs independently from evaluator knobs."""
        return get_collection_config()

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
                f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
                for message in session.conversation_history
            ]
        )
        prompt = EXTRACTION_PROMPT.format(conversation=conv_text)

        try:
            result = self._chat([{"role": "user", "content": prompt}], temperature=0)

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
        session.append_message(
            "assistant",
            initial_msg,
            turn_index=0,
            phase="initial_prompt",
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

        turn_record = session.begin_turn(user_message)

        rag_result = query_rag(self.retriever, user_message)
        rag_result["query"] = user_message
        rag_context = rag_result["context"]
        sources = rag_result["sources"]

        system_prompt = SYSTEM_PROMPT.format(
            report_status=session.get_report_status_text(),
            rag_context=rag_context if rag_context else "No relevant documents found.",
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session.chat_messages())

        response = self._chat(messages)
        session.finalize_turn(turn_record, response, rag_result)

        self._extract_report_data(session)
        turn_record["report_completion_after_turn"] = session.get_completion_ratio()

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
            f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
            for message in session.conversation_history
        )
        prompt = SUMMARY_PROMPT.format(conversation=conv_text)
        try:
            return self._chat([{"role": "user", "content": prompt}], temperature=0.3)
        except Exception:
            return None

    def get_report_analysis(self, session_id: str) -> dict | None:
        """Generate a RAG-grounded analysis with actionable recommendations."""
        session = self.get_session(session_id)
        if not session:
            return None

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
        rag_context = (
            "\n\n---\n\n".join(labeled) if labeled else "No reference documents found."
        )

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
                    recommendations.append(
                        {
                            "category": rec.get("category") or "General",
                            "text": rec["text"],
                            "source": rec.get("source") or "",
                        }
                    )

            if not recommendations and not data.get("analysis"):
                return None

            return {
                "analysis": data.get("analysis", ""),
                "recommendations": recommendations,
            }
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    @staticmethod
    def _field_value_text(value):
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return ", ".join(ConversationManager._field_value_text(item) for item in value)
        if isinstance(value, dict):
            return "; ".join(
                f"{key}: {ConversationManager._field_value_text(item)}"
                for key, item in value.items()
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
