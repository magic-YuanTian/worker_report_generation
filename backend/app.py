import io
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from flask import Flask, request, jsonify, send_file  # noqa: E402
from flask_cors import CORS  # noqa: E402
from rag_engine import init_rag  # noqa: E402
from conversation_manager import ConversationManager  # noqa: E402
from report_generator import generate_report_pdf  # noqa: E402

app = Flask(__name__)
CORS(app)

print("Starting server — initializing RAG engine (this may take a minute on first run)...")
retriever = init_rag()
manager = ConversationManager(retriever)
print("Server ready.")


@app.route("/api/session", methods=["POST"])
def new_session():
    session = manager.create_session()
    result = manager.get_initial_message(session.id)
    return jsonify({"session_id": session.id, **result})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    message = data.get("message", "").strip()

    if not session_id or not message:
        return jsonify({"error": "session_id and message are required"}), 400

    result = manager.process_message(session_id, message)
    if result is None:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(result)


@app.route("/api/report/<session_id>", methods=["GET"])
def get_report(session_id):
    session = manager.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(
        {
            "report_data": session.get_report_summary(),
            "completion": session.get_completion_ratio(),
        }
    )


@app.route("/api/report/<session_id>/download", methods=["GET"])
def download_report(session_id):
    session = manager.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    explanation = manager.get_conversation_summary(session_id)
    analysis = manager.get_report_analysis(session_id)
    pdf_bytes = generate_report_pdf(
        session.report_data, session_id, explanation=explanation, analysis=analysis
    )

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"daily_report_{session_id[:8]}.pdf",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
