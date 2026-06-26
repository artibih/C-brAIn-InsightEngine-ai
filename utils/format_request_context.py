import json

def build_full_context(context):
    if not context:
        return {}
    events = context.get("events", [])

    parsed_events = []

    for e in events:
        if isinstance(e, dict):
            parsed_events.append(e)
            continue

        if isinstance(e, str):
            if e.startswith("data:"):
                e = e.replace("data: ", "")

            try:
                parsed_events.append(json.loads(e))
            except Exception:
                continue 

    context = {
        "plan": {},
        "retrieval": [],
        "statistics": [],
        "synthesis": {},
        "critique": {}
    }

    for e in parsed_events:
        agent = e.get("agent")

        if agent == "hypothesis_planner":
            context["plan"] = {
                "objective": e.get("objective"),
                "analysis_steps": e.get("analysis_steps", [])
            }

        elif agent == "statistical_executor":
            context["statistics"].append(e.get("structured_results"))

        elif agent == "synthesizer":
            context["synthesis"] = {
                "findings": e.get("findings", [])
            }

        elif agent == "critique_agent":
            context["critique"] = {
                "issues": e.get("issues", []),
                "revision_instructions": e.get("revision_instructions", [])
            }

    return context

def normalize_previous_reviews(prev):
    if not prev:
        return None

    result = {}

    if "experiment_id" in prev:
        result["experiment_id"] = prev["experiment_id"]

    if isinstance(prev, list):
        items = prev
    else:
        items = prev.get("events", [])

    for item in items:

        if "data" in item:
            data = item["data"]

            for key in [
                "review_journal_editor",
                "review_domain_expert",
                "review_methodological",
                "review_benchmark_evidence",
            ]:
                if key in data:
                    result[key] = data[key].get("output", {})

            if "final_review" in data:
                result["final_review"] = data["final_review"]

        elif "agent" in item:
            agent = item.get("agent", "").replace("reviewer_", "review_")
            data = item.get("data", {})
            if agent in data:
                result[agent] = data[agent].get("output", {})
            if "final_review" in data:
                result["final_review"] = data["final_review"]

    return result if result else None