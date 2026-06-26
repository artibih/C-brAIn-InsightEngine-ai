from typing import Dict, Any, List
from collections import Counter
import structlog
from config.settings import settings

logger = structlog.get_logger(__name__)

class NLIConsensusArbiter:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.n_samples = getattr(settings, "number_of_hallucination_detectors", 3)

    def arbitrate(self, hallucination_samples: List[Dict[str, Any]], official_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not official_claims:
            return {}

        official_cids = {str(c.get("claim_id")): c for c in official_claims}
        claims_consensus = {cid: [] for cid in official_cids.keys()}

        for sample in hallucination_samples:
            seen_cids_in_this_sample = set() 
            
            for verdict_block in sample.get("verdicts", []):
                cid = str(verdict_block.get("claim_id", ""))
                verdict = verdict_block.get("verdict")
              
                if not verdict or not isinstance(verdict, str):
                    continue
          
                if cid in claims_consensus and cid not in seen_cids_in_this_sample:
                    claims_consensus[cid].append(verdict_block)
                    seen_cids_in_this_sample.add(cid)

        final_verdicts = []

        for cid, votes in claims_consensus.items():
            if not votes:
                final_verdicts.append({
                    "claim_id": cid,
                    "claim_text": official_cids[cid].get("claim_text", "Text unavailable"),
                    "matched_evidence": "Not Found",
                    "reasoning": "System Warning: All parallel agents failed to evaluate this claim.",
                    "verdict": "neutral",
                    "confidence_score": 0.0,
                    "revision_instruction": "None"
                })
                continue

            verdict_strings = [v.get("verdict").lower() for v in votes]
            counts = Counter(verdict_strings)
            top_candidates = counts.most_common()
            
            is_tie = len(top_candidates) > 1 and (top_candidates[0][1] == top_candidates[1][1])
            
            if is_tie:
                tied_verdicts = [v[0] for v in top_candidates if v[1] == top_candidates[0][1]]
                if "contradicted" in tied_verdicts:
                    consensus_verdict = "contradicted"
                elif "neutral" in tied_verdicts:
                    consensus_verdict = "neutral"
                else:
                    consensus_verdict = top_candidates[0][0]
            else:
                consensus_verdict = top_candidates[0][0]

            confidence_score = round(counts[consensus_verdict] / self.n_samples, 2)

            winning_vote = next(v for v in votes if v.get("verdict").lower() == consensus_verdict)

            final_verdicts.append({
                "claim_id": cid,
                "claim_text": winning_vote.get("claim_text") or official_cids[cid].get("claim_text", ""),
                "matched_evidence": winning_vote.get("matched_evidence", ""),
                "reasoning": winning_vote.get("reasoning", ""),
                "verdict": consensus_verdict,
                "confidence_score": confidence_score,
                "revision_instruction": winning_vote.get("revision_instruction")
            })

        total = len(final_verdicts)
        entailed = sum(1 for v in final_verdicts if v["verdict"] == "entailed")
        contradicted = sum(1 for v in final_verdicts if v["verdict"] == "contradicted")
        neutral = sum(1 for v in final_verdicts if v["verdict"] == "neutral")
        
        risk = (contradicted * 1.0 + neutral * 0.5) / total if total > 0 else 0.0

        revision_instructions = []
        for v in final_verdicts:
            instruction = v.get("revision_instruction")
            if instruction and str(instruction).strip().lower() not in ["null", "none", ""]:
                formatted = (
                    f"Claim [{v['claim_id']}]: \"{v['claim_text']}\" "
                    f"FAILED ({v['verdict'].upper()} - Confidence: {v['confidence_score']}).\n"
                    f"Reasoning: {v['reasoning']}\n"
                    f"Action Required: {instruction}"
                )
                revision_instructions.append(formatted)

        logger.info(
            "nli_consensus_achieved",
            experiment_id=self.experiment_id,
            total_claims=total,
            entailed=entailed,
            hallucination_risk_score=round(risk, 3)
        )

        return {
            "verdicts": final_verdicts,
            "summary": {
                "total_claims": total,
                "entailed": entailed,
                "contradicted": contradicted,
                "neutral": neutral,
                "hallucination_risk_score": round(risk, 3),
            },
            "needs_revision": (total > 0 and entailed < total),
            "revision_instructions": revision_instructions
        }