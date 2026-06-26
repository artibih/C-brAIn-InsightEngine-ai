"""Universal schema for the Knowledge Graph: experiments, claims, and extraction output."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Optional
import hashlib



class ResultTrend(str, Enum):
    """Direction of the effect observed (trend)."""
    INCREASED = "Increased"
    DECREASED = "Decreased"
    NO_CHANGE = "No Change"
    INCONCLUSIVE = "Inconclusive"



class Method(BaseModel):
    """Details about the technique used."""
    id: Optional[str] = Field(None, description="Deterministic ID; if omitted, generated from name+parameters")
    name: str = Field(..., description="Specific name of the method, e.g., 'Western Blot', 'Morris Water Maze'")
    parameters: Optional[str] = Field(None, description="Key settings, e.g., 'Antibody 6E10 1:1000'")

    def get_stable_id(self) -> str:
        """Deterministic id for Neo4j merge."""
        raw = f"{self.name}|{self.parameters or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def stable_id(self) -> str:
        return self.id or self.get_stable_id()


SPECIES_VALUES = ("Human", "Mouse", "Rat", "In Vitro", "Other")


class Cohort(BaseModel):
    """Details about the biological sample or population."""
    id: Optional[str] = Field(None, description="Deterministic ID; if omitted, generated from key fields")
    group_name: str = Field(..., description="Name of the group, e.g., '5xFAD Mice', 'AD Patients'")
    species: Literal["Human", "Mouse", "Rat", "In Vitro", "Other"] = Field(..., description="Biological origin")
    characteristics: Optional[str] = Field(None, description="Key traits, e.g., 'Age 6 months', 'APOE4 carrier'")
    sample_size: Optional[int] = Field(None, description="The 'N' number if mentioned")

    @field_validator("species", mode="before")
    @classmethod
    def coerce_species(cls, v: object) -> str:
        """Coerce LLM outputs like 'Human, Mouse' to a single allowed species (first match wins)."""
        if isinstance(v, str) and v in SPECIES_VALUES:
            return v
        if isinstance(v, str):
            s = v.strip()
            for part in s.split(","):
                candidate = part.strip()
                if candidate in SPECIES_VALUES:
                    return candidate
            for allowed in SPECIES_VALUES:
                if allowed.lower() in s.lower():
                    return allowed
        return "Other"

    def get_stable_id(self) -> str:
        """Deterministic id for Neo4j merge."""
        raw = f"{self.group_name}|{self.species}|{self.characteristics or ''}|{self.sample_size or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def stable_id(self) -> str:
        return self.id or self.get_stable_id()


class Result(BaseModel):
    """
    The immediate outcome of this specific experiment.
    Significance (p-value) is separate from direction (trend).
    """
    id: Optional[str] = Field(None, description="Set to experiment_id + '_result' when nested in Experiment")
    description: str = Field(..., description="Short text summary of the finding")
    p_value: Optional[str] = Field(
        None,
        description="Statistical significance as reported, e.g., 'p < 0.05', 'p = 0.03', 'ns', 'not significant'"
    )
    trend: ResultTrend = Field(
        ...,
        description="Direction of the effect: Increased, Decreased, No Change, or Inconclusive"
    )

    @field_validator("trend", mode="before")
    @classmethod
    def coerce_trend(cls, v: object) -> ResultTrend:
        """Coerce invalid trend values (e.g. LLM returning 'Proven') to Inconclusive."""
        if isinstance(v, ResultTrend):
            return v
        if isinstance(v, str):
            try:
                return ResultTrend(v)
            except ValueError:
                pass
            return ResultTrend.INCONCLUSIVE
        return ResultTrend.INCONCLUSIVE

    def get_stable_id(self, experiment_id: str) -> str:
        """Deterministic id scoped to the parent experiment."""
        return f"{experiment_id}_result"

class Experiment(BaseModel):
    """
    A single experimental unit: Method, Cohort, and Result as one atomic cluster.
    """
    experiment_id: str = Field(..., description="Unique ID for this experiment in the paper, e.g., 'Exp_1', 'Fig_1A'")
    method: Method = Field(..., description="The tool used")
    cohort: Cohort = Field(..., description="The subject tested")
    result: Result = Field(..., description="The outcome yielded")

    def result_stable_id(self) -> str:
        return self.result.get_stable_id(self.experiment_id)

    @model_validator(mode="after")
    def _set_result_id(self):
        if self.result.id is None:
            self.result.id = self.result.get_stable_id(self.experiment_id)
        return self

class Claim(BaseModel):
    """A high-level scientific hypothesis or assertion made by the authors."""
    claim_id: str = Field(..., description="Unique ID, e.g., 'Claim_1'")
    text: str = Field(..., description="The exact scientific claim as written by the authors.")
    status: Literal["Hypothesized", "Proven", "Refuted"] = Field(
        ...,
        description="How the authors present this claim"
    )

class ExtractedContent(BaseModel):
    """Full output from the Extraction Agent: atomic experiments and global claims."""
    experiments: List[Experiment] = Field(..., description="All experimental units found")
    claims: List[Claim] = Field(..., description="Scientific claims or hypotheses tested")
