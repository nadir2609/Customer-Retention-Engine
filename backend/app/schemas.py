from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    tenure: int = Field(ge=1, le=60)
    usage_frequency: int = Field(ge=1, le=30)
    support_calls: int = Field(ge=0, le=10)
    payment_delay: int = Field(ge=0, le=30)
    subscription_type: Literal["Basic", "Standard", "Premium"]
    contract_length: Literal["Monthly", "Quarterly", "Annual"]
    total_spend: float = Field(ge=100, le=1000)
    last_interaction: int = Field(ge=1, le=30)
    age_group: Literal["Child", "Youth", "Adult", "Senior"]

    def to_model_row(self) -> dict[str, Any]:
        return {
            "Gender": self.gender,
            "Tenure": self.tenure,
            "Usage Frequency": self.usage_frequency,
            "Support Calls": self.support_calls,
            "Payment Delay": self.payment_delay,
            "Subscription Type": self.subscription_type,
            "Contract Length": self.contract_length,
            "Total Spend": self.total_spend,
            "Last Interaction": self.last_interaction,
            "AgeGroup": self.age_group,
        }


class RecommendationPayload(BaseModel):
    status: Literal["NO_ACTION", "SINGLE_CHANGE", "MULTI_CHANGE", "HIGH_RISK"]
    message: str
    churn_probability: float | None = None
    original_churn_prob: float | None = None
    new_churn_prob: float | None = None
    recommendations: dict[str, dict[str, Any]] | None = None


class PredictionResponse(BaseModel):
    prediction: Literal["Churn", "Not Churn"]
    churn_probability: float
    retention_probability: float
    recommendation: RecommendationPayload


class MetadataResponse(BaseModel):
    model_name: str
    feature_columns: list[str]
    categories: dict[str, list[str]]
    numeric_ranges: dict[str, dict[str, float]]
