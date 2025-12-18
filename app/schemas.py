from typing import List, Dict, Optional

from pydantic import BaseModel


class ChartData(BaseModel):
    id: str
    type: str  # 'histogram', 'bar', 'line', 'heatmap'
    title: str
    description: str
    data: Dict  # generic payload that frontend can map into Chart.js/Recharts


class AnalysisSummary(BaseModel):
    dataset_shape: Dict[str, int]
    column_overview: Dict[str, str]
    missing_summary: Dict[str, int]
    detected_roles: Dict[str, List[str]]  # e.g. numeric/categorical/date lists


class InsightBlock(BaseModel):
    title: str
    detail: str


class RecommendationBlock(BaseModel):
    title: str
    action: str


class AnalysisResult(BaseModel):
    summary: AnalysisSummary
    summary_statistics: Dict[str, Dict]
    correlations: Dict[str, float]
    key_insights: List[InsightBlock]
    recommendations: List[RecommendationBlock]
    charts: List[ChartData]
    cleaned_csv_base64: str  # cleaned dataset as base64-encoded CSV
    insight_text: str  # full plain-text summary for download


class ErrorResponse(BaseModel):
    detail: str


