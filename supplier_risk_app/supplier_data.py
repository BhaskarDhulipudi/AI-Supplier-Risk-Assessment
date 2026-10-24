import csv
import os
from typing import Any, Dict, List, Optional

DATASET_FILE = os.path.join(os.path.dirname(__file__), 'supplier_risk_assessment_dataset.csv')


def _to_number(value: Any) -> Any:
    if value is None or value == '':
        return value
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def score_to_label(score: float) -> str:
    score = max(0.0, min(float(score), 100.0))
    if score >= 75:
        return 'Critical'
    if score >= 55:
        return 'High'
    if score >= 35:
        return 'Medium'
    return 'Low'


def calculate_financial_risk(supplier: Dict[str, Any]) -> int:
    if not supplier:
        return 0

    score = _safe_float(supplier.get('Financial_Risk_Score'))
    debt_to_equity = _safe_float(supplier.get('Debt_to_Equity'))
    credit_rating = str(supplier.get('Credit_Rating', '')).upper()

    if debt_to_equity > 2.5:
        score += 15
    if debt_to_equity > 4.0:
        score += 10
    if credit_rating in {'CCC', 'CC', 'C', 'D'}:
        score += 20
    elif credit_rating in {'B', 'B-', 'B+'}:
        score += 10
    elif credit_rating in {'BB', 'BBB'}:
        score += 5

    score = max(0.0, min(score, 100.0))
    return round(score)


def calculate_product_risk(supplier: Dict[str, Any]) -> int:
    if not supplier:
        return 0

    operational = _safe_float(supplier.get('Operational_Risk_Score'))
    delivery = _safe_float(supplier.get('Delivery_Risk_Score'))
    dependency = _safe_float(supplier.get('Supplier_Dependency_Percent'))
    cybersecurity = _safe_float(supplier.get('Cybersecurity_Risk_Score'))

    score = (operational + delivery + dependency + cybersecurity) / 4.0
    score = max(0.0, min(score, 100.0))
    return round(score)


def calculate_quality_risk(supplier: Dict[str, Any]) -> int:
    if not supplier:
        return 0

    quality = _safe_float(supplier.get('Quality_Risk_Score'))
    compliance = _safe_float(supplier.get('Compliance_Risk_Score'))
    defect_rate = _safe_float(supplier.get('Defect_Rate_Percent'))
    cybersecurity = _safe_float(supplier.get('Cybersecurity_Risk_Score'))

    defect_risk = min(defect_rate * 2.0, 100.0)
    score = (quality + compliance + defect_risk + cybersecurity) / 4.0
    score = max(0.0, min(score, 100.0))
    return round(score)


def calculate_overall_risk(financial_score: float, product_score: float, quality_score: float) -> int:
    score = (financial_score * 0.40) + (product_score * 0.30) + (quality_score * 0.30)
    score = max(0.0, min(score, 100.0))
    return round(score)


def calculate_risk_breakdown(supplier: Dict[str, Any]) -> Dict[str, Any]:
    cached_breakdown = supplier.get('_risk_breakdown')
    if isinstance(cached_breakdown, dict):
        return dict(cached_breakdown)

    financial_score = calculate_financial_risk(supplier)
    product_score = calculate_product_risk(supplier)
    quality_score = calculate_quality_risk(supplier)
    overall_score = calculate_overall_risk(financial_score, product_score, quality_score)

    return {
        'financial_risk_score': financial_score,
        'financial_risk_level': score_to_label(financial_score),
        'product_risk_score': product_score,
        'product_risk_level': score_to_label(product_score),
        'quality_risk_score': quality_score,
        'quality_risk_level': score_to_label(quality_score),
        'overall_risk_score': overall_score,
        'overall_risk_level': score_to_label(overall_score),
    }


def load_suppliers() -> List[Dict[str, Any]]:
    if not os.path.exists(DATASET_FILE):
        return []
    with open(DATASET_FILE, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key not in {'Supplier_ID', 'Supplier_Name', 'Industry', 'Country', 'Credit_Rating', 'Risk_Level', 'Risk_Reason', 'Recommended_Action'}:
                row[key] = _to_number(value)

        breakdown = calculate_risk_breakdown(row)
        row['_risk_breakdown'] = breakdown
        row['Financial_Risk_Score'] = breakdown['financial_risk_score']
        row['Financial_Risk_Level'] = breakdown['financial_risk_level']
        row['Product_Risk_Score'] = breakdown['product_risk_score']
        row['Product_Risk_Level'] = breakdown['product_risk_level']
        row['Quality_Risk_Score'] = breakdown['quality_risk_score']
        row['Quality_Risk_Level'] = breakdown['quality_risk_level']
        row['Overall_Risk_Score'] = breakdown['overall_risk_score']
        row['Overall_Risk_Level'] = breakdown['overall_risk_level']
        row['Risk_Level'] = breakdown['overall_risk_level']

    return rows


def find_supplier(identifier: str) -> Optional[Dict[str, Any]]:
    identifier = (identifier or '').strip().lower()
    if not identifier:
        return None
    suppliers = load_suppliers()
    exact = next((r for r in suppliers if str(r.get('Supplier_ID', '')).lower() == identifier), None)
    if exact:
        return exact
    exact = next((r for r in suppliers if str(r.get('Supplier_Name', '')).lower() == identifier), None)
    if exact:
        return exact
    matches = [r for r in suppliers if identifier in str(r.get('Supplier_Name', '')).lower()]
    return matches[0] if matches else None


def risk_level(score: float) -> str:
    if score >= 75:
        return 'Critical'
    if score >= 55:
        return 'High'
    if score >= 35:
        return 'Medium'
    return 'Low'


def supplier_profile_text(supplier: Dict[str, Any]) -> str:
    if not supplier:
        return 'No structured supplier record was found in the CSV dataset.'
    labels = {
        'Supplier_ID': 'Supplier ID', 'Supplier_Name': 'Supplier Name', 'Industry': 'Industry',
        'Country': 'Country', 'Annual_Revenue_USD': 'Annual Revenue (USD)', 'Debt_to_Equity': 'Debt to Equity',
        'Credit_Rating': 'Credit Rating', 'On_Time_Delivery_Rate_Percent': 'On-Time Delivery (%)',
        'Defect_Rate_Percent': 'Defect Rate (%)', 'Average_Lead_Time_Days': 'Average Lead Time (days)',
        'Contract_Compliance_Percent': 'Contract Compliance (%)', 'Cybersecurity_Score': 'Cybersecurity Score',
        'ESG_Score': 'ESG Score', 'Geopolitical_Risk_Score': 'Geopolitical Risk Score',
        'Supplier_Dependency_Percent': 'Supplier Dependency (%)', 'Financial_Risk_Score': 'Financial Risk Score',
        'Operational_Risk_Score': 'Operational Risk Score', 'Quality_Risk_Score': 'Quality Risk Score',
        'Delivery_Risk_Score': 'Delivery Risk Score', 'Cybersecurity_Risk_Score': 'Cybersecurity Risk Score',
        'Compliance_Risk_Score': 'Compliance Risk Score', 'ESG_Risk_Score': 'ESG Risk Score',
        'Overall_Risk_Score': 'Overall Risk Score', 'Risk_Level': 'Dataset Risk Level',
        'Risk_Reason': 'Dataset Risk Reason', 'Recommended_Action': 'Dataset Recommended Action'
    }
    lines = []
    for key, label in labels.items():
        if key in supplier:
            lines.append(f'{label}: {supplier[key]}')
    return '\n'.join(lines)
