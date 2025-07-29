import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_bytes
import re
import spacy
from spacy.matcher import Matcher
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
from PIL import Image
import base64
import joblib
from typing import Dict, List, Set, Any  # ADDED IMPORT

# Set Tesseract path - UPDATE THIS TO YOUR ACTUAL TESSERACT PATH
#TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows example
#pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Load NLP model

from spacy.cli import download

# Check if the model is installed, otherwise download it
import spacy
import subprocess
import sys


import os
import pytesseract

# Set Tesseract path explicitly for Hugging Face
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Verify Tesseract installation
def verify_tesseract():
    try:
        test_img = Image.new('RGB', (100, 100), color='white')
        pytesseract.image_to_string(test_img)
        return True
    except Exception as e:
        st.error(f"Tesseract test failed: {str(e)}")
        return False
# Check if model is already installed, if not, install it
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize models (dummy implementations that actually work)
class InsuranceModel:
    def __init__(self):
        # In a real app, this would be a trained model
        self.risk_factors = {
            "diabetes": 1.8, "hypertension": 1.5, "cancer": 3.0,
            "stroke": 2.5, "heart disease": 2.7, "COPD": 2.0
        }
        self.coverage_rules = [
            {"type": "Orthopedic Surgery Cover", "trigger_words": ["meniscus", "knee", "surgery", "orthopedic", "tear"]},
            {"type": "Physiotherapy OPD Cover", "trigger_words": ["physiotherapy", "therapy", "rehab"]},
            {"type": "Workplace Injury Insurance", "trigger_words": ["injury", "work", "accident"]}
        ]
    
    def calculate_risk_score(self, medical_data):
        """Calculate insurance risk score (0-100) based on conditions"""
        base_score = 30
        conditions = medical_data.get("medical_condition", {}).get("symptoms", [])
        
        # Apply risk multipliers
        for condition, multiplier in self.risk_factors.items():
            if any(condition in cond.lower() for cond in conditions):
                base_score *= multiplier
        
        # Age factor (simplified)
        age = medical_data.get("patient_info", {}).get("age")
        if age and age.isdigit():
            age = int(age)
            base_score *= min(1 + (max(age, 30) - 30) * 0.02, 1.8)
        
        return min(int(base_score), 100)
    
    def recommend_coverage(self, medical_data):
        """Recommend specific insurance coverage based on medical conditions"""
        text_data = str(medical_data).lower()
        recommendations = []
        
        for rule in self.coverage_rules:
            if any(word in text_data for word in rule["trigger_words"]):
                recommendations.append(rule["type"])
        
        if not recommendations:
            recommendations.append("Comprehensive Health Insurance Plan")
            
        return recommendations

class LoanModel:
    def __init__(self):
        # In a real app, this would be a trained model
        self.risk_factors = {
            "cancer": 0.4, "stroke": 0.35, "heart disease": 0.3,
            "diabetes": 0.25, "hypertension": 0.2, "COPD": 0.3
        }
    
    def calculate_loan_risk(self, medical_data, loan_amount, duration_years):
        """Calculate loan approval probability (0-100%)"""
        base_risk = 20  # Base approval probability percentage
        conditions = medical_data.get("medical_condition", {}).get("symptoms", [])
        
        # Apply risk reductions for serious conditions
        for condition, risk_reduction in self.risk_factors.items():
            if any(condition in cond.lower() for cond in conditions):
                base_risk *= (1 - risk_reduction)
        
        # Adjust for loan parameters
        loan_risk = loan_amount / 500000  # Higher loan = higher risk
        duration_risk = duration_years * 0.05
        
        # Final probability
        probability = max(5, min(95, base_risk - (loan_risk + duration_risk)*10))
        
        return probability

class MedicalReportProcessor:
    def __init__(self):
        # Initialize matcher for medical terms
        self.matcher = Matcher(nlp.vocab)
        self._add_medical_patterns()

        # Regex templates for structured extraction
        self.templates = {
            'demographics': {
                'name': r"Patient Name\s*[:\-]\s*(.*?)\s*(?:Referring MD|Doctor|Physician|$)",
                'dob': r"Date of Birth\s*[:\-]\s*([\d/.-]+)",
                'age': r"Age\s*[:\-]\s*(\d{1,3})",
                'gender': r"Gender\s*[:\-]\s*(Male|Female|Other|M|F)",
                'physician': r"Referring MD\s*[:\-]\s*(.*?)\n"
            },
            'diagnosis': {
                'primary': r"DIAGNOSIS\s*[:\-]\s*(.*?)\n",
                'icd_code': r"ICD-10\s*[:\-]\s*([A-Z]\d{2}\.?\d*)"
            },
            'treatment': {
                'plan': r"PLAN\s*[:\-]\s*(.*?)(?=\n\n|$)",
                'medications': r"Current Medications\s*[:\-]\s*(.*?)\n"
            },
            'progress': {
                'initial': r"Initial Condition.*?[:\-]\s*(.*?)\n",
                'current': r"Current Status.*?[:\-]\s*(.*?)\n",
                'goals': r"Goals.*?[:\-]\s*(.*?)(?=\n\n|$)"
            }
        }

    def _add_medical_patterns(self):
        # Patterns for pain symptoms
        pain_patterns = [
            [{"LOWER": "pain"}],
            [{"LOWER": "discomfort"}],
            [{"LOWER": "ache"}],
            [{"LOWER": "burning"}],
            [{"LOWER": "throbbing"}]
        ]
        self.matcher.add("PAIN", pain_patterns)

        # Patterns for medications
        med_patterns = [
            [{"LOWER": {"REGEX": "(aspirin|ibuprofen|paracetamol|metformin|amoxicillin|atorvastatin)"}}],
            [{"LOWER": {"REGEX": ".*ine$|.*ol$|.*cin$"}}],
        ]
        self.matcher.add("MEDICATION", med_patterns)

        # Procedure keywords
        proc_patterns = [
            [{"LOWER": "surgery"}],
            [{"LOWER": "injection"}],
            [{"LOWER": "therapy"}],
            [{"LOWER": "operation"}],
            [{"LOWER": "procedure"}]
        ]
        self.matcher.add("PROCEDURE", proc_patterns)

    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data using regex patterns"""
        results = {}
        for section, fields in self.templates.items():
            results[section] = {}
            for field, pattern in fields.items():
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    value = re.sub(r'[\n\r]+', ' ', value)
                    results[section][field] = value
                else:
                    results[section][field] = None
        return results

    def analyze_unstructured_text(self, text: str) -> Dict[str, Any]:
        """Analyze unstructured text with spaCy"""
        doc = nlp(text)
        matches = self.matcher(doc)

        entities = {
            "symptoms": set(),
            "medications": set(),
            "procedures": set(),
            "dates": set(),
        }

        # Collect matches
        for match_id, start, end in matches:
            label = nlp.vocab.strings[match_id]
            span = doc[start:end].text.strip()
            if label == "PAIN":
                entities["symptoms"].add(span)
            elif label == "MEDICATION":
                entities["medications"].add(span)
            elif label == "PROCEDURE":
                entities["procedures"].add(span)

        # Extract DATE entities
        for ent in doc.ents:
            if ent.label_ == "DATE":
                entities["dates"].add(ent.text.strip())

        # Extract key sentences
        key_sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if any(token.ent_type_ in ["DRUG"] for token in sent):
                key_sentences.append(sent_text)
            if len(key_sentences) >= 5:
                break

        return {
            "entities": {k: list(v) for k, v in entities.items()},
            "key_sentences": key_sentences
        }

    def generate_summary(self, structured_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a clean medical summary"""
        def safe_get(d, key):
            return d.get(key) if d else None

        summary = {
            "patient_info": structured_data.get("demographics", {}),
            "medical_condition": {
                "diagnosis": safe_get(structured_data.get("diagnosis", {}), "primary") or "Not specified",
                "icd_code": safe_get(structured_data.get("diagnosis", {}), "icd_code") or "N/A",
                "symptoms": analysis["entities"].get("symptoms", [])
            },
            "treatment": {
                "plan": safe_get(structured_data.get("treatment", {}), "plan") or "Not specified",
                "medications": analysis["entities"].get("medications", []),
                "procedures": analysis["entities"].get("procedures", [])
            },
            "progress": {
                "initial_condition": safe_get(structured_data.get("progress", {}), "initial") or "Not specified",
                "current_status": safe_get(structured_data.get("progress", {}), "current") or "Not specified",
                "goals": safe_get(structured_data.get("progress", {}), "goals") or "Not specified",
                "key_findings": analysis.get("key_sentences", [])
            },
            "timeline": {
                "dates": analysis["entities"].get("dates", []),
                "age": safe_get(structured_data.get("demographics", {}), "age") or "N/A"
            }
        }
        return summary

    def generate_human_readable(self, summary: Dict[str, Any]) -> str:
        """Generate markdown report"""
        pi = summary.get('patient_info', {})
        mc = summary.get('medical_condition', {})
        tr = summary.get('treatment', {})
        pr = summary.get('progress', {})
        tl = summary.get('timeline', {})

        output = f"# Medical Summary Report\n\n"
        output += f"## Patient Demographics\n"
        output += f"- **Name:** {pi.get('name') or 'N/A'}\n"
        output += f"- **Date of Birth:** {pi.get('dob') or 'N/A'}\n"
        output += f"- **Age:** {pi.get('age') or 'N/A'}\n"
        output += f"- **Gender:** {pi.get('gender') or 'N/A'}\n"
        output += f"- **Referring Physician:** {pi.get('physician') or 'N/A'}\n\n"

        output += f"## Medical Condition\n"
        output += f"- **Diagnosis:** {mc.get('diagnosis')}\n"
        output += f"- **ICD Code:** {mc.get('icd_code')}\n"
        output += f"- **Symptoms:** {', '.join(mc.get('symptoms') or ['N/A'])}\n\n"

        output += f"## Treatment Plan\n"
        output += f"- **Plan:** {tr.get('plan')}\n"
        output += f"- **Medications:** {', '.join(tr.get('medications') or ['N/A'])}\n"
        output += f"- **Procedures:** {', '.join(tr.get('procedures') or ['N/A'])}\n\n"

        output += f"## Patient Progress\n"
        output += f"- **Initial Condition:** {pr.get('initial_condition')}\n"
        output += f"- **Current Status:** {pr.get('current_status')}\n"
        output += f"- **Treatment Goals:** {pr.get('goals')}\n\n"

        output += f"## Key Clinical Findings\n"
        if pr.get('key_findings'):
            for i, finding in enumerate(pr['key_findings'], 1):
                output += f"{i}. {finding}\n"
        else:
            output += "No key findings detected.\n"

        output += f"\n## Timeline and Other Info\n"
        output += f"- **Dates mentioned:** {', '.join(tl.get('dates') or ['N/A'])}\n"
        output += f"- **Age:** {tl.get('age')}\n"

        return output
    
#POPPLER_PATH = r"C:\Users\VRINDA\Downloads\Release-23.11.0-0\poppler-23.11.0\Library\bin"

class DocumentProcessor:
    @staticmethod
    def extract_text(uploaded_file):
        """Extract text from both scanned and digital PDFs using PyMuPDF only"""
        text = ""
        uploaded_file.seek(0)  # Reset file pointer
        file_bytes = uploaded_file.read()
        
        try:
            # Try digital PDF extraction
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
                if len(text.strip()) > 50:  # Check if extraction successful
                    return text
        except Exception as e:
            st.warning(f"Digital text extraction failed: {str(e)}")
        
        # Fallback to OCR for scanned PDFs using PyMuPDF

                # Fallback to OCR with memory limits
        try:
            images = convert_from_bytes(
                file_bytes,
                dpi=200,
                thread_count=2,
                fmt='jpeg',
                poppler_timeout=30
            )
            
            for img in images:
                text += pytesseract.image_to_string(img) + "\n\n"
                
            return text if text.strip() else "No text extracted"
        except Exception as e:
            st.error(f"PDF processing failed: {str(e)}")
            return ""
        '''        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    # Render page to image (300 DPI)
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Perform OCR
                    text += pytesseract.image_to_string(img)
            return text
        except Exception as e:
            st.error(f"PDF OCR failed: {str(e)}")
            return ""'''
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\f', '', text)
        return text.strip()

# Initialize models
insurance_model = InsuranceModel()
loan_model = LoanModel()
report_processor = MedicalReportProcessor()

# Streamlit App UI
def main():
    st.set_page_config(
        page_title="Medical Report Analyzer", 
        page_icon="🏥", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Medical Report Analyzer")
    st.caption("Upload medical reports for insurance assessment and loan eligibility")
    
    # Create sidebar for model controls
    with st.sidebar:
        st.header("Model Parameters")
        st.subheader("Insurance Model")
        coverage_amount = st.slider("Coverage Amount ($)", 10000, 1000000, 100000, step=5000)
        
        st.subheader("Loan Model")
        loan_amount = st.slider("Loan Amount ($)", 5000, 500000, 50000, step=1000)
        loan_duration = st.slider("Loan Duration (years)", 1, 30, 5)
        
        st.markdown("---")
        st.caption("Developed by Medical AI Systems")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        # Process PDF
        with st.spinner("Processing medical report..."):
            raw_text = DocumentProcessor.extract_text(uploaded_file)
            clean_text = DocumentProcessor.clean_text(raw_text)
            
            # Display extracted text
            with st.expander("View Extracted Text"):
                st.text(clean_text[:3000] + ("..." if len(clean_text) > 3000 else ""))
            
            # Process medical report
            structured_data = report_processor.extract_structured_data(clean_text)
            analysis = report_processor.analyze_unstructured_text(clean_text)
            summary = report_processor.generate_summary(structured_data, analysis)
            readable_report = report_processor.generate_human_readable(summary)
        
        # Display medical summary
        st.subheader("Medical Report Summary")
        st.markdown(readable_report, unsafe_allow_html=True)
        
        # Insurance and Loan Analysis Tabs
        tab1, tab2 = st.tabs(["Insurance Assessment", "Loan Eligibility"])
        
        with tab1:
            st.subheader("Insurance Risk Assessment")
            
            # Calculate insurance risk
            risk_score = insurance_model.calculate_risk_score(summary)
            coverage_recommendations = insurance_model.recommend_coverage(summary)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{risk_score}/100")
            col2.metric("Recommended Coverage", coverage_amount)
            
            # Risk visualization
            st.progress(risk_score/100)
            
            # Coverage recommendations
            st.subheader("Recommended Coverage Types")
            for coverage in coverage_recommendations:
                st.info(f"✅ {coverage}")
            
            # Risk factors
            st.subheader("Key Risk Factors")
            if summary.get("medical_condition", {}).get("symptoms"):
                for symptom in summary["medical_condition"]["symptoms"]:
                    st.write(f"- {symptom}")
            else:
                st.write("No significant risk factors identified")
        
        with tab2:
            st.subheader("Loan Eligibility Assessment")
            
            # Calculate loan risk
            approval_prob = loan_model.calculate_loan_risk(
                summary, 
                loan_amount, 
                loan_duration
            )
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Approval Probability", f"{approval_prob:.1f}%")
            col2.metric("Loan Amount", f"${loan_amount:,}")
            col3.metric("Loan Duration", f"{loan_duration} years")
            
            # Loan visualization
            st.subheader("Risk Analysis")
            
            # Risk factors
            st.write("Medical factors affecting loan approval:")
            if summary.get("medical_condition", {}).get("symptoms"):
                for symptom in summary["medical_condition"]["symptoms"]:
                    st.write(f"- {symptom}")
            else:
                st.write("No significant medical risk factors")
            
            # Decision explanation
            st.subheader("Recommendation")
            if approval_prob > 70:
                st.success("✅ Strong candidate for loan approval")
                st.markdown("""
                **Next Steps:**
                - Complete loan application
                - Provide income verification
                - Submit collateral documents
                """)
            elif approval_prob > 40:
                st.warning("⚠️ Conditional approval recommended")
                st.markdown("""
                **Requirements:**
                - Higher interest rate
                - Co-signer required
                - Medical clearance certificate
                """)
            else:
                st.error("❌ Loan not recommended")
                st.markdown("""
                **Reasons:**
                - High medical risk factors
                - Potential impact on repayment ability
                - Consider alternative financing options
                """)

if __name__ == "__main__":
    main()