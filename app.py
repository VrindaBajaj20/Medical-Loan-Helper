import streamlit as st
import fitz  # PyMuPDF
import pytesseract
import re
import spacy
from spacy.matcher import Matcher
import numpy as np
import pandas as pd
import os
import subprocess
import sys
from PIL import Image
from typing import Dict, List, Set, Any
from pdfminer.high_level import extract_text as pdfminer_extract_text

# ==================== DEPENDENCY MANAGEMENT ====================
class SystemDependencies:
    @staticmethod
    def initialize():
        """Ensure all system and Python dependencies are ready"""
        # Install system dependencies
        SystemDependencies._install_system_deps()
        
        # Load NLP model
        nlp = SystemDependencies._load_spacy_model()
        
        # Configure Tesseract
        SystemDependencies._configure_tesseract()
        
        return nlp

    @staticmethod
    def _install_system_deps():
        """Install required system packages"""
        deps = {
            'poppler': ['poppler-utils'],
            'tesseract': ['tesseract-ocr', 'tesseract-ocr-eng']
        }
        
        for dep, packages in deps.items():
            if not getattr(SystemDependencies, f'_check_{dep}')():
                st.warning(f"Installing {dep}...")
                try:
                    subprocess.run(["apt-get", "update"], check=True)
                    subprocess.run(["apt-get", "install", "-y"] + packages, check=True)
                    st.success(f"{dep.capitalize()} installed successfully!")
                except Exception as e:
                    st.error(f"Failed to install {dep}: {str(e)}")
                    st.stop()

    @staticmethod
    def _load_spacy_model():
        """Load or download the spaCy model"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            st.warning("Downloading spaCy model...")
            try:
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                return spacy.load("en_core_web_sm")
            except Exception as e:
                st.error(f"Failed to load spaCy model: {str(e)}")
                st.stop()

    @staticmethod
    def _configure_tesseract():
        """Set up Tesseract paths"""
        try:
            # Try standard Hugging Face path first
            tesseract_path = "/usr/bin/tesseract"
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            else:
                # Fallback to auto-detection
                pytesseract.pytesseract.tesseract_cmd = pytesseract.get_tesseract_version()
        except Exception as e:
            st.warning(f"Tesseract configuration warning: {str(e)}")

    @staticmethod
    def _check_poppler():
        """Verify poppler-utils is installed"""
        try:
            subprocess.run(["pdftoppm", "-v"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def _check_tesseract():
        """Verify Tesseract is installed"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except (pytesseract.TesseractNotFoundError, EnvironmentError):
            return False

# ==================== CORE MODELS ====================
class InsuranceModel:
    def __init__(self):
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
        base_score = 30
        conditions = medical_data.get("medical_condition", {}).get("symptoms", [])
        
        for condition, multiplier in self.risk_factors.items():
            if any(condition in cond.lower() for cond in conditions):
                base_score *= multiplier
        
        age = medical_data.get("patient_info", {}).get("age")
        if age and age.isdigit():
            age = int(age)
            base_score *= min(1 + (max(age, 30) - 30) * 0.02, 1.8)
        
        return min(int(base_score), 100)
    
    def recommend_coverage(self, medical_data):
        text_data = str(medical_data).lower()
        recommendations = []
        
        for rule in self.coverage_rules:
            if any(word in text_data for word in rule["trigger_words"]):
                recommendations.append(rule["type"])
        
        return recommendations or ["Comprehensive Health Insurance Plan"]

class LoanModel:
    def __init__(self):
        self.risk_factors = {
            "cancer": 0.4, "stroke": 0.35, "heart disease": 0.3,
            "diabetes": 0.25, "hypertension": 0.2, "COPD": 0.3
        }
    
    def calculate_loan_risk(self, medical_data, loan_amount, duration_years):
        base_risk = 20
        conditions = medical_data.get("medical_condition", {}).get("symptoms", [])
        
        for condition, risk_reduction in self.risk_factors.items():
            if any(condition in cond.lower() for cond in conditions):
                base_risk *= (1 - risk_reduction)
        
        loan_risk = loan_amount / 500000
        duration_risk = duration_years * 0.05
        
        return max(5, min(95, base_risk - (loan_risk + duration_risk)*10))

class MedicalReportProcessor:
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = Matcher(nlp.vocab)
        self._add_medical_patterns()
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
        patterns = {
            "PAIN": [[{"LOWER": word}] for word in ["pain", "discomfort", "ache", "burning", "throbbing"]],
            "MEDICATION": [
                [{"LOWER": {"REGEX": "(aspirin|ibuprofen|paracetamol|metformin|amoxicillin|atorvastatin)"}}],
                [{"LOWER": {"REGEX": ".*ine$|.*ol$|.*cin$"}}]
            ],
            "PROCEDURE": [[{"LOWER": word}] for word in ["surgery", "injection", "therapy", "operation", "procedure"]]
        }
        for label, pattern_list in patterns.items():
            self.matcher.add(label, pattern_list)

    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        results = {}
        for section, fields in self.templates.items():
            results[section] = {}
            for field, pattern in fields.items():
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                results[section][field] = match.group(1).strip() if match else None
        return results

    def analyze_unstructured_text(self, text: str) -> Dict[str, Any]:
        doc = self.nlp(text)
        matches = self.matcher(doc)
        entities = {k: set() for k in ["symptoms", "medications", "procedures", "dates"]}

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end].text.strip()
            if label == "PAIN":
                entities["symptoms"].add(span)
            elif label == "MEDICATION":
                entities["medications"].add(span)
            elif label == "PROCEDURE":
                entities["procedures"].add(span)

        entities["dates"] = {ent.text.strip() for ent in doc.ents if ent.label_ == "DATE"}
        
        key_sentences = [
            sent.text.strip() for sent in doc.sents 
            if any(token.ent_type_ in ["DRUG"] for token in sent)
        ][:5]

        return {
            "entities": {k: list(v) for k, v in entities.items()},
            "key_sentences": key_sentences
        }

    def generate_summary(self, structured_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        def safe_get(d, key):
            return d.get(key) if d else None

        return {
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

    def generate_human_readable(self, summary: Dict[str, Any]) -> str:
        sections = [
            ("Patient Demographics", summary.get('patient_info', {})),
            ("Medical Condition", summary.get('medical_condition', {})),
            ("Treatment Plan", summary.get('treatment', {})),
            ("Patient Progress", summary.get('progress', {})),
            ("Timeline and Other Info", summary.get('timeline', {}))
        ]
        
        output = "# Medical Summary Report\n\n"
        for title, data in sections:
            output += f"## {title}\n"
            if title == "Patient Demographics":
                output += "\n".join(f"- **{k.capitalize()}:** {v or 'N/A'}" 
                                  for k, v in data.items())
            elif title == "Medical Condition":
                output += f"- **Diagnosis:** {data.get('diagnosis')}\n"
                output += f"- **ICD Code:** {data.get('icd_code')}\n"
                output += f"- **Symptoms:** {', '.join(data.get('symptoms') or ['N/A'])}\n"
            elif title == "Treatment Plan":
                output += f"- **Plan:** {data.get('plan')}\n"
                output += f"- **Medications:** {', '.join(data.get('medications') or ['N/A'])}\n"
                output += f"- **Procedures:** {', '.join(data.get('procedures') or ['N/A'])}\n"
            elif title == "Patient Progress":
                output += f"- **Initial Condition:** {data.get('initial_condition')}\n"
                output += f"- **Current Status:** {data.get('current_status')}\n"
                output += f"- **Treatment Goals:** {data.get('goals')}\n"
                if data.get('key_findings'):
                    output += "\n## Key Clinical Findings\n"
                    output += "\n".join(f"{i}. {finding}" 
                                      for i, finding in enumerate(data['key_findings'], 1))
                else:
                    output += "\n## Key Clinical Findings\nNo key findings detected.\n"
            elif title == "Timeline and Other Info":
                output += f"- **Dates mentioned:** {', '.join(data.get('dates') or ['N/A'])}\n"
                output += f"- **Age:** {data.get('age')}\n"
            output += "\n"
        
        return output

class DocumentProcessor:
    @staticmethod
    def extract_text(uploaded_file):
        """Robust text extraction with multiple fallback methods"""
        file_bytes = uploaded_file.read()
        
        # Method 1: Native PDF text extraction
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
                if len(text.strip()) > 50:
                    return text
        except Exception:
            pass
        
        # Method 2: PDF rendering + Tesseract OCR
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img) + "\n\n"
            return text
        except Exception:
            pass
        
        # Method 3: PDFMiner fallback
        try:
            return pdfminer_extract_text(uploaded_file)
        except Exception as e:
            st.error(f"All extraction methods failed: {str(e)}")
            return ""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\f', '', text)
        return text.strip()

# ==================== STREAMLIT APP ====================
def main():
    # Initialize all dependencies and models
    nlp = SystemDependencies.initialize()
    insurance_model = InsuranceModel()
    loan_model = LoanModel()
    report_processor = MedicalReportProcessor(nlp)

    # App configuration
    st.set_page_config(
        page_title="Medical Report Analyzer", 
        page_icon="🏥", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Medical Report Analyzer")
    st.caption("Upload medical reports for insurance assessment and loan eligibility")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Model Parameters")
        st.subheader("Insurance Model")
        coverage_amount = st.slider("Coverage Amount ($)", 10000, 1000000, 100000, step=5000)
        
        st.subheader("Loan Model")
        loan_amount = st.slider("Loan Amount ($)", 5000, 500000, 50000, step=1000)
        loan_duration = st.slider("Loan Duration (years)", 1, 30, 5)
        
        st.markdown("---")
        st.caption("Developed by Medical AI Systems")
    
    # Main file processing
    uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing medical report..."):
            # Extract and clean text
            raw_text = DocumentProcessor.extract_text(uploaded_file)
            clean_text = DocumentProcessor.clean_text(raw_text)
            
            # Show extracted text preview
            with st.expander("View Extracted Text"):
                st.text(clean_text[:3000] + ("..." if len(clean_text) > 3000 else ""))
            
            # Process report
            structured_data = report_processor.extract_structured_data(clean_text)
            analysis = report_processor.analyze_unstructured_text(clean_text)
            summary = report_processor.generate_summary(structured_data, analysis)
            readable_report = report_processor.generate_human_readable(summary)
        
        # Display results
        st.subheader("Medical Report Summary")
        st.markdown(readable_report, unsafe_allow_html=True)
        
        # Analysis tabs
        tab1, tab2 = st.tabs(["Insurance Assessment", "Loan Eligibility"])
        
        with tab1:
            st.subheader("Insurance Risk Assessment")
            risk_score = insurance_model.calculate_risk_score(summary)
            coverage_recommendations = insurance_model.recommend_coverage(summary)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{risk_score}/100")
            col2.metric("Recommended Coverage", coverage_amount)
            
            st.progress(risk_score/100)
            
            st.subheader("Recommended Coverage Types")
            for coverage in coverage_recommendations:
                st.info(f"✅ {coverage}")
            
            st.subheader("Key Risk Factors")
            if summary.get("medical_condition", {}).get("symptoms"):
                for symptom in summary["medical_condition"]["symptoms"]:
                    st.write(f"- {symptom}")
            else:
                st.write("No significant risk factors identified")
        
        with tab2:
            st.subheader("Loan Eligibility Assessment")
            approval_prob = loan_model.calculate_loan_risk(summary, loan_amount, loan_duration)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Approval Probability", f"{approval_prob:.1f}%")
            col2.metric("Loan Amount", f"${loan_amount:,}")
            col3.metric("Loan Duration", f"{loan_duration} years")
            
            st.subheader("Risk Analysis")
            st.write("Medical factors affecting loan approval:")
            if summary.get("medical_condition", {}).get("symptoms"):
                for symptom in summary["medical_condition"]["symptoms"]:
                    st.write(f"- {symptom}")
            else:
                st.write("No significant medical risk factors")
            
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

