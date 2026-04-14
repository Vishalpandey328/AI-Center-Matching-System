import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
import time
import json
import pickle
import io
import chardet
from datetime import datetime
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize
from collections import defaultdict
import torch
from typing import List, Dict, Tuple
import openpyxl
from openpyxl import load_workbook

# ------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="AI Powered Center Matching System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL CYBERPUNK UI STYLE
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0f1f 0%, #1a1f2f 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 32px;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 10px 0 5px 0;
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        text-align: center;
        color: #a0a0a0;
        margin-bottom: 15px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        color: #a0a0a0;
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        font-weight: 600;
        color: #667eea;
        margin-top: 4px;
    }
    
    .stButton > button {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
    }
    
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        font-weight: 500;
        color: white;
        margin: 15px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 15px 0;
    }
    
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("<h1 class='main-title'>AI Center Matching System</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Multi-Strategy Neural Matching • Self-Learning AI</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>AI Engine</div>
        <div class='metric-value'>MULTI-STRATEGY</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Learning</div>
        <div class='metric-value'>ENABLED</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Matchers</div>
        <div class='metric-value'>5 ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-label'>Status</div>
        <div class='metric-value'><span class='status-dot'></span>ONLINE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# FILE LOADING WITH ENCODING HANDLING
# --------------------------------------------------

def detect_encoding(file_bytes):
    """Detect file encoding"""
    try:
        result = chardet.detect(file_bytes)
        return result['encoding'] if result['encoding'] else 'utf-8'
    except:
        return 'utf-8'

def load_file_with_encoding(uploaded_file):
    """Load CSV or Excel file with proper encoding handling"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Try different encodings for CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    st.success(f"✅ Successfully loaded with {encoding} encoding")
                    return df
                except:
                    continue
            
            # If all fail, try with error handling
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1', errors='ignore')
            st.warning("⚠️ Loaded with latin-1 encoding (some characters may be replaced)")
            return df
            
        else:  # Excel file
            # Try different engines for Excel
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                return df
            except:
                try:
                    df = pd.read_excel(uploaded_file, engine='xlrd')
                    return df
                except:
                    # If Excel fails, try reading as CSV
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1', errors='ignore')
                    st.warning("⚠️ File loaded as CSV (Excel format issue)")
                    return df
                    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def safe_text_convert(text):
    """Safely convert text to string handling encoding issues"""
    if pd.isna(text):
        return ""
    try:
        return str(text).encode('utf-8', errors='ignore').decode('utf-8')
    except:
        return str(text)

# --------------------------------------------------
# MULTI-STRATEGY MATCHING ENGINE
# --------------------------------------------------

class MultiStrategyMatcher:
    """Combines multiple matching strategies for better accuracy"""
    
    def __init__(self):
        self.strategies = {
            'exact': {'weight': 0.15, 'threshold': 0.9},
            'fuzzy': {'weight': 0.25, 'threshold': 0.7},
            'vector': {'weight': 0.20, 'threshold': 0.6},
            'token': {'weight': 0.20, 'threshold': 0.65},
            'address': {'weight': 0.20, 'threshold': 0.5}
        }
        
    def exact_match_score(self, str1, str2):
        """Exact string matching after normalization"""
        if pd.isna(str1) or pd.isna(str2):
            return 0.0
        str1_norm = safe_text_convert(str1).lower().strip()
        str2_norm = safe_text_convert(str2).lower().strip()
        return 1.0 if str1_norm == str2_norm else 0.0
    
    def fuzzy_match_score(self, str1, str2):
        """Multiple fuzzy matching techniques"""
        if pd.isna(str1) or pd.isna(str2):
            return 0.0
        
        str1 = safe_text_convert(str1).lower()
        str2 = safe_text_convert(str2).lower()
        
        try:
            ratio_score = fuzz.ratio(str1, str2) / 100
            partial_score = fuzz.partial_ratio(str1, str2) / 100
            token_score = fuzz.token_set_ratio(str1, str2) / 100
            token_sort_score = fuzz.token_sort_ratio(str1, str2) / 100
            
            final_score = (0.3 * ratio_score + 0.2 * partial_score + 
                          0.3 * token_score + 0.2 * token_sort_score)
            
            return final_score
        except:
            return 0.0
    
    def token_match_score(self, str1, str2):
        """Token-based matching (word overlap)"""
        if pd.isna(str1) or pd.isna(str2):
            return 0.0
        
        try:
            tokens1 = set(safe_text_convert(str1).lower().split())
            tokens2 = set(safe_text_convert(str2).lower().split())
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def calculate_comprehensive_score(self, input_row, master_row, address_matcher=None):
        """Calculate matching score using all strategies"""
        
        scores = {}
        
        # Name matching
        name = safe_text_convert(input_row.get('center_name', ''))
        master_name = safe_text_convert(master_row.get('center_name', ''))
        
        scores['name_exact'] = self.exact_match_score(name, master_name)
        scores['name_fuzzy'] = self.fuzzy_match_score(name, master_name)
        scores['name_token'] = self.token_match_score(name, master_name)
        
        # District matching
        district = safe_text_convert(input_row.get('district', ''))
        master_district = safe_text_convert(master_row.get('district', ''))
        scores['district_match'] = self.fuzzy_match_score(district, master_district)
        
        # State matching
        state = safe_text_convert(input_row.get('state', ''))
        master_state = safe_text_convert(master_row.get('state', ''))
        scores['state_match'] = self.fuzzy_match_score(state, master_state)
        
        # Address matching
        address = safe_text_convert(input_row.get('address', ''))
        master_address = safe_text_convert(master_row.get('address', ''))
        
        if address_matcher:
            scores['address_match'] = address_matcher.calculate_address_similarity(address, master_address)
        else:
            scores['address_match'] = self.fuzzy_match_score(address, master_address)
        
        # Calculate weighted final score
        final_score = (
            0.30 * max(scores['name_exact'], scores['name_fuzzy'], scores['name_token']) +
            0.25 * scores['district_match'] +
            0.10 * scores['state_match'] +
            0.35 * scores['address_match']
        )
        
        return final_score, scores

# --------------------------------------------------
# ENHANCED ADDRESS MATCHER
# --------------------------------------------------

class AddressMatcher:
    def __init__(self):
        self.address_patterns = {
            r'\bblock\b': 'block', r'\bnear\b': 'near', r'\bopposite\b': 'opp',
            r'\bpolice station\b': 'ps', r'\brailway station\b': 'railway stn',
            r'\bmetro station\b': 'metro stn', r'\bmain road\b': 'main rd',
            r'\broad\b': 'rd', r'\bstreet\b': 'st', r'\bnagar\b': 'nagar',
            r'\bvihar\b': 'vihar', r'\bcolony\b': 'colony', r'\bextension\b': 'extn',
            r'\bphase\b': 'ph', r'\bsector\b': 'sec'
        }
    
    def normalize_address(self, address):
        if pd.isna(address):
            return ""
        try:
            address = safe_text_convert(address).lower()
            address = re.sub(r'[^\w\s\-/]', ' ', address)
            for pattern, replacement in self.address_patterns.items():
                address = re.sub(pattern, replacement, address)
            address = re.sub(r'\s+', ' ', address).strip()
            return address
        except:
            return ""
    
    def calculate_address_similarity(self, addr1, addr2):
        if not addr1 or not addr2:
            return 0.0
        
        try:
            addr1_norm = self.normalize_address(addr1)
            addr2_norm = self.normalize_address(addr2)
            
            token_sim = fuzz.token_set_ratio(addr1_norm, addr2_norm) / 100
            partial_sim = fuzz.partial_ratio(addr1_norm, addr2_norm) / 100
            
            pincode1 = re.findall(r'\b\d{6}\b', addr1_norm)
            pincode2 = re.findall(r'\b\d{6}\b', addr2_norm)
            pincode_sim = 1.0 if pincode1 and pincode2 and pincode1[0] == pincode2[0] else 0.0
            
            final_score = 0.5 * token_sim + 0.3 * partial_sim + 0.2 * pincode_sim
            return final_score
        except:
            return 0.0

# --------------------------------------------------
# SYNONYM MANAGER
# --------------------------------------------------

class SynonymManager:
    def __init__(self, synonym_file="synonyms.csv"):
        self.synonym_file = synonym_file
        self.load_synonyms()
    
    def load_synonyms(self):
        if os.path.exists(self.synonym_file):
            try:
                self.synonyms_df = pd.read_csv(self.synonym_file, encoding='utf-8')
            except:
                self.synonyms_df = pd.read_csv(self.synonym_file, encoding='latin-1')
        else:
            self.synonyms_df = pd.DataFrame({
                'word': ['govt', 'rajkiya', 'mahila', 'balika', 'balak', 'pg', 'inter', 
                        'vidyalaya', 'kendra', 'nagar', 'gram', 'prakhand', 'mandal'],
                'replacement': ['government', 'government', 'girls', 'girls', 'boys', 
                               'postgraduate', 'intermediate', 'school', 'center', 
                               'city', 'village', 'block', 'district']
            })
            self.save_synonyms()
    
    def save_synonyms(self):
        self.synonyms_df.to_csv(self.synonym_file, index=False, encoding='utf-8')
    
    def get_enabled_synonyms(self):
        return self.synonyms_df
    
    def add_synonym(self, word, replacement):
        new_row = pd.DataFrame({'word': [safe_text_convert(word).lower()], 
                               'replacement': [safe_text_convert(replacement).lower()]})
        self.synonyms_df = pd.concat([self.synonyms_df, new_row], ignore_index=True)
        self.save_synonyms()
        return True
    
    def get_statistics(self):
        return {'total': len(self.synonyms_df)}

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def enhanced_clean_text(text, synonym_manager):
    if pd.isna(text):
        return ""
    try:
        text = safe_text_convert(text).lower()
        text = re.sub(r"[^\w\s\-/]", " ", text)
        
        for _, row in synonym_manager.get_enabled_synonyms().iterrows():
            word = safe_text_convert(row["word"]).lower()
            replacement = safe_text_convert(row["replacement"]).lower()
            text = re.sub(rf'\b{word}\b', replacement, text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except:
        return ""

# --------------------------------------------------
# ENHANCED MATCHING FUNCTION
# --------------------------------------------------

def match_centers_enhanced(input_data, master_df, model, index, synonym_manager, 
                          confidence_threshold=0.60, strategy='balanced'):
    """Enhanced matching with multiple strategies"""
    
    results = []
    match_details = []
    multi_matcher = MultiStrategyMatcher()
    address_matcher = AddressMatcher()
    
    # Prepare master data with safe text conversion
    master_df['clean_name'] = master_df['center_name'].apply(lambda x: enhanced_clean_text(x, synonym_manager))
    master_df['clean_district'] = master_df['district'].apply(lambda x: enhanced_clean_text(x, synonym_manager))
    master_df['clean_state'] = master_df['state'].apply(lambda x: enhanced_clean_text(x, synonym_manager))
    master_df['clean_address'] = master_df['address'].apply(lambda x: enhanced_clean_text(x, synonym_manager))
    
    # Adjust thresholds based on strategy
    if strategy == 'aggressive':
        confidence_threshold = max(0.45, confidence_threshold - 0.10)
        address_threshold = 0.3
        district_threshold = 0.4
    elif strategy == 'balanced':
        confidence_threshold = confidence_threshold
        address_threshold = 0.35
        district_threshold = 0.45
    else:  # conservative
        confidence_threshold = min(0.85, confidence_threshold + 0.05)
        address_threshold = 0.5
        district_threshold = 0.6
    
    total = len(input_data)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in input_data.iterrows():
        progress_bar.progress((idx + 1) / total)
        status_text.info(f"Processing record {idx+1} of {total}")
        
        try:
            # Clean input with safe conversion
            clean_name = enhanced_clean_text(row['center_name'], synonym_manager)
            clean_district = enhanced_clean_text(row['district'], synonym_manager)
            clean_state = enhanced_clean_text(row['state'], synonym_manager)
            clean_address = enhanced_clean_text(row['address'], synonym_manager)
            
            query_text = f"{clean_name} {clean_district} {clean_state} {clean_address}"
            
            # Generate embedding
            query_embedding = model.encode([query_text])
            query_embedding = normalize(query_embedding.astype(np.float32))
            
            # Vector search
            k = min(30, len(master_df))
            distances, indices = index.search(query_embedding, k)
            
            scored_candidates = []
            
            for master_idx, distance in zip(indices[0], distances[0]):
                if master_idx < len(master_df):
                    master_row = master_df.iloc[master_idx]
                    
                    # Calculate comprehensive score
                    final_score, component_scores = multi_matcher.calculate_comprehensive_score(
                        row.to_dict(), master_row.to_dict(), address_matcher
                    )
                    
                    vector_score = 1 / (1 + distance)
                    combined_score = 0.7 * final_score + 0.3 * vector_score
                    
                    scored_candidates.append({
                        'master_id': master_row.get('center_id', master_idx),
                        'master_name': safe_text_convert(master_row['center_name']),
                        'master_address': safe_text_convert(master_row.get('address', '')),
                        'master_district': safe_text_convert(master_row.get('district', '')),
                        'master_state': safe_text_convert(master_row.get('state', '')),
                        'score': combined_score,
                        'component_scores': component_scores
                    })
            
            # Sort by score
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Find best match
            best_match = None
            for candidate in scored_candidates[:10]:
                if (candidate['score'] >= confidence_threshold and
                    candidate['component_scores'].get('district_match', 0) >= district_threshold):
                    
                    address_score = candidate['component_scores'].get('address_match', 0)
                    if address_score >= address_threshold or candidate['score'] >= confidence_threshold + 0.15:
                        best_match = candidate
                        break
            
            # If still no match, try with relaxed conditions
            if best_match is None and scored_candidates:
                for candidate in scored_candidates[:5]:
                    if candidate['score'] >= confidence_threshold - 0.1:
                        best_match = candidate
                        break
            
            if best_match:
                results.append(best_match['master_name'])
                match_details.append({
                    'master_id': best_match['master_id'],
                    'master_name': best_match['master_name'],
                    'master_address': best_match['master_address'],
                    'master_district': best_match['master_district'],
                    'master_state': best_match['master_state'],
                    'confidence': best_match['score'],
                    'name_score': best_match['component_scores'].get('name_fuzzy', 0),
                    'address_score': best_match['component_scores'].get('address_match', 0),
                    'district_score': best_match['component_scores'].get('district_match', 0),
                    'state_score': best_match['component_scores'].get('state_match', 0)
                })
            else:
                results.append("⚡ No Match")
                match_details.append({
                    'master_id': 'NULL',
                    'master_name': 'No Match Found',
                    'master_address': 'N/A',
                    'master_district': 'N/A',
                    'master_state': 'N/A',
                    'confidence': 0,
                    'name_score': 0,
                    'address_score': 0,
                    'district_score': 0,
                    'state_score': 0
                })
        except Exception as e:
            st.warning(f"Error processing record {idx}: {str(e)}")
            results.append("⚡ Error")
            match_details.append({
                'master_id': 'ERROR',
                'master_name': 'Processing Error',
                'master_address': 'N/A',
                'master_district': 'N/A',
                'master_state': 'N/A',
                'confidence': 0,
                'name_score': 0,
                'address_score': 0,
                'district_score': 0,
                'state_score': 0
            })
    
    progress_bar.empty()
    status_text.empty()
    return results, match_details

# --------------------------------------------------
# CREATE DETAILED REPORT
# --------------------------------------------------

def create_detailed_report(input_df, match_details):
    report_df = input_df.copy()
    
    report_df['Matched Center ID'] = [d['master_id'] for d in match_details]
    report_df['Matched Center Name'] = [d['master_name'] for d in match_details]
    report_df['Matched Address'] = [d['master_address'] for d in match_details]
    report_df['Matched District'] = [d['master_district'] for d in match_details]
    report_df['Matched State'] = [d['master_state'] for d in match_details]
    report_df['Confidence Score'] = [d['confidence'] for d in match_details]
    report_df['Name Match Score'] = [d['name_score'] for d in match_details]
    report_df['Address Match Score'] = [d['address_score'] for d in match_details]
    report_df['District Match Score'] = [d['district_score'] for d in match_details]
    report_df['State Match Score'] = [d['state_score'] for d in match_details]
    
    report_df['Match Status'] = report_df['Matched Center Name'].apply(
        lambda x: '✅ Matched' if x not in ['No Match Found', 'Processing Error'] else '❌ No Match'
    )
    
    column_order = [
        'center_name', 'district', 'state', 'address',
        'Match Status', 'Matched Center Name', 'Matched Center ID',
        'Matched Address', 'Matched District', 'Matched State',
        'Confidence Score', 'Name Match Score', 'Address Match Score',
        'District Match Score', 'State Match Score'
    ]
    
    return report_df[column_order]

# --------------------------------------------------
# INITIALIZE MODELS
# --------------------------------------------------

@st.cache_resource
def init_synonym_manager():
    return SynonymManager()

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

synonym_manager = init_synonym_manager()

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

model_info = {
    "multi-qa-mpnet-base-dot-v1": {"name": "MPNet Base (Best Balance)", "size": "420 MB", "accuracy": "Very High"},
    "all-mpnet-base-v2": {"name": "MPNet V2 (All-rounder)", "size": "420 MB", "accuracy": "High"},
    "all-MiniLM-L6-v2": {"name": "MiniLM (Fastest)", "size": "80 MB", "accuracy": "Good"}
}

# --------------------------------------------------
# FILE UPLOAD SECTION
# --------------------------------------------------

st.markdown("<h2 class='section-header'>📂 Data Upload</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    master_file = st.file_uploader("Master Database (Excel/CSV)", type=["xlsx", "csv"], key="master")

with col2:
    input_file = st.file_uploader("Input Stream (Excel/CSV)", type=["xlsx", "csv"], key="input")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("<div class='sidebar-header'>🎯 Matching Configuration</div>", unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "AI Model",
        options=list(model_info.keys()),
        format_func=lambda x: model_info[x]["name"]
    )
    
    matching_strategy = st.select_slider(
        "Matching Strategy",
        options=['conservative', 'balanced', 'aggressive'],
        value='aggressive',  # Changed to aggressive by default for more matches
        help="Aggressive: More matches | Conservative: Higher accuracy"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.55,  # Lowered default for more matches
        step=0.05,
        help="Lower = more matches, Higher = more accurate"
    )
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Templates
    st.markdown("<div class='sidebar-header'>📥 Templates</div>", unsafe_allow_html=True)
    
    master_template = pd.DataFrame({
        "center_id": ["1001", "1002"],
        "center_name": ["ABC Public School", "XYZ College"],
        "district": ["Lucknow", "Kanpur"],
        "state": ["Uttar Pradesh", "Uttar Pradesh"],
        "address": ["Near City Mall", "Civil Lines"]
    })
    
    input_template = pd.DataFrame({
        "center_name": ["ABC School", "XYZ College"],
        "district": ["Lucknow", "Kanpur"],
        "state": ["Uttar Pradesh", "Uttar Pradesh"],
        "address": ["City Mall Area", "Civil Lines Kanpur"]
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Master Template", master_template.to_csv(index=False), "master_template.csv")
    with col2:
        st.download_button("Input Template", input_template.to_csv(index=False), "input_template.csv")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Synonyms
    st.markdown("<div class='sidebar-header'>📚 Synonyms</div>", unsafe_allow_html=True)
    
    stats = synonym_manager.get_statistics()
    st.info(f"Total synonyms: {stats['total']}")
    
    new_word = st.text_input("New Word", placeholder="e.g., vidyalaya")
    new_replacement = st.text_input("Replacement", placeholder="e.g., school")
    
    if st.button("➕ Add Synonym", use_container_width=True):
        if new_word and new_replacement:
            synonym_manager.add_synonym(new_word, new_replacement)
            st.success(f"Added: {new_word} → {new_replacement}")
            st.rerun()

# --------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------

if master_file and input_file:
    try:
        # Load files with proper encoding
        with st.spinner("Loading master file..."):
            master_df = load_file_with_encoding(master_file)
            if master_df is None:
                st.error("Failed to load master file")
                st.stop()
        
        with st.spinner("Loading input file..."):
            input_df = load_file_with_encoding(input_file)
            if input_df is None:
                st.error("Failed to load input file")
                st.stop()
        
        # Convert all text columns to safe strings
        text_columns = ['center_name', 'district', 'state', 'address']
        for col in text_columns:
            if col in master_df.columns:
                master_df[col] = master_df[col].apply(safe_text_convert)
            if col in input_df.columns:
                input_df[col] = input_df[col].apply(safe_text_convert)
        
        st.success(f"✅ Loaded {len(master_df)} master records and {len(input_df)} input records")
        
        # Load model
        with st.spinner(f"Loading {model_info[selected_model]['name']}..."):
            model = load_embedding_model(selected_model)
        
        # Process master data
        with st.spinner("Generating embeddings..."):
            master_df['clean_text'] = master_df.apply(
                lambda x: enhanced_clean_text(f"{x['center_name']} {x['district']} {x['state']} {x['address']}", 
                                              synonym_manager), axis=1
            )
            
            # Filter out empty texts
            master_df = master_df[master_df['clean_text'].str.len() > 0]
            
            if len(master_df) == 0:
                st.error("No valid master records after processing")
                st.stop()
            
            embeddings = model.encode(master_df['clean_text'].tolist(), show_progress_bar=True)
            embeddings = normalize(embeddings.astype(np.float32))
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
        
        # Perform matching
        with st.spinner("Matching with multi-strategy AI..."):
            results, match_details = match_centers_enhanced(
                input_df, master_df, model, index, synonym_manager, 
                confidence_threshold, matching_strategy
            )
        
        # Create report
        report_df = create_detailed_report(input_df, match_details)
        
        # Display results
        st.markdown("<h2 class='section-header'>📊 Matching Results</h2>", unsafe_allow_html=True)
        
        matches_found = report_df[report_df['Match Status'] == '✅ Matched']
        match_rate = (len(matches_found) / len(report_df)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Match Rate", f"{match_rate:.1f}%")
        with col2:
            st.metric("Total Records", len(report_df))
        with col3:
            st.metric("Matches Found", len(matches_found))
        with col4:
            avg_conf = matches_found['Confidence Score'].mean() * 100 if len(matches_found) > 0 else 0
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        # Display results table
        display_df = report_df.copy()
        score_cols = ['Confidence Score', 'Name Match Score', 'Address Match Score', 'District Match Score', 'State Match Score']
        for col in score_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) and x > 0 else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = report_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv,
            f"matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
        
        # Show unmatched records
        unmatched = report_df[report_df['Match Status'] == '❌ No Match']
        if len(unmatched) > 0:
            st.markdown("<h2 class='section-header'>⚠️ Unmatched Records</h2>", unsafe_allow_html=True)
            st.warning(f"{len(unmatched)} records could not be matched. Suggestions:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("💡 Try these adjustments:")
                st.markdown("""
                - Lower confidence threshold further
                - Switch to 'aggressive' strategy
                - Add more synonyms
                """)
            with col2:
                st.info("📋 Common variations to add as synonyms:")
                st.markdown("""
                - vidyalaya → school
                - kendra → center
                - nagar → city
                - gram → village
                """)
            
            with st.expander(f"View {len(unmatched)} Unmatched Records"):
                st.dataframe(unmatched[['center_name', 'district', 'state', 'address']], use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Please upload both Master Database and Input Stream files to begin matching")