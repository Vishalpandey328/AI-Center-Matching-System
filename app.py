import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
import time
import json
import pickle
from datetime import datetime, timedelta
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
import torch

# ------------------------
# PAGE CONFIG
st.set_page_config(
    page_title="AI Powered Center Matching System with Self-Learning",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [Your existing CSS styles remain the same]

# --------------------------------------------------
# REINFORCEMENT LEARNING MODEL
# --------------------------------------------------

class ReinforcementLearningMatcher:
    """Self-learning matcher using reinforcement learning principles"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.learning_rate = learning_rate  # How fast to learn
        self.discount_factor = discount_factor  # Future reward importance
        self.exploration_rate = exploration_rate  # Explore vs exploit
        self.q_table = {}  # State-action values
        self.feature_weights = {
            'name_weight': 0.35,
            'address_weight': 0.25,
            'district_weight': 0.20,
            'state_weight': 0.10,
            'vector_weight': 0.10
        }
        self.successful_patterns = defaultdict(int)
        self.failure_patterns = defaultdict(int)
        self.match_history = []
        self.model_file = "rl_model.pkl"
        
        # Load existing model if available
        self.load_model()
    
    def get_state(self, input_text, candidate_text, similarity_scores):
        """Create a state representation from matching features"""
        # Create a hashable state representation
        features = (
            round(similarity_scores.get('name', 0), 2),
            round(similarity_scores.get('address', 0), 2),
            round(similarity_scores.get('district', 0), 2),
            round(similarity_scores.get('state', 0), 2),
            round(similarity_scores.get('vector', 0), 2),
            len(input_text.split()),
            len(candidate_text.split())
        )
        return str(features)
    
    def get_action(self, state, available_actions=['accept', 'reject', 'adjust']):
        """Choose action using epsilon-greedy policy"""
        import random
        
        # Exploration: try random action
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        # Exploitation: choose best action from Q-table
        if state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)
        return 'accept'  # Default action
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning algorithm"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in ['accept', 'reject', 'adjust']}
        
        # Get current Q-value
        current_q = self.q_table[state].get(action, 0)
        
        # Get maximum future Q-value
        max_future_q = 0
        if next_state in self.q_table:
            max_future_q = max(self.q_table[next_state].values())
        
        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )
        
        # Update Q-table
        self.q_table[state][action] = new_q
    
    def update_weights(self, successful_match, features_used):
        """Dynamically adjust feature weights based on success patterns"""
        # Analyze which features contributed most to successful matches
        for feature, value in features_used.items():
            if value > 0.8:  # High similarity
                self.feature_weights[f'{feature}_weight'] = min(
                    0.5, 
                    self.feature_weights[f'{feature}_weight'] + self.learning_rate * 0.05
                )
            elif value < 0.3:  # Low similarity but still matched
                self.feature_weights[f'{feature}_weight'] = max(
                    0.05,
                    self.feature_weights[f'{feature}_weight'] - self.learning_rate * 0.03
                )
        
        # Normalize weights
        total = sum(self.feature_weights.values())
        for key in self.feature_weights:
            self.feature_weights[key] /= total
    
    def learn_from_match(self, input_text, matched_text, was_correct, confidence, 
                        similarity_scores, user_feedback=None):
        """Main learning function - learns from each match attempt"""
        
        state = self.get_state(input_text, matched_text, similarity_scores)
        
        # Calculate reward based on outcome
        reward = 0
        if was_correct:
            reward = 1.0 + (confidence - 0.7)  # Higher reward for high confidence matches
            self.successful_patterns[state] += 1
            
            # Store successful pattern
            self.match_history.append({
                'timestamp': datetime.now(),
                'input': input_text[:100],
                'matched': matched_text[:100],
                'confidence': confidence,
                'success': True,
                'similarity_scores': similarity_scores
            })
            
            # Update weights based on successful match
            self.update_weights(True, similarity_scores)
            
        else:
            reward = -0.5 - (1 - confidence)  # Penalize wrong matches
            self.failure_patterns[state] += 1
            
            self.match_history.append({
                'timestamp': datetime.now(),
                'input': input_text[:100],
                'matched': matched_text[:100],
                'confidence': confidence,
                'success': False,
                'similarity_scores': similarity_scores
            })
        
        # Apply user feedback if available
        if user_feedback:
            if user_feedback == 'thumbs_up':
                reward += 0.3
            elif user_feedback == 'thumbs_down':
                reward -= 0.5
            elif user_feedback == 'correct_match':
                reward += 0.5
        
        # Update Q-value
        next_state = self.get_state(input_text, matched_text, similarity_scores)
        self.update_q_value(state, 'accept', reward, next_state)
        
        # Reduce exploration rate over time (more exploitation as we learn)
        self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
        
        # Save model after significant learning
        if len(self.match_history) % 10 == 0:
            self.save_model()
    
    def get_adjusted_threshold(self, base_threshold=0.70):
        """Dynamically adjust threshold based on learning"""
        if len(self.match_history) < 10:
            return base_threshold
        
        # Calculate success rate from recent matches
        recent_matches = self.match_history[-50:]
        if not recent_matches:
            return base_threshold
        
        success_rate = sum(1 for m in recent_matches if m['success']) / len(recent_matches)
        
        # Adjust threshold based on success rate
        if success_rate > 0.9:
            return base_threshold + 0.05  # More strict
        elif success_rate < 0.7:
            return base_threshold - 0.05  # More lenient
        
        return base_threshold
    
    def predict_match_quality(self, input_text, candidate_text, similarity_scores):
        """Predict if a match will be successful based on past learning"""
        state = self.get_state(input_text, candidate_text, similarity_scores)
        
        if state in self.q_table:
            q_value = self.q_table[state].get('accept', 0)
            # Convert Q-value to probability (sigmoid-like)
            probability = 1 / (1 + np.exp(-q_value))
            return probability
        return 0.5  # Default neutral prediction
    
    def save_model(self):
        """Save the learned model to disk"""
        model_data = {
            'q_table': self.q_table,
            'feature_weights': self.feature_weights,
            'successful_patterns': dict(self.successful_patterns),
            'failure_patterns': dict(self.failure_patterns),
            'match_history': self.match_history[-1000:],  # Keep last 1000
            'exploration_rate': self.exploration_rate
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Load previously learned model"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.q_table = model_data.get('q_table', {})
                    self.feature_weights = model_data.get('feature_weights', self.feature_weights)
                    self.successful_patterns = defaultdict(int, model_data.get('successful_patterns', {}))
                    self.failure_patterns = defaultdict(int, model_data.get('failure_patterns', {}))
                    self.match_history = model_data.get('match_history', [])
                    self.exploration_rate = model_data.get('exploration_rate', 0.1)
            except:
                pass
    
    def get_learning_stats(self):
        """Get statistics about the learning progress"""
        total_matches = len(self.match_history)
        successful = sum(1 for m in self.match_history if m['success'])
        
        return {
            'total_learned_matches': total_matches,
            'success_rate': successful / total_matches if total_matches > 0 else 0,
            'unique_patterns': len(self.q_table),
            'exploration_rate': self.exploration_rate,
            'feature_weights': self.feature_weights,
            'recent_success_rate': self._get_recent_success_rate()
        }
    
    def _get_recent_success_rate(self, n=50):
        """Calculate success rate of recent matches"""
        recent = self.match_history[-n:]
        if not recent:
            return 0
        return sum(1 for m in recent if m['success']) / len(recent)

# --------------------------------------------------
# ENHANCED TEXT CLEANING WITH DYNAMIC WEIGHTS
# --------------------------------------------------

def enhanced_clean_text(text, synonyms_df):
    """Enhanced text cleaning with better synonym handling"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove special characters but keep important ones
    text = re.sub(r"[^\w\s\-/]", " ", text)
    
    # Expand common abbreviations
    abbreviations = {
        r'\bvidhan sabha\b': 'vidhansabha',
        r'\bnagar nigam\b': 'nagarnigam',
        r'\bnagar palika\b': 'nagarparishad',
        r'\bgram panchayat\b': 'grampanchayat',
        r'\bst\b': 'saint',
        r'\bmt\b': 'mount',
        r'\bnr\b': 'near',
        r'\b\&\b': 'and',
        r'\bph\b': 'public high',
        r'\bghs\b': 'government high school',
        r'\bgps\b': 'government primary school',
        r'\bggic\b': 'government girls inter college',
        r'\bgic\b': 'government inter college',
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
    
    # Apply custom synonyms
    for _, row in synonyms_df.iterrows():
        word = str(row["word"]).lower()
        replacement = str(row["replacement"]).lower()
        text = re.sub(rf'\b{word}\b', replacement, text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --------------------------------------------------
# IMPROVED MATCHING WITH REINFORCEMENT LEARNING
# --------------------------------------------------

def match_with_rl(input_data, master_df, model, index, synonyms_df, rl_model, confidence_threshold=0.70):
    """Matching function that uses reinforcement learning for improvement"""
    
    results = []
    match_details = []
    learning_opportunities = []
    
    # Prepare master data
    master_df['clean_name'] = master_df['center_name'].apply(
        lambda x: enhanced_clean_text(x, synonyms_df)
    )
    master_df['clean_address'] = master_df['address'].apply(
        lambda x: enhanced_clean_text(x, synonyms_df)
    )
    master_df['clean_district'] = master_df['district'].apply(
        lambda x: enhanced_clean_text(x, synonyms_df)
    )
    master_df['clean_state'] = master_df['state'].apply(
        lambda x: enhanced_clean_text(x, synonyms_df)
    )
    
    # Get dynamically adjusted threshold from RL model
    adjusted_threshold = rl_model.get_adjusted_threshold(confidence_threshold)
    
    total = len(input_data)
    progress_bar = st.progress(0)
    
    for idx, row in input_data.iterrows():
        progress_bar.progress((idx + 1) / total)
        
        # Clean input text
        clean_name = enhanced_clean_text(row['center_name'], synonyms_df)
        clean_district = enhanced_clean_text(row['district'], synonyms_df)
        clean_state = enhanced_clean_text(row['state'], synonyms_df)
        clean_address = enhanced_clean_text(row['address'], synonyms_df)
        
        query_text = f"{clean_name} {clean_district} {clean_state} {clean_address}"
        
        # Generate query embedding
        query_embedding = model.encode([query_text])
        query_embedding = normalize(query_embedding.astype(np.float32))
        
        # Vector search
        distances, indices = index.search(query_embedding, 15)
        
        # Score candidates
        scored_candidates = []
        
        for master_idx, distance in zip(indices[0], distances[0]):
            if master_idx < len(master_df):
                master_row = master_df.iloc[master_idx]
                
                # Calculate similarity scores
                name_score = fuzz.token_set_ratio(clean_name, master_row['clean_name']) / 100
                address_score = fuzz.token_set_ratio(clean_address, master_row['clean_address']) / 100
                district_score = fuzz.ratio(clean_district, master_row['clean_district']) / 100
                state_score = fuzz.ratio(clean_state, master_row['clean_state']) / 100
                vector_score = 1 / (1 + distance)
                
                similarity_scores = {
                    'name': name_score,
                    'address': address_score,
                    'district': district_score,
                    'state': state_score,
                    'vector': vector_score
                }
                
                # Use RL weights for final score
                final_score = (
                    rl_model.feature_weights['name_weight'] * name_score +
                    rl_model.feature_weights['address_weight'] * address_score +
                    rl_model.feature_weights['district_weight'] * district_score +
                    rl_model.feature_weights['state_weight'] * state_score +
                    rl_model.feature_weights['vector_weight'] * vector_score
                )
                
                # Predict match quality using RL
                predicted_quality = rl_model.predict_match_quality(
                    query_text, 
                    master_row['clean_name'], 
                    similarity_scores
                )
                
                scored_candidates.append({
                    'master_id': master_row['center_id'],
                    'master_name': master_row['center_name'],
                    'score': final_score,
                    'predicted_quality': predicted_quality,
                    'similarity_scores': similarity_scores,
                    'master_row': master_row
                })
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply threshold
        if scored_candidates and scored_candidates[0]['score'] >= adjusted_threshold:
            best = scored_candidates[0]
            results.append(best['master_name'])
            match_details.append({
                'master_id': best['master_id'],
                'confidence': best['score'],
                'predicted_quality': best['predicted_quality'],
                'similarity_scores': best['similarity_scores'],
                'matched_text': best['master_name']
            })
            
            # Store for potential learning
            learning_opportunities.append({
                'input_text': query_text,
                'matched_text': best['master_name'],
                'confidence': best['score'],
                'similarity_scores': best['similarity_scores'],
                'row_index': idx
            })
        else:
            results.append("⚡ No Match")
            match_details.append({
                'master_id': 'NULL',
                'confidence': scored_candidates[0]['score'] if scored_candidates else 0,
                'predicted_quality': 0,
                'similarity_scores': {},
                'matched_text': None
            })
    
    progress_bar.empty()
    return results, match_details, learning_opportunities

# --------------------------------------------------
# USER FEEDBACK COMPONENT
# --------------------------------------------------

def feedback_component(row_idx, match_info):
    """Component to collect user feedback for learning"""
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("👍", key=f"thumb_up_{row_idx}"):
            return "thumbs_up"
    with col2:
        if st.button("👎", key=f"thumb_down_{row_idx}"):
            return "thumbs_down"
    with col3:
        if st.button("✓ Correct Match", key=f"correct_{row_idx}"):
            return "correct_match"
    
    return None

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

# Initialize variables
master_file = None
input_file = None

# Initialize Reinforcement Learning Model
@st.cache_resource
def init_rl_model():
    return ReinforcementLearningMatcher(learning_rate=0.1, exploration_rate=0.1)

rl_model = init_rl_model()

# File upload section
st.markdown("<h2 class='section-header'>Data Upload</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    master_file = st.file_uploader("Upload Master File", type=["xlsx", "csv"], key="master")

with col2:
    input_file = st.file_uploader("Upload Input File", type=["xlsx", "csv"], key="input")

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-header'>🎯 Reinforcement Learning Settings</div>", unsafe_allow_html=True)
    
    model_option = st.selectbox(
        "Choose Model",
        [
            "multi-qa-mpnet-base-dot-v1",
            "all-mpnet-base-v2",
            "BAAI/bge-large-en-v1.5",
            "all-MiniLM-L6-v2"
        ],
        index=0
    )
    
    confidence_threshold = st.slider(
        "Base Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05
    )
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Learning Statistics
    st.markdown("<div class='sidebar-header'>📊 Learning Statistics</div>", unsafe_allow_html=True)
    
    learning_stats = rl_model.get_learning_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Matches Learned", learning_stats['total_learned_matches'])
    with col2:
        st.metric("Success Rate", f"{learning_stats['success_rate']*100:.1f}%")
    
    st.metric("Unique Patterns", learning_stats['unique_patterns'])
    st.metric("Exploration Rate", f"{learning_stats['exploration_rate']*100:.1f}%")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Feature Weights (Dynamically adjusted)
    st.markdown("<div class='sidebar-header'>⚖️ Learned Feature Weights</div>", unsafe_allow_html=True)
    
    for feature, weight in learning_stats['feature_weights'].items():
        st.progress(weight, text=f"{feature.replace('_weight', '')}: {weight:.2f}")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    if st.button("🔄 Reset Learning Model", use_container_width=True):
        rl_model = ReinforcementLearningMatcher()
        st.success("Model reset successfully!")
        st.rerun()
    
    if st.button("💾 Save Learning Model", use_container_width=True):
        rl_model.save_model()
        st.success("Model saved!")
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Synonym management (simplified)
    st.markdown("<div class='sidebar-header'>📚 Synonyms</div>", unsafe_allow_html=True)
    
    synonym_file = "synonyms.csv"
    if os.path.exists(synonym_file):
        synonyms_df = pd.read_csv(synonym_file)
    else:
        synonyms_df = pd.DataFrame({
            "word": ["govt", "rajkiya", "mahila", "balika", "balak", "pg", "inter"],
            "replacement": ["government", "government", "girls", "girls", "boys", "postgraduate", "intermediate"]
        })
    
    st.info(f"Loaded {len(synonyms_df)} synonyms")

# Main processing
if master_file and input_file:
    try:
        # Load files
        if master_file.name.endswith(".csv"):
            master_df = pd.read_csv(master_file)
        else:
            master_df = pd.read_excel(master_file)
        
        if input_file.name.endswith(".csv"):
            input_df = pd.read_csv(input_file)
        else:
            input_df = pd.read_excel(input_file)
        
        st.success(f"✅ Loaded {len(master_df)} master records and {len(input_df)} input records")
        
        # Load model
        @st.cache_resource
        def load_embedding_model(model_name):
            return SentenceTransformer(model_name)
        
        with st.spinner(f"Loading {model_option}..."):
            model = load_embedding_model(model_option)
        
        # Process master data
        with st.spinner("Processing master data and generating embeddings..."):
            master_df['clean_text'] = master_df.apply(
                lambda x: enhanced_clean_text(
                    f"{x['center_name']} {x['district']} {x['state']} {x['address']}", 
                    synonyms_df
                ), 
                axis=1
            )
            
            embeddings = model.encode(master_df['clean_text'].tolist(), show_progress_bar=True)
            embeddings = normalize(embeddings.astype(np.float32))
            
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)
        
        # Perform matching with RL
        with st.spinner("Matching with Reinforcement Learning..."):
            results, match_details, learning_opportunities = match_with_rl(
                input_df, master_df, model, index, synonyms_df, 
                rl_model, confidence_threshold
            )
        
        # Add results to dataframe
        input_df['Matched Center'] = results
        input_df['Confidence Score'] = [d['confidence'] for d in match_details]
        input_df['Predicted Quality'] = [d['predicted_quality'] for d in match_details]
        input_df['Master ID'] = [d['master_id'] for d in match_details]
        
        # Display results
        st.markdown("<h2 class='section-header'>Matching Results</h2>", unsafe_allow_html=True)
        
        # Summary metrics
        matches_found = input_df[input_df['Matched Center'] != "⚡ No Match"]
        match_rate = (len(matches_found) / len(input_df)) * 100 if len(input_df) > 0 else 0
        avg_confidence = matches_found['Confidence Score'].mean() * 100 if len(matches_found) > 0 else 0
        avg_predicted_quality = matches_found['Predicted Quality'].mean() * 100 if len(matches_found) > 0 else 0
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Match Rate", f"{match_rate:.1f}%")
        with metric_col2:
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        with metric_col3:
            st.metric("Predicted Quality", f"{avg_predicted_quality:.1f}%")
        with metric_col4:
            st.metric("Total Matches", len(matches_found))
        
        # Interactive results table with feedback
        st.markdown("### Results with Feedback (Click to help AI learn)")
        
        for idx, row in input_df.iterrows():
            with st.expander(f"Record {idx+1}: {row['center_name']} → {row['Matched Center']} (Confidence: {row['Confidence Score']*100:.1f}%)"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Input:** {row['center_name']}, {row['district']}, {row['state']}")
                    st.write(f"**Matched:** {row['Matched Center']}")
                    st.write(f"**Confidence:** {row['Confidence Score']:.2%}")
                    st.write(f"**AI Prediction:** {row['Predicted Quality']:.2%}")
                
                with col2:
                    if row['Matched Center'] != "⚡ No Match":
                        feedback = feedback_component(idx, row['Matched Center'])
                        if feedback:
                            # Learn from user feedback
                            match_detail = match_details[idx]
                            rl_model.learn_from_match(
                                input_text=f"{row['center_name']} {row['district']} {row['state']}",
                                matched_text=row['Matched Center'],
                                was_correct=True,
                                confidence=row['Confidence Score'],
                                similarity_scores=match_detail.get('similarity_scores', {}),
                                user_feedback=feedback
                            )
                            st.success("✅ Thanks! AI learned from your feedback")
                            st.rerun()
        
        # Learning summary
        st.markdown("### 📈 Learning Progress")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total Learning Iterations:** {learning_stats['total_learned_matches']}")
            st.info(f"**Success Rate:** {learning_stats['success_rate']*100:.1f}%")
            st.info(f"**Patterns Learned:** {learning_stats['unique_patterns']}")
        
        with col2:
            st.warning(f"**Exploration Rate:** {learning_stats['exploration_rate']*100:.1f}%")
            st.success(f"**Recent Success:** {learning_stats['recent_success_rate']*100:.1f}%")
        
        # Batch learning option
        st.markdown("### 🧠 Batch Learning")
        if st.button("Learn from All Successful Matches", use_container_width=True):
            with st.spinner("AI is learning from all matches..."):
                for idx, row in input_df.iterrows():
                    if row['Matched Center'] != "⚡ No Match" and row['Confidence Score'] > 0.8:
                        match_detail = match_details[idx]
                        rl_model.learn_from_match(
                            input_text=f"{row['center_name']} {row['district']} {row['state']}",
                            matched_text=row['Matched Center'],
                            was_correct=True,
                            confidence=row['Confidence Score'],
                            similarity_scores=match_detail.get('similarity_scores', {})
                        )
                st.success("✅ AI has learned from all successful matches!")
                st.rerun()
        
        # Download button
        csv = input_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            csv,
            f"rl_matching_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.exception(e)

else:
    st.info("👈 Please upload both Master Database and Input Stream files to begin matching")