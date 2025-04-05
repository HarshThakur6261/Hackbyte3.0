from flask import Flask, request, jsonify, abort
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
import os
import logging
import datetime
import json
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FitnessRecommender:
    """Enhanced recommendation system for fitness exercises"""
    
    def __init__(self):
        """Initialize the recommender with available exercise data"""
        self.exercises = self._load_exercises()
        self.exercise_ids = np.array([ex['id'] for ex in self.exercises]) if self.exercises else np.arange(100)
        logger.info(f"Initialized recommender with {len(self.exercises)} exercises")
        
    def _load_exercises(self) -> List[Dict[str, Any]]:
        """Load exercise data from file or create dummy data"""
        try:
            # Try to load from pickle file
            if os.path.exists('all_items.pkl'):
                return joblib.load('all_items.pkl')
            # Try to load from JSON file
            elif os.path.exists('exercises.json'):
                with open('exercises.json', 'r') as f:
                    return json.load(f)
            else:
                logger.warning("No exercise data found. Creating dummy data.")
                return self._create_dummy_exercises()
        except Exception as e:
            logger.error(f"Error loading exercises: {e}")
            return self._create_dummy_exercises()
    
    def _create_dummy_exercises(self) -> List[Dict[str, Any]]:
        """Create dummy exercise data for testing"""
        exercises = []
        # Create a variety of exercise types
        categories = ['cardio', 'strength', 'flexibility', 'balance', 'endurance']
        difficulty_levels = ['easy', 'medium', 'hard']
        
        for i in range(100):
            category = categories[i % len(categories)]
            difficulty = difficulty_levels[min(i // 30, 2)]  # Distribute difficulties
            
            exercise = {
                'id': i,
                'name': f"{category.capitalize()} Exercise {i}",
                'category': category,
                'difficulty': difficulty,
                'duration': 10 + (i % 20),  # 10-30 minutes
                'calories': 50 + (i * 5),   # 50-550 calories
                'equipment_required': i % 3 == 0,  # Every third exercise needs equipment
                'description': f"This is a {difficulty} {category} exercise"
            }
            exercises.append(exercise)
        
        logger.info("Created 100 dummy exercises")
        return exercises
    
    def predict(self, user_data: Dict[str, Any], user_history: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Generate personalized exercise recommendations"""
        if not self.exercises:
            logger.warning("No exercises available for recommendation")
            return np.array([])
        
        # Initialize scores
        scores = np.ones(len(self.exercises)) * 0.5  # Start with neutral scores
        
        # 1. Adjust based on skill level
        self._adjust_for_skill_level(scores, user_data.get('skill_level', 'beginner'))
        
        # 2. Adjust based on user preferences if available
        if 'fitness_goals' in user_data:
            self._adjust_for_goals(scores, user_data['fitness_goals'])
        
        # 3. Adjust based on user history
        if user_history:
            self._adjust_for_history(scores, user_history)
        
        # 4. Adjust for activity level
        active_days = user_data.get('active_days', 0)
        if active_days > 20:  # Very active user
            # Boost harder exercises
            for i, exercise in enumerate(self.exercises):
                if exercise['difficulty'] == 'hard':
                    scores[i] += 0.2
        elif active_days < 5:  # Less active user
            # Boost easier exercises
            for i, exercise in enumerate(self.exercises):
                if exercise['difficulty'] == 'easy':
                    scores[i] += 0.2
        
        # 5. Exclude completed exercises
        if user_history and 'completed_challenges' in user_history:
            completed = user_history['completed_challenges']
            for item_id in completed:
                indices = np.where(self.exercise_ids == item_id)[0]
                if len(indices) > 0:
                    scores[indices[0]] = float('-inf')
        
        # Get top recommendations
        valid_indices = np.where(scores > float('-inf'))[0]
        if len(valid_indices) == 0:
            logger.warning("All exercises filtered out, resetting scores")
            scores = np.random.rand(len(self.exercises))
            valid_indices = np.arange(len(self.exercises))
        
        # Sort by score and get top 10
        top_indices = valid_indices[np.argsort(scores[valid_indices])[-10:][::-1]]
        return self.exercise_ids[top_indices]
    
    def _adjust_for_skill_level(self, scores: np.ndarray, skill_level: str) -> None:
        """Adjust scores based on user skill level"""
        difficulty_map = {
            'beginner': 'easy',
            'intermediate': 'medium',
            'advanced': 'hard'
        }
        
        target_difficulty = difficulty_map.get(skill_level, 'medium')
        
        for i, exercise in enumerate(self.exercises):
            if exercise['difficulty'] == target_difficulty:
                scores[i] += 0.3
            elif (skill_level == 'beginner' and exercise['difficulty'] == 'hard') or \
                 (skill_level == 'advanced' and exercise['difficulty'] == 'easy'):
                # Penalize exercises that are too hard for beginners or too easy for advanced
                scores[i] -= 0.2
    
    def _adjust_for_goals(self, scores: np.ndarray, goals: List[str]) -> None:
        """Adjust scores based on user fitness goals"""
        goal_category_map = {
            'weight_loss': 'cardio',
            'strength': 'strength',
            'flexibility': 'flexibility',
            'balance': 'balance',
            'endurance': 'endurance'
        }
        
        for goal in goals:
            if goal in goal_category_map:
                category = goal_category_map[goal]
                for i, exercise in enumerate(self.exercises):
                    if exercise['category'] == category:
                        scores[i] += 0.25
    
    def _adjust_for_history(self, scores: np.ndarray, user_history: Dict[str, Any]) -> None:
        """Adjust scores based on user history"""
        # Boost categories that user has completed with positive rewards
        if 'completed_challenges' in user_history and 'reward_history' in user_history:
            completed = user_history['completed_challenges']
            rewards = user_history['reward_history']
            
            # If we have both completed exercises and rewards
            if completed and rewards and len(completed) == len(rewards):
                # Find categories of exercises with positive rewards
                positive_categories = set()
                for i, ex_id in enumerate(completed):
                    if rewards[i] > 0:
                        for exercise in self.exercises:
                            if exercise['id'] == ex_id:
                                positive_categories.add(exercise['category'])
                                break
                
                # Boost exercises in those categories
                for category in positive_categories:
                    for i, exercise in enumerate(self.exercises):
                        if exercise['category'] == category:
                            scores[i] += 0.2

# Initialize our recommendation system
recommender = FitnessRecommender()

# In-memory storage for user interactions and history
user_history = {}
user_rewards = defaultdict(list)
user_completions = defaultdict(set)

# Try to load existing data if available
try:
    if os.path.exists('user_rewards.pkl'):
        user_rewards = joblib.load('user_rewards.pkl')
    if os.path.exists('user_completions.pkl'):
        user_completions = joblib.load('user_completions.pkl')
    if os.path.exists('user_history.pkl'):
        user_history = joblib.load('user_history.pkl')
    logger.info("Loaded existing user data")
except Exception as e:
    logger.error(f"Error loading user data: {e}")

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint to get personalized exercise recommendations"""
    try:
        data = request.json
        if not data or 'user_id' not in data:
            return jsonify({'error': 'Missing user_id in request'}), 400
        
        user_id = data['user_id']
        
        # Create user context if we have history
        user_context = None
        if user_id in user_rewards and len(user_rewards[user_id]) > 0:
            user_context = {
                'avg_reward': np.mean(user_rewards[user_id]),
                'completed_challenges': list(user_completions[user_id]),
                'reward_history': user_rewards[user_id][-5:]  # Last 5 rewards
            }
        
        # Get recommendations
        recommendations = recommender.predict(data, user_context)
        
        # Store recommendation in history
        user_history[user_id] = {
            'recommendations': recommendations,
            'context': data.get('context', {}),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Prepare detailed response
        recommendation_details = []
        for rec_id in recommendations:
            # Find the exercise details
            for exercise in recommender.exercises:
                if exercise['id'] == rec_id:
                    recommendation_details.append(exercise)
                    break
        
        return jsonify({
            'recommendations': recommendations.tolist(),
            'recommendation_details': recommendation_details,
            'has_history': user_id in user_rewards and len(user_rewards[user_id]) > 0
        })
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """Endpoint to process user feedback on exercises"""
    try:
        feedback = request.json
        if not feedback or 'user_id' not in feedback or 'challenge_id' not in feedback:
            return jsonify({'error': 'Missing required fields in feedback'}), 400
        
        user_id = feedback['user_id']
        challenge_id = feedback['challenge_id']
        
        # Calculate reward based on user interaction
        reward = calculate_reward(feedback)
        
        # Store reward in user history
        user_rewards[user_id].append(reward)
        
        # If challenge was completed, add to user's completed challenges
        if feedback.get('completed', False):
            user_completions[user_id].add(challenge_id)
        
        # Update user history with feedback
        if user_id in user_history:
            user_history[user_id]['feedback'] = feedback
        
        # Save data periodically (every 10 feedbacks)
        total_feedbacks = sum(len(rewards) for rewards in user_rewards.values())
        if total_feedbacks % 10 == 0:
            try:
                joblib.dump(user_rewards, 'user_rewards.pkl')
                joblib.dump(user_completions, 'user_completions.pkl')
                joblib.dump(user_history, 'user_history.pkl')
                logger.info("Auto-saved user data")
            except Exception as e:
                logger.error(f"Error auto-saving data: {e}")
        
        return jsonify({
            'status': 'feedback processed',
            'reward': reward,
            'completed_challenges': len(user_completions[user_id])
        })
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

def calculate_reward(feedback: Dict[str, Any]) -> float:
    """Calculate reward based on user feedback"""
    # Base reward for completion
    base_reward = 2.0 if feedback.get('completed', False) else -0.5
    
    # User rating (if available)
    rating_reward = feedback.get('user_rating', 0) * 0.5
    
    # Time-based factors
    time_spent = feedback.get('completion_time', 0)
    if time_spent > 0:
        if time_spent < 10:  # Too quick might mean it was too easy
            time_penalty = -0.2
        elif time_spent > 3600:  # More than an hour might be frustrating
            time_penalty = -0.5
        else:
            # Optimal time range gets a bonus
            time_penalty = 0.3
    else:
        time_penalty = 0
    
    # Engagement factors
    attempts = feedback.get('attempts', 1)
    engagement_reward = min(0.1 * attempts, 0.5)  # Cap at 0.5
    
    # Difficulty adjustment
    difficulty = feedback.get('difficulty', 'medium')
    difficulty_multiplier = {
        'easy': 0.8,
        'medium': 1.0,
        'hard': 1.5
    }.get(difficulty, 1.0)
    
    # Calculate total reward
    total_reward = (base_reward + rating_reward + time_penalty + engagement_reward) * difficulty_multiplier
    
    return total_reward

@app.route('/user_progress/<user_id>', methods=['GET'])
def get_user_progress(user_id: str):
    """Endpoint to retrieve user's fitness progress and history"""
    try:
        if user_id not in user_completions:
            return jsonify({'error': 'User not found'}), 404
        
        completed_challenges = list(user_completions[user_id])
        avg_reward = np.mean(user_rewards[user_id]) if user_rewards[user_id] else 0
        
        # Get details of completed exercises
        completed_details = []
        for ex_id in completed_challenges:
            for exercise in recommender.exercises:
                if exercise['id'] == ex_id:
                    completed_details.append(exercise)
                    break
        
        # Calculate stats
        total_duration = sum(ex.get('duration', 0) for ex in completed_details)
        total_calories = sum(ex.get('calories', 0) for ex in completed_details)
        
        # Get category distribution
        categories = {}
        for ex in completed_details:
            category = ex.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        
        return jsonify({
            'user_id': user_id,
            'completed_challenges': completed_challenges,
            'completed_details': completed_details,
            'completion_count': len(completed_challenges),
            'average_reward': avg_reward,
                    'recent_rewards': user_rewards[user_id][-5:] if user_rewards[user_id] else [],
                    'stats': {
                        'total_duration': total_duration,
                        'total_calories': total_calories,
                        'category_distribution': categories
                    }
                })
    except Exception as e:
        logger.error(f"Error in user_progress endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Failed to fetch user progress',
            'details': str(e)
        }), 500@app.route('/save_data', methods=['POST'])
def save_data():
    """Admin endpoint to save the current user data"""
    try:
        # Check for admin authentication
        if request.headers.get('X-Admin-Key') != 'your-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        
        # Save user data
        joblib.dump(user_rewards, 'user_rewards.pkl')
        joblib.dump(user_completions, 'user_completions.pkl')
        joblib.dump(user_history, 'user_history.pkl')
        
        # Also save exercises if they were generated
        if recommender.exercises and all(isinstance(ex, dict) for ex in recommender.exercises):
            with open('exercises.json', 'w') as f:
                json.dump(recommender.exercises, f)
        
        logger.info("Data saved successfully")
        return jsonify({'status': 'data saved successfully'})
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return jsonify({'error': f'Failed to save data: {str(e)}'}), 500

@app.route('/exercises', methods=['GET'])
def get_exercises():
    """Endpoint to get all available exercises"""
    try:
        # Optional filtering
        category = request.args.get('category')
        difficulty = request.args.get('difficulty')
        
        exercises = recommender.exercises
        
        # Apply filters if provided
        if category:
            exercises = [ex for ex in exercises if ex.get('category') == category]
        if difficulty:
            exercises = [ex for ex in exercises if ex.get('difficulty') == difficulty]
        
        return jsonify({
            'exercises': exercises,
            'count': len(exercises)
        })
    except Exception as e:
        logger.error(f"Error in exercises endpoint: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/exercise/<int:exercise_id>', methods=['GET'])
def get_exercise(exercise_id):
    """Endpoint to get details of a specific exercise"""
    try:
        for exercise in recommender.exercises:
            if exercise['id'] == exercise_id:
                return jsonify(exercise)
        
        return jsonify({'error': 'Exercise not found'}), 404
    except Exception as e:
        logger.error(f"Error in exercise endpoint: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'exercises_count': len(recommender.exercises),
        'users_count': len(user_history)
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Set host to 0.0.0.0 to make it accessible from outside the container if needed
    app.run(host='0.0.0.0', port=5000, debug=True)