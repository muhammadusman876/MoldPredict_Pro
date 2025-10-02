"""
RAG service for generating intelligent air quality recommendations
Combines knowledge base retrieval with language model generation
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import with error handling for optional dependencies
try:
    from langchain_community.llms import OpenAI  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
    from langchain.chains import LLMChain  # type: ignore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .knowledge_base import AirQualityKnowledgeBase

logger = logging.getLogger(__name__)

class AirQualityRAGService:
    """RAG service for intelligent air quality recommendations"""
    
    def __init__(self, 
                 knowledge_base_path: str = "knowledge_base/chroma_db",
                 use_llm: bool = False,
                 openai_api_key: Optional[str] = None):
        
        # Initialize knowledge base
        self.knowledge_base = AirQualityKnowledgeBase(knowledge_base_path)
        
        # Initialize LLM if available and requested
        self.use_llm = use_llm and LANGCHAIN_AVAILABLE
        self.llm = None
        
        if self.use_llm and openai_api_key:
            try:
                self.llm = OpenAI(api_key=openai_api_key, temperature=0.7)
                self._setup_prompts()
                logger.info("LLM initialized for enhanced recommendations")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.use_llm = False
        elif self.use_llm:
            logger.warning("LLM requested but OpenAI API key not provided")
            self.use_llm = False
    
    def _setup_prompts(self):
        """Setup prompt templates for different types of recommendations"""
        
        if not self.use_llm:
            return
        
        # General recommendation prompt
        self.recommendation_prompt = PromptTemplate(
            input_variables=["conditions", "knowledge", "predictions"],
            template="""
            You are an expert air quality advisor. Based on the current conditions, retrieved knowledge, and predictions, provide specific, actionable recommendations.

            Current Conditions:
            - CO2: {conditions[co2_ppm]} ppm
            - Temperature: {conditions[temperature]}°C
            - Humidity: {conditions[humidity]}%
            - Trend: {conditions[trend]}

            Predictions:
            {predictions}

            Relevant Knowledge:
            {knowledge}

            Provide 3-5 specific, actionable recommendations prioritized by urgency. Format as a numbered list with brief explanations.
            Focus on immediate actions if conditions are concerning, otherwise provide preventive measures.
            """
        )
        
        # Emergency response prompt
        self.emergency_prompt = PromptTemplate(
            input_variables=["conditions", "knowledge"],
            template="""
            URGENT AIR QUALITY ALERT
            
            Current dangerous conditions:
            - CO2: {conditions[co2_ppm]} ppm
            - Temperature: {conditions[temperature]}°C
            - Humidity: {conditions[humidity]}%
            
            Relevant safety information:
            {knowledge}
            
            Provide IMMEDIATE step-by-step emergency actions to address these dangerous air quality conditions.
            Prioritize health and safety. Use clear, direct language.
            """
        )
        
        # Setup LLM chains
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=self.recommendation_prompt)
        self.emergency_chain = LLMChain(llm=self.llm, prompt=self.emergency_prompt)
    
    def get_recommendations(self, 
                          current_conditions: Dict[str, Any],
                          predictions: Optional[Dict[str, Any]] = None,
                          context: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive recommendations based on current conditions and predictions"""
        
        co2_ppm = current_conditions.get('co2_ppm', 400)
        temperature = current_conditions.get('temperature_celsius', 20)
        humidity = current_conditions.get('humidity_percent', 50)
        trend = predictions.get('trend', 'stable') if predictions else 'stable'
        
        # Determine urgency level
        urgency = self._assess_urgency(co2_ppm, temperature, humidity)
        
        # Get relevant knowledge
        knowledge_results = self.knowledge_base.get_recommendations_for_conditions(
            co2_ppm, temperature, humidity, trend
        )
        
        # Format knowledge for context
        knowledge_text = "\n".join([
            f"- {result['metadata']['title']}: {result['content'][:200]}..."
            for result in knowledge_results[:3]
        ])
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'urgency_level': urgency,
            'conditions': current_conditions,
            'knowledge_sources': [r['metadata']['title'] for r in knowledge_results],
            'recommendations': []
        }
        
        # Generate recommendations based on availability of LLM
        if self.use_llm and self.llm:
            recommendations['recommendations'] = self._generate_llm_recommendations(
                current_conditions, predictions, knowledge_text, urgency
            )
            recommendations['generated_by'] = 'llm_rag'
        else:
            recommendations['recommendations'] = self._generate_rule_based_recommendations(
                co2_ppm, temperature, humidity, trend, knowledge_results
            )
            recommendations['generated_by'] = 'rule_based_rag'
        
        # Add knowledge context
        recommendations['knowledge_context'] = knowledge_results
        
        return recommendations
    
    def _assess_urgency(self, co2_ppm: float, temperature: float, humidity: float) -> str:
        """Assess urgency level based on conditions"""
        
        if co2_ppm > 2000 or temperature > 30 or temperature < 15 or humidity > 80:
            return 'critical'
        elif co2_ppm > 1200 or temperature > 27 or temperature < 17 or humidity > 65 or humidity < 25:
            return 'high'
        elif co2_ppm > 1000 or temperature > 25 or temperature < 19 or humidity > 60 or humidity < 30:
            return 'medium'
        else:
            return 'low'
    
    def _generate_llm_recommendations(self, 
                                    conditions: Dict[str, Any],
                                    predictions: Optional[Dict[str, Any]],
                                    knowledge: str,
                                    urgency: str) -> List[Dict[str, str]]:
        """Generate recommendations using LLM"""
        
        if not self.use_llm or not self.llm:
            return []
        
        try:
            # Format predictions for prompt
            predictions_text = "No predictions available"
            if predictions:
                predictions_text = f"Trend: {predictions.get('trend', 'unknown')}"
                if 'predicted_co2' in predictions:
                    predictions_text += f", Predicted CO2: {predictions['predicted_co2']} ppm"
            
            # Choose appropriate chain based on urgency
            if urgency == 'critical':
                response = self.emergency_chain.run(
                    conditions=conditions,
                    knowledge=knowledge
                )
            else:
                response = self.recommendation_chain.run(
                    conditions=conditions,
                    knowledge=knowledge,
                    predictions=predictions_text
                )
            
            # Parse response into structured recommendations
            recommendations = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and format
                    clean_line = line.lstrip('0123456789.-').strip()
                    if clean_line:
                        recommendations.append({
                            'action': clean_line,
                            'priority': urgency,
                            'type': 'llm_generated'
                        })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"LLM recommendation generation failed: {e}")
            # Fallback to rule-based
            return self._generate_rule_based_recommendations(
                conditions.get('co2_ppm', 400),
                conditions.get('temperature_celsius', 20),
                conditions.get('humidity_percent', 50),
                predictions.get('trend', 'stable') if predictions else 'stable',
                []
            )
    
    def _generate_rule_based_recommendations(self, 
                                           co2_ppm: float,
                                           temperature: float,
                                           humidity: float,
                                           trend: str,
                                           knowledge_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate recommendations using rule-based logic"""
        
        recommendations = []
        
        # CO2-based recommendations
        if co2_ppm > 2000:
            recommendations.append({
                'action': 'EMERGENCY: Evacuate area immediately due to dangerous CO2 levels (>2000 ppm)',
                'priority': 'critical',
                'type': 'safety'
            })
        elif co2_ppm > 1500:
            recommendations.append({
                'action': 'Open all windows and doors immediately. Turn on all exhaust fans.',
                'priority': 'critical',
                'type': 'ventilation'
            })
        elif co2_ppm > 1000:
            recommendations.append({
                'action': 'Increase ventilation immediately. Open windows if outdoor air quality permits.',
                'priority': 'high',
                'type': 'ventilation'
            })
        elif co2_ppm > 800:
            recommendations.append({
                'action': 'Improve ventilation. Check HVAC system and consider opening windows.',
                'priority': 'medium',
                'type': 'ventilation'
            })
        
        # Temperature-based recommendations
        if temperature > 27:
            recommendations.append({
                'action': f'Temperature is high ({temperature:.1f}°C). Use air conditioning or fans for cooling.',
                'priority': 'medium',
                'type': 'comfort'
            })
        elif temperature < 18:
            recommendations.append({
                'action': f'Temperature is low ({temperature:.1f}°C). Increase heating while maintaining ventilation.',
                'priority': 'medium',
                'type': 'comfort'
            })
        
        # Humidity-based recommendations
        if humidity > 65:
            recommendations.append({
                'action': f'High humidity ({humidity:.1f}%). Use dehumidifier or improve ventilation.',
                'priority': 'medium',
                'type': 'comfort'
            })
        elif humidity < 30:
            recommendations.append({
                'action': f'Low humidity ({humidity:.1f}%). Use humidifier or reduce heating.',
                'priority': 'medium',
                'type': 'comfort'
            })
        
        # Trend-based recommendations
        if trend == 'increasing':
            if co2_ppm > 600:  # Already elevated and rising
                recommendations.append({
                    'action': 'CO2 levels are rising. Take preventive ventilation action now.',
                    'priority': 'high',
                    'type': 'preventive'
                })
            else:
                recommendations.append({
                    'action': 'Monitor conditions closely as levels are trending upward.',
                    'priority': 'low',
                    'type': 'monitoring'
                })
        
        # Add knowledge-based recommendations if available
        for knowledge in knowledge_results[:2]:  # Limit to top 2 knowledge items
            if knowledge['metadata'].get('category') == 'emergency':
                recommendations.append({
                    'action': f"Based on guidelines: {knowledge['content'][:100]}...",
                    'priority': 'high',
                    'type': 'knowledge_based'
                })
        
        # Ensure we have at least one recommendation
        if not recommendations:
            recommendations.append({
                'action': 'Air quality appears normal. Continue regular monitoring and maintenance.',
                'priority': 'low',
                'type': 'maintenance'
            })
        
        return recommendations
    
    def get_emergency_response(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Get emergency response recommendations for critical conditions"""
        
        # Get emergency-specific knowledge
        emergency_knowledge = self.knowledge_base.get_knowledge_by_category('emergency')
        
        knowledge_text = "\n".join([
            f"- {item['metadata']['title']}: {item['content']}"
            for item in emergency_knowledge
        ])
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'alert_level': 'EMERGENCY',
            'conditions': conditions,
            'immediate_actions': []
        }
        
        # Generate emergency recommendations
        if self.use_llm and self.llm:
            try:
                llm_response = self.emergency_chain.run(
                    conditions=conditions,
                    knowledge=knowledge_text
                )
                
                # Parse LLM response
                actions = []
                for line in llm_response.strip().split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        clean_line = line.lstrip('0123456789.-').strip()
                        if clean_line:
                            actions.append(clean_line)
                
                response['immediate_actions'] = actions
                response['generated_by'] = 'llm_emergency'
                
            except Exception as e:
                logger.error(f"Emergency LLM generation failed: {e}")
                response['immediate_actions'] = self._get_emergency_fallback_actions(conditions)
                response['generated_by'] = 'rule_based_emergency'
        else:
            response['immediate_actions'] = self._get_emergency_fallback_actions(conditions)
            response['generated_by'] = 'rule_based_emergency'
        
        return response
    
    def _get_emergency_fallback_actions(self, conditions: Dict[str, Any]) -> List[str]:
        """Fallback emergency actions when LLM is not available"""
        
        actions = []
        co2_ppm = conditions.get('co2_ppm', 0)
        
        if co2_ppm > 5000:
            actions.extend([
                "EVACUATE the area immediately - CO2 levels are life-threatening",
                "Call emergency services if anyone feels faint or unconscious",
                "Do not re-enter until CO2 levels are below 1000 ppm"
            ])
        elif co2_ppm > 2000:
            actions.extend([
                "Leave the area immediately and get fresh air",
                "Open all windows and doors",
                "Turn on all available ventilation",
                "Do not return until levels drop significantly"
            ])
        else:
            actions.extend([
                "Open all windows and doors immediately",
                "Turn on all exhaust fans and ventilation systems",
                "Reduce occupancy in the area",
                "Monitor levels every 15 minutes"
            ])
        
        return actions
    
    def add_custom_knowledge(self, title: str, content: str, category: str = "custom") -> str:
        """Add custom knowledge to the knowledge base"""
        
        return self.knowledge_base.add_custom_knowledge(title, content, category)
    
    def search_knowledge(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base"""
        
        return self.knowledge_base.search_knowledge(query, n_results)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of the RAG service"""
        
        kb_stats = self.knowledge_base.get_knowledge_stats()
        
        return {
            'rag_service_active': True,
            'llm_available': self.use_llm,
            'knowledge_base': kb_stats,
            'capabilities': {
                'recommendations': True,
                'emergency_response': True,
                'knowledge_search': True,
                'custom_knowledge': True,
                'llm_enhanced': self.use_llm
            }
        }
