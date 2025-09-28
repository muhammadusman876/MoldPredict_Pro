"""
LangGraph workflow for intelligent air quality analysis
Multi-agent system: sensor → predictor → knowledge → advisor
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

# Import with error handling for optional dependencies
try:
    from langgraph.graph import StateGraph, START, END  # type: ignore
    from langgraph.graph.message import add_messages  # type: ignore
    from typing_extensions import TypedDict  # type: ignore
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    TypedDict = dict  # Fallback

from ..ml.ml_service import AirQualityMLService
from ..rag.rag_service import AirQualityRAGService

logger = logging.getLogger(__name__)

# State definition for the workflow
class AirQualityState(TypedDict):
    """State structure for the air quality analysis workflow"""
    # Input data
    sensor_data: Dict[str, Any]
    raw_readings: List[Dict[str, Any]]
    
    # Analysis results
    current_analysis: Dict[str, Any]
    predictions: Dict[str, Any]
    knowledge_context: List[Dict[str, Any]]
    
    # Final outputs
    recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    action_plan: Dict[str, Any]
    
    # Workflow metadata
    workflow_step: str
    timestamp: str
    errors: List[str]

class AirQualityWorkflow:
    """LangGraph workflow for comprehensive air quality analysis"""
    
    def __init__(self, 
                 ml_service: Optional[AirQualityMLService] = None,
                 rag_service: Optional[AirQualityRAGService] = None,
                 models_dir: str = "data/models"):
        
        # Initialize services
        self.ml_service = ml_service or AirQualityMLService(models_dir)
        self.rag_service = rag_service or AirQualityRAGService()
        
        # Initialize workflow graph if LangGraph is available
        self.graph = None
        if LANGGRAPH_AVAILABLE:
            self._build_workflow_graph()
        else:
            logger.warning("LangGraph not available. Using simplified workflow.")
    
    def _build_workflow_graph(self):
        """Build the LangGraph workflow"""
        
        if not LANGGRAPH_AVAILABLE:
            logger.info("LangGraph not available - using simplified workflow")
            return
        
        try:
            # Create workflow graph
            workflow = StateGraph(AirQualityState)  # type: ignore
            
            # Add nodes (agents)
            workflow.add_node("sensor_analyzer", self._sensor_analyzer_agent)
            workflow.add_node("predictor_agent", self._predictor_agent)
            workflow.add_node("knowledge_agent", self._knowledge_agent)
            workflow.add_node("advisor_agent", self._advisor_agent)
            workflow.add_node("risk_assessor", self._risk_assessor_agent)
            
            # Define workflow edges
            workflow.add_edge(START, "sensor_analyzer")  # type: ignore
            workflow.add_edge("sensor_analyzer", "predictor_agent")
            workflow.add_edge("predictor_agent", "knowledge_agent")
            workflow.add_edge("knowledge_agent", "advisor_agent")
            workflow.add_edge("advisor_agent", "risk_assessor")
            workflow.add_edge("risk_assessor", END)  # type: ignore
            
            # Compile the graph
            self.graph = workflow.compile()
            
            logger.info("LangGraph workflow initialized with 5 agents")
            
        except Exception as e:
            logger.warning(f"Failed to build LangGraph workflow: {e}. Using simplified workflow.")
            self.graph = None
    
    def _sensor_analyzer_agent(self, state: AirQualityState) -> AirQualityState:
        """Agent 1: Analyze sensor data and current conditions"""
        
        logger.debug("Sensor Analyzer Agent: Processing sensor data")
        
        try:
            # Extract current conditions from raw readings
            readings = state.get('raw_readings', [])
            
            if not readings:
                state['errors'].append("No sensor readings provided")
                return state
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(readings)
            
            # Analyze current conditions
            latest_reading = df.iloc[-1] if len(df) > 0 else {}
            
            analysis = {
                'current_co2': float(latest_reading.get('co2_ppm', 0)),
                'current_temperature': float(latest_reading.get('temperature_celsius', 20)),
                'current_humidity': float(latest_reading.get('humidity_percent', 50)),
                'data_quality': self._assess_data_quality(df),
                'trend_indicators': self._calculate_trends(df),
                'readings_count': len(readings),
                'time_span_hours': self._calculate_time_span(readings)
            }
            
            # Health status assessment
            analysis['health_status'] = self._assess_health_status(
                analysis['current_co2'],
                analysis['current_temperature'],
                analysis['current_humidity']
            )
            
            state['current_analysis'] = analysis
            state['workflow_step'] = 'sensor_analysis_complete'
            
            logger.debug(f"Sensor analysis complete: CO2={analysis['current_co2']}, Health={analysis['health_status']}")
            
        except Exception as e:
            error_msg = f"Sensor analyzer error: {str(e)}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
        
        return state
    
    def _predictor_agent(self, state: AirQualityState) -> AirQualityState:
        """Agent 2: Generate predictions using ML models"""
        
        logger.debug("Predictor Agent: Generating predictions")
        
        try:
            readings = state.get('raw_readings', [])
            
            if not readings:
                state['errors'].append("No readings for prediction")
                return state
            
            # Convert to DataFrame
            df = pd.DataFrame(readings)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
            
            # Get comprehensive predictions
            predictions = self.ml_service.get_comprehensive_prediction(df)
            
            # Enhance predictions with trend analysis
            predictions['trend_analysis'] = self._analyze_prediction_trends(predictions)
            predictions['confidence_assessment'] = self._assess_prediction_confidence(predictions)
            
            state['predictions'] = predictions
            state['workflow_step'] = 'predictions_complete'
            
            logger.debug(f"Predictions generated: {predictions.get('predictions', {}).keys()}")
            
        except Exception as e:
            error_msg = f"Predictor agent error: {str(e)}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
            # Provide fallback predictions
            state['predictions'] = {
                'current_conditions': state.get('current_analysis', {}),
                'predictions': {},
                'recommendations': []
            }
        
        return state
    
    def _knowledge_agent(self, state: AirQualityState) -> AirQualityState:
        """Agent 3: Retrieve relevant knowledge and guidelines"""
        
        logger.debug("Knowledge Agent: Retrieving relevant knowledge")
        
        try:
            current_analysis = state.get('current_analysis', {})
            predictions = state.get('predictions', {})
            
            # Build context for knowledge retrieval
            conditions = {
                'co2_ppm': current_analysis.get('current_co2', 400),
                'temperature_celsius': current_analysis.get('current_temperature', 20),
                'humidity_percent': current_analysis.get('current_humidity', 50),
                'trend': predictions.get('trend_analysis', {}).get('overall_trend', 'stable')
            }
            
            # Get relevant knowledge
            knowledge_results = self.rag_service.get_recommendations(
                conditions, predictions
            )
            
            # Categorize knowledge by urgency and type
            categorized_knowledge = self._categorize_knowledge(knowledge_results)
            
            state['knowledge_context'] = categorized_knowledge['knowledge_context']
            state['workflow_step'] = 'knowledge_retrieval_complete'
            
            logger.debug(f"Knowledge retrieved: {len(state['knowledge_context'])} relevant documents")
            
        except Exception as e:
            error_msg = f"Knowledge agent error: {str(e)}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
            state['knowledge_context'] = []
        
        return state
    
    def _advisor_agent(self, state: AirQualityState) -> AirQualityState:
        """Agent 4: Generate comprehensive recommendations"""
        
        logger.debug("Advisor Agent: Generating recommendations")
        
        try:
            current_analysis = state.get('current_analysis', {})
            predictions = state.get('predictions', {})
            knowledge_context = state.get('knowledge_context', [])
            
            # Generate comprehensive recommendations
            recommendations = []
            
            # Immediate action recommendations
            immediate_actions = self._generate_immediate_actions(current_analysis)
            recommendations.extend(immediate_actions)
            
            # Predictive recommendations based on ML predictions
            predictive_actions = self._generate_predictive_actions(predictions)
            recommendations.extend(predictive_actions)
            
            # Knowledge-based recommendations
            knowledge_actions = self._generate_knowledge_actions(knowledge_context)
            recommendations.extend(knowledge_actions)
            
            # Prioritize and deduplicate recommendations
            prioritized_recommendations = self._prioritize_recommendations(recommendations)
            
            state['recommendations'] = prioritized_recommendations
            state['workflow_step'] = 'recommendations_complete'
            
            logger.debug(f"Generated {len(prioritized_recommendations)} prioritized recommendations")
            
        except Exception as e:
            error_msg = f"Advisor agent error: {str(e)}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
            state['recommendations'] = []
        
        return state
    
    def _risk_assessor_agent(self, state: AirQualityState) -> AirQualityState:
        """Agent 5: Assess risks and create action plan"""
        
        logger.debug("Risk Assessor Agent: Creating risk assessment and action plan")
        
        try:
            current_analysis = state.get('current_analysis', {})
            predictions = state.get('predictions', {})
            recommendations = state.get('recommendations', [])
            
            # Assess current and predicted risks
            risk_assessment = {
                'current_risk_level': self._assess_current_risk(current_analysis),
                'predicted_risk_level': self._assess_predicted_risk(predictions),
                'risk_factors': self._identify_risk_factors(current_analysis, predictions),
                'mitigation_urgency': self._determine_urgency(current_analysis, predictions)
            }
            
            # Create structured action plan
            action_plan = {
                'immediate_actions': [r for r in recommendations if r.get('priority') in ['critical', 'high']],
                'short_term_actions': [r for r in recommendations if r.get('priority') == 'medium'],
                'long_term_actions': [r for r in recommendations if r.get('priority') == 'low'],
                'monitoring_plan': self._create_monitoring_plan(current_analysis, predictions),
                'escalation_triggers': self._define_escalation_triggers(current_analysis)
            }
            
            state['risk_assessment'] = risk_assessment
            state['action_plan'] = action_plan
            state['workflow_step'] = 'workflow_complete'
            
            logger.debug(f"Risk assessment complete: {risk_assessment['current_risk_level']} current risk")
            
        except Exception as e:
            error_msg = f"Risk assessor error: {str(e)}"
            logger.error(error_msg)
            state['errors'].append(error_msg)
        
        return state
    
    def analyze_air_quality(self, readings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main entry point for air quality analysis"""
        
        # Initialize state
        initial_state: AirQualityState = {
            'sensor_data': {},
            'raw_readings': readings,
            'current_analysis': {},
            'predictions': {},
            'knowledge_context': [],
            'recommendations': [],
            'risk_assessment': {},
            'action_plan': {},
            'workflow_step': 'initialized',
            'timestamp': datetime.now().isoformat(),
            'errors': []
        }
        
        if LANGGRAPH_AVAILABLE and self.graph:
            # Use LangGraph workflow
            try:
                result = self.graph.invoke(initial_state)
                return self._format_workflow_result(result)
            except Exception as e:
                logger.error(f"LangGraph workflow failed: {e}")
                # Fallback to simplified workflow
                return self._run_simplified_workflow(initial_state)
        else:
            # Use simplified workflow
            return self._run_simplified_workflow(initial_state)
    
    def _run_simplified_workflow(self, state: AirQualityState) -> Dict[str, Any]:
        """Simplified workflow without LangGraph"""
        
        logger.info("Running simplified workflow")
        
        # Run agents sequentially
        state = self._sensor_analyzer_agent(state)
        state = self._predictor_agent(state)
        state = self._knowledge_agent(state)
        state = self._advisor_agent(state)
        state = self._risk_assessor_agent(state)
        
        return self._format_workflow_result(state)
    
    def _format_workflow_result(self, state: AirQualityState) -> Dict[str, Any]:
        """Format the workflow result for API response"""
        
        return {
            'timestamp': state.get('timestamp'),
            'workflow_status': state.get('workflow_step'),
            'errors': state.get('errors', []),
            'current_analysis': state.get('current_analysis', {}),
            'predictions': state.get('predictions', {}),
            'recommendations': state.get('recommendations', []),
            'risk_assessment': state.get('risk_assessment', {}),
            'action_plan': state.get('action_plan', {}),
            'knowledge_sources': len(state.get('knowledge_context', [])),
            'workflow_type': 'langgraph' if LANGGRAPH_AVAILABLE and self.graph else 'simplified'
        }
    
    # Helper methods for agents
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess quality of sensor data"""
        if len(df) == 0:
            return {'quality_score': 0, 'issues': ['No data']}
        
        issues = []
        if df.isnull().sum().sum() > 0:
            issues.append('Missing values detected')
        
        quality_score = max(0, 100 - len(issues) * 20)
        return {'quality_score': quality_score, 'issues': issues}
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Calculate trend indicators"""
        if len(df) < 2:
            return {'co2_trend': 'insufficient_data'}
        
        co2_trend = 'stable'
        if 'co2_ppm' in df.columns:
            recent_avg = df['co2_ppm'].tail(3).mean()
            earlier_avg = df['co2_ppm'].head(3).mean()
            
            if recent_avg > earlier_avg * 1.1:
                co2_trend = 'increasing'
            elif recent_avg < earlier_avg * 0.9:
                co2_trend = 'decreasing'
        
        return {'co2_trend': co2_trend}
    
    def _calculate_time_span(self, readings: List[Dict[str, Any]]) -> float:
        """Calculate time span of readings in hours"""
        if len(readings) < 2:
            return 0
        
        try:
            timestamps = [pd.to_datetime(r.get('timestamp', datetime.now())) for r in readings]
            return (max(timestamps) - min(timestamps)).total_seconds() / 3600
        except:
            return 0
    
    def _assess_health_status(self, co2: float, temp: float, humidity: float) -> str:
        """Assess health status based on current conditions"""
        if co2 > 2000:
            return 'dangerous'
        elif co2 > 1200 or temp > 28 or temp < 16 or humidity > 70:
            return 'poor'
        elif co2 > 1000 or temp > 26 or temp < 18 or humidity > 60:
            return 'fair'
        else:
            return 'good'
    
    def _analyze_prediction_trends(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends from predictions"""
        trend_analysis = {'overall_trend': 'stable'}
        
        # Analyze short-term predictions
        short_term = predictions.get('predictions', {}).get('short_term', {})
        if isinstance(short_term, dict) and 'trend' in short_term:
            trend_analysis['short_term_trend'] = short_term['trend']
            trend_analysis['overall_trend'] = short_term['trend']
        
        return trend_analysis
    
    def _assess_prediction_confidence(self, predictions: Dict[str, Any]) -> Dict[str, str]:
        """Assess confidence in predictions"""
        confidence = {'overall': 'medium'}
        
        # Check if models are available
        short_term = predictions.get('predictions', {}).get('short_term', {})
        long_term = predictions.get('predictions', {}).get('long_term', {})
        
        if isinstance(short_term, dict) and 'error' not in short_term:
            confidence['short_term'] = 'high'
        else:
            confidence['short_term'] = 'low'
        
        if isinstance(long_term, dict) and 'error' not in long_term:
            confidence['long_term'] = 'high'
        else:
            confidence['long_term'] = 'low'
        
        return confidence
    
    def _categorize_knowledge(self, knowledge_results: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize retrieved knowledge"""
        return {
            'knowledge_context': knowledge_results.get('knowledge_context', []),
            'urgency_level': knowledge_results.get('urgency_level', 'low')
        }
    
    def _generate_immediate_actions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate immediate action recommendations"""
        actions = []
        co2 = analysis.get('current_co2', 400)
        
        if co2 > 1000:
            actions.append({
                'action': f'Immediate ventilation required - CO2 at {co2:.0f} ppm',
                'priority': 'high' if co2 > 1500 else 'medium',
                'type': 'immediate',
                'estimated_time': '5-15 minutes'
            })
        
        return actions
    
    def _generate_predictive_actions(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictive action recommendations"""
        actions = []
        
        # Check prediction trends
        trend_analysis = predictions.get('trend_analysis', {})
        if trend_analysis.get('overall_trend') == 'increasing':
            actions.append({
                'action': 'Monitor closely - conditions predicted to worsen',
                'priority': 'medium',
                'type': 'predictive',
                'estimated_time': '30-60 minutes'
            })
        
        return actions
    
    def _generate_knowledge_actions(self, knowledge_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate knowledge-based recommendations"""
        actions = []
        
        for knowledge in knowledge_context[:2]:  # Limit to top 2
            actions.append({
                'action': f"Based on guidelines: {knowledge.get('content', '')[:100]}...",
                'priority': 'low',
                'type': 'knowledge_based',
                'source': knowledge.get('metadata', {}).get('title', 'Guidelines')
            })
        
        return actions
    
    def _prioritize_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize and deduplicate recommendations"""
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        sorted_recs = sorted(
            recommendations,
            key=lambda x: priority_order.get(x.get('priority', 'low'), 3)
        )
        
        # Remove duplicates and limit count
        seen_actions = set()
        unique_recs = []
        
        for rec in sorted_recs:
            action_key = rec.get('action', '')[:50]  # First 50 chars as key
            if action_key not in seen_actions:
                seen_actions.add(action_key)
                unique_recs.append(rec)
                
                if len(unique_recs) >= 10:  # Limit to 10 recommendations
                    break
        
        return unique_recs
    
    def _assess_current_risk(self, analysis: Dict[str, Any]) -> str:
        """Assess current risk level"""
        health_status = analysis.get('health_status', 'good')
        
        if health_status == 'dangerous':
            return 'critical'
        elif health_status == 'poor':
            return 'high'
        elif health_status == 'fair':
            return 'medium'
        else:
            return 'low'
    
    def _assess_predicted_risk(self, predictions: Dict[str, Any]) -> str:
        """Assess predicted risk level"""
        # Simplified risk assessment based on trends
        trend = predictions.get('trend_analysis', {}).get('overall_trend', 'stable')
        
        if trend == 'increasing':
            return 'medium'
        else:
            return 'low'
    
    def _identify_risk_factors(self, analysis: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors"""
        factors = []
        
        co2 = analysis.get('current_co2', 400)
        if co2 > 800:
            factors.append(f"Elevated CO2 levels ({co2:.0f} ppm)")
        
        temp = analysis.get('current_temperature', 20)
        if temp > 26 or temp < 18:
            factors.append(f"Temperature outside comfort range ({temp:.1f}°C)")
        
        return factors
    
    def _determine_urgency(self, analysis: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """Determine overall urgency level"""
        current_risk = self._assess_current_risk(analysis)
        predicted_risk = self._assess_predicted_risk(predictions)
        
        if current_risk == 'critical':
            return 'immediate'
        elif current_risk == 'high' or predicted_risk == 'high':
            return 'urgent'
        elif current_risk == 'medium' or predicted_risk == 'medium':
            return 'moderate'
        else:
            return 'routine'
    
    def _create_monitoring_plan(self, analysis: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring plan"""
        return {
            'frequency': 'every_15_minutes' if analysis.get('health_status') == 'poor' else 'every_30_minutes',
            'parameters': ['co2_ppm', 'temperature_celsius', 'humidity_percent'],
            'alert_thresholds': {
                'co2_ppm': 1000,
                'temperature_celsius': 26,
                'humidity_percent': 60
            }
        }
    
    def _define_escalation_triggers(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Define escalation triggers"""
        return [
            {
                'condition': 'CO2 > 2000 ppm',
                'action': 'Emergency ventilation protocol',
                'contact': 'Facilities management'
            },
            {
                'condition': 'No improvement after 1 hour',
                'action': 'HVAC system inspection',
                'contact': 'Maintenance team'
            }
        ]
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get workflow status and capabilities"""
        
        return {
            'workflow_available': True,
            'langgraph_enabled': LANGGRAPH_AVAILABLE and self.graph is not None,
            'ml_service_status': self.ml_service.get_model_status(),
            'rag_service_status': self.rag_service.get_service_status(),
            'agents': {
                'sensor_analyzer': True,
                'predictor': True,
                'knowledge_retriever': True,
                'advisor': True,
                'risk_assessor': True
            }
        }
