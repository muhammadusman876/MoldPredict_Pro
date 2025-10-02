"""
RAG (Retrieval-Augmented Generation) knowledge base for air quality recommendations
Integrates WHO guidelines, HVAC best practices, and health information
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Import with error handling for optional dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class AirQualityKnowledgeBase:
    """Knowledge base for air quality information and recommendations"""
    
    def __init__(self, db_path: str = "knowledge_base/chroma_db"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available. Please install chromadb")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers is not available. Please install sentence-transformers")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Collection for air quality knowledge
        self.collection = self.client.get_or_create_collection(
            name="air_quality_knowledge",
            metadata={"description": "Air quality guidelines, HVAC best practices, and health information"}
        )
        
        # Initialize with default knowledge if empty
        if self.collection.count() == 0:
            self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with default air quality information"""
        
        logger.info("Initializing knowledge base with default information")
        
        # WHO guidelines and health information
        who_guidelines = [
            {
                "id": "who_co2_guidelines",
                "title": "WHO CO2 Guidelines",
                "content": """
                According to WHO guidelines, indoor CO2 levels should be maintained below 1000 ppm for good air quality.
                Levels above 1000 ppm indicate inadequate ventilation. Levels above 5000 ppm can cause drowsiness.
                Optimal indoor CO2 levels are between 400-1000 ppm. Fresh outdoor air typically contains 400-420 ppm CO2.
                """,
                "category": "guidelines",
                "source": "WHO",
                "tags": ["co2", "health", "guidelines", "who"]
            },
            {
                "id": "co2_health_effects",
                "title": "CO2 Health Effects",
                "content": """
                CO2 levels and health impacts:
                - 400-1000 ppm: Good air quality, no health effects
                - 1000-2000 ppm: Drowsiness, poor air quality
                - 2000-5000 ppm: Workplace exposure limit, drowsiness and stale air
                - 5000+ ppm: Immediately dangerous, can cause unconsciousness
                High CO2 levels reduce cognitive performance and cause fatigue.
                """,
                "category": "health",
                "source": "Medical Research",
                "tags": ["co2", "health", "effects", "cognitive"]
            },
            {
                "id": "temperature_humidity_guidelines",
                "title": "Temperature and Humidity Guidelines",
                "content": """
                Optimal indoor conditions for health and comfort:
                - Temperature: 20-24°C (68-75°F) for most activities
                - Relative humidity: 40-60% for optimal health
                - Humidity below 30% can cause dry skin, irritated respiratory system
                - Humidity above 60% can promote mold growth and dust mites
                Temperature and humidity affect perceived air quality and comfort.
                """,
                "category": "guidelines",
                "source": "ASHRAE",
                "tags": ["temperature", "humidity", "comfort", "health"]
            }
        ]
        
        # HVAC and ventilation best practices
        hvac_practices = [
            {
                "id": "ventilation_strategies",
                "title": "Ventilation Strategies",
                "content": """
                Effective ventilation strategies to improve indoor air quality:
                - Natural ventilation: Open windows and doors when outdoor air quality is good
                - Mechanical ventilation: Use exhaust fans in kitchens and bathrooms
                - Air circulation: Use ceiling fans to improve air movement
                - HVAC maintenance: Change filters regularly, clean ducts
                - Fresh air intake: Ensure HVAC systems bring in adequate outdoor air
                Target: 15-20 CFM of fresh air per person minimum.
                """,
                "category": "hvac",
                "source": "HVAC Best Practices",
                "tags": ["ventilation", "hvac", "fresh air", "circulation"]
            },
            {
                "id": "air_purification",
                "title": "Air Purification Methods",
                "content": """
                Air purification methods for improving indoor air quality:
                - HEPA filters: Remove 99.97% of particles 0.3 microns or larger
                - Activated carbon filters: Remove odors and volatile organic compounds
                - UV-C sterilization: Kills bacteria and viruses
                - Plants: Some plants can help improve air quality naturally
                - Source control: Eliminate or reduce pollution sources
                Combine multiple methods for best results.
                """,
                "category": "hvac",
                "source": "Air Quality Standards",
                "tags": ["purification", "filters", "hepa", "plants"]
            },
            {
                "id": "energy_efficiency",
                "title": "Energy-Efficient Air Quality Management",
                "content": """
                Balance air quality and energy efficiency:
                - Demand-controlled ventilation: Adjust ventilation based on occupancy
                - Heat recovery ventilation: Pre-condition incoming fresh air
                - Smart scheduling: Ventilate during optimal outdoor conditions
                - Zoned systems: Control air quality by area/room
                - Regular maintenance: Ensure systems operate efficiently
                Energy-efficient solutions reduce costs while maintaining air quality.
                """,
                "category": "efficiency",
                "source": "Energy Star",
                "tags": ["energy", "efficiency", "smart", "controls"]
            }
        ]
        
        # Specific recommendations and actions
        recommendations = [
            {
                "id": "immediate_actions_high_co2",
                "title": "Immediate Actions for High CO2",
                "content": """
                When CO2 levels exceed 1000 ppm, take immediate action:
                1. Open windows and doors if outdoor air quality is acceptable
                2. Turn on exhaust fans in kitchens and bathrooms
                3. Reduce occupancy if possible
                4. Check HVAC system operation and increase fresh air intake
                5. Use portable fans to improve air circulation
                Monitor levels and ensure they decrease within 30 minutes.
                """,
                "category": "emergency",
                "source": "Safety Guidelines",
                "tags": ["immediate", "high co2", "emergency", "ventilation"]
            },
            {
                "id": "preventive_measures",
                "title": "Preventive Air Quality Measures",
                "content": """
                Preventive measures to maintain good air quality:
                - Schedule regular HVAC maintenance (quarterly)
                - Monitor CO2 levels continuously with sensors
                - Establish ventilation schedules based on occupancy
                - Use timers for exhaust fans during high-activity periods
                - Maintain indoor plants that improve air quality
                - Keep indoor humidity between 40-60%
                Prevention is more effective than reactive measures.
                """,
                "category": "prevention",
                "source": "Best Practices",
                "tags": ["prevention", "maintenance", "monitoring", "scheduling"]
            },
            {
                "id": "seasonal_considerations",
                "title": "Seasonal Air Quality Considerations",
                "content": """
                Adapt air quality management to seasons:
                - Summer: Use air conditioning efficiently, monitor humidity
                - Winter: Balance heating with ventilation, watch for dry air
                - Spring/Fall: Take advantage of natural ventilation opportunities
                - Pollen seasons: Keep windows closed, use HEPA filters
                - Wildfire seasons: Monitor outdoor air quality before ventilating
                Seasonal adjustments optimize comfort and health year-round.
                """,
                "category": "seasonal",
                "source": "Climate Guidelines",
                "tags": ["seasonal", "summer", "winter", "pollen", "wildfire"]
            }
        ]
        
        # Combine all knowledge
        all_knowledge = who_guidelines + hvac_practices + recommendations
        
        # Add to knowledge base
        for item in all_knowledge:
            self.add_knowledge(
                item["content"],
                metadata={
                    "title": item["title"],
                    "category": item["category"],
                    "source": item["source"],
                    "tags": ",".join(item["tags"])
                },
                doc_id=item["id"]
            )
        
        logger.info(f"Initialized knowledge base with {len(all_knowledge)} documents")
    
    def add_knowledge(self, content: str, metadata: Dict[str, str], doc_id: Optional[str] = None) -> str:
        """Add a piece of knowledge to the database"""
        
        if doc_id is None:
            doc_id = f"doc_{self.collection.count()}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(content).tolist()
        
        # Add to collection
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        logger.debug(f"Added knowledge document: {doc_id}")
        return doc_id
    
    def search_knowledge(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant knowledge based on query"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'relevance_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'id': results['ids'][0][i]
            })
        
        return formatted_results
    
    def get_recommendations_for_conditions(self, 
                                         co2_ppm: float, 
                                         temperature: float, 
                                         humidity: float,
                                         prediction_trend: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get specific recommendations based on current conditions"""
        
        # Build context-aware query
        query_parts = []
        
        if co2_ppm > 1000:
            query_parts.append(f"high CO2 {co2_ppm} ppm immediate action ventilation")
        elif co2_ppm > 800:
            query_parts.append(f"elevated CO2 {co2_ppm} ppm ventilation improvement")
        else:
            query_parts.append("CO2 maintenance prevention")
        
        if temperature > 26:
            query_parts.append("high temperature cooling")
        elif temperature < 18:
            query_parts.append("low temperature heating")
        
        if humidity > 60:
            query_parts.append("high humidity dehumidification")
        elif humidity < 30:
            query_parts.append("low humidity humidification")
        
        if prediction_trend == "increasing":
            query_parts.append("preventive measures trending upward")
        elif prediction_trend == "decreasing":
            query_parts.append("monitoring improvement")
        
        query = " ".join(query_parts)
        
        # Search for relevant recommendations
        results = self.search_knowledge(query, n_results=3)
        
        # Add condition-specific context
        for result in results:
            result['conditions'] = {
                'co2_ppm': co2_ppm,
                'temperature': temperature,
                'humidity': humidity,
                'trend': prediction_trend
            }
        
        return results
    
    def get_knowledge_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all knowledge for a specific category"""
        
        results = self.collection.get(
            where={"category": category}
        )
        
        formatted_results = []
        for i in range(len(results['documents'])):
            formatted_results.append({
                'content': results['documents'][i],
                'metadata': results['metadatas'][i],
                'id': results['ids'][i]
            })
        
        return formatted_results
    
    def add_custom_knowledge(self, title: str, content: str, category: str = "custom", source: str = "user", tags: List[str] = None) -> str:
        """Add custom knowledge to the database"""
        
        if tags is None:
            tags = []
        
        metadata = {
            "title": title,
            "category": category,
            "source": source,
            "tags": ",".join(tags)
        }
        
        return self.add_knowledge(content, metadata)
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        
        total_docs = self.collection.count()
        
        # Get all documents to analyze categories
        all_docs = self.collection.get()
        
        categories = {}
        sources = {}
        
        for metadata in all_docs['metadatas']:
            category = metadata.get('category', 'unknown')
            source = metadata.get('source', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'total_documents': total_docs,
            'categories': categories,
            'sources': sources,
            'database_path': str(self.db_path),
            'embedding_model': 'all-MiniLM-L6-v2'
        }
    
    def delete_knowledge(self, doc_id: str) -> bool:
        """Delete a knowledge document"""
        
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted knowledge document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def update_knowledge(self, doc_id: str, content: str, metadata: Dict[str, str]) -> bool:
        """Update an existing knowledge document"""
        
        try:
            # Delete old document
            self.collection.delete(ids=[doc_id])
            
            # Add updated document
            self.add_knowledge(content, metadata, doc_id)
            
            logger.info(f"Updated knowledge document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
