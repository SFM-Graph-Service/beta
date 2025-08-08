# SFM Graph Service - Module Architecture UML Diagram

This document provides a comprehensive UML diagram of the Social Fabric Matrix (SFM) Graph Service architecture, illustrating the relationships between modules, classes, and their key methods and properties.

## Architecture Overview

The SFM Graph Service implements F. Gregory Hayden's Social Fabric Matrix methodology for institutional economics analysis. The architecture is organized into several layers:

1. **Base Infrastructure Layer**: Core node and relationship classes with UUID identity and rich metadata
2. **Core Entity Layer**: Primary SFM entities (Actor, Institution, Policy, Resource, Process, Flow)  
3. **Specialized Analysis Layer**: Advanced analysis and system property classes for complex measurements
4. **Framework Components Layer**: System boundary, criteria, indicators, and validation frameworks
5. **Domain-Specific Modules**: Specialized modules for power analysis, democratic processes, behavioral modeling
6. **Supporting Infrastructure**: Enums, exceptions, metadata, temporal/spatial contexts, and utilities

## Key Architectural Patterns

### Entity-Relationship Architecture
- **UUID-based Identity**: All entities use UUID for consistent, global identification
- **Rich Metadata**: Comprehensive versioning, data quality tracking, and temporal dynamics
- **Loose Coupling**: Entities reference each other through UUIDs, enabling flexible composition
- **Type Safety**: Extensive use of enums for controlled vocabularies

### Analytical Framework Integration
- **Ceremonial-Instrumental Analysis**: Core dichotomy analysis throughout the system
- **Power Structure Analysis**: Comprehensive stakeholder power assessment and coalition modeling  
- **Temporal Analysis**: Support for change processes and institutional evolution
- **System Integration**: Holistic system coherence checking and quality assurance

### Validation and Quality Assurance
- **Multi-layer Validation**: Schema validation, integrity checking, and quality analysis
- **Error Handling**: Comprehensive exception hierarchy with rich context
- **Data Quality Tracking**: Built-in quality metrics and uncertainty handling

## UML Class Diagram

```plantuml
@startuml SFM_Graph_Service_Architecture

!define BASECOLOR #E1F5FE
!define CORECOLOR #F3E5F5
!define SPECIALCOLOR #E8F5E8
!define FRAMEWORKCOLOR #FFF3E0
!define DOMAINCOLOR #FCE4EC
!define SUPPORTCOLOR #F1F8E9

skinparam class {
  BackgroundColor BASECOLOR
  BorderColor #0277BD
  ArrowColor #0277BD
}

package "Base Infrastructure Layer" <<Rectangle>> {
  class Node <<BASECOLOR>> {
    +id: UUID
    +label: str
    +description: Optional[str]
    +meta: Dict[str, str]
    +version: int
    +created_at: datetime
    +modified_at: Optional[datetime]
    +certainty: Optional[float]
    +data_quality: Optional[str]
    +previous_version_id: Optional[UUID]
    --
    +__iter__(): Iterator[Tuple[str, Any]]
  }

  class Relationship <<BASECOLOR>> {
    +id: UUID
    +source_id: UUID
    +target_id: UUID
    +kind: RelationshipKind
    +weight: Optional[float]
    +time: Optional[TimeSlice]
    +space: Optional[SpatialUnit]
    +scenario: Optional[Scenario]
    +meta: Dict[str, str]
    +certainty: Optional[float]
    +variability: Optional[float]
    +version: int
    +created_at: datetime
    +modified_at: Optional[datetime]
    +data_quality: Optional[str]
    +temporal_dynamics: Optional[TemporalDynamics]
  }
}

package "Core Entity Layer" <<Rectangle>> {
  class Actor <<CORECOLOR>> {
    +legal_form: Optional[str]
    +sector: Optional[str]
    +power_resources: Dict[str, float]
    +decision_making_capacity: Optional[float]
    +institutional_affiliations: List[UUID]
    +cognitive_frameworks: List[UUID]
    +behavioral_patterns: List[UUID]
    +network_centrality: Optional[float]
    +coalition_memberships: List[UUID]
    +influence_relationships: Dict[UUID, float]
    +resource_dependencies: Dict[UUID, str]
    +bargaining_power: Optional[float]
    +veto_power: List[str]
    +agenda_setting_power: Optional[float]
    +power_trajectory: List[Dict[str, float]]
    +power_consolidation_strategies: List[str]
    +power_distribution_preferences: Dict[str, float]
    +legitimacy_sources: List[str]
    +authority_scope: List[str]
    +legitimacy_challenges: List[str]
    --
    +calculate_power_index(): float
    +get_dominant_power_resource(): Optional[str]
    +assess_institutional_embeddedness(): float
    +analyze_actor_ci_orientation(): Dict[str, Any]
    +assess_transformation_influence_capacity(): Dict[str, Any]
    +generate_actor_ci_engagement_strategy(): Dict[str, List[str]]
  }

  class Institution <<CORECOLOR>> {
    +formality_level: Optional[str]
    +scope: Optional[str]
    +enforcement_mechanism: Optional[str]
    +rule_types: List[str]
    +enforcement_strength: Optional[float]
    +legitimacy_score: Optional[float]
    +change_frequency: Optional[float]
    +institutional_complementarity: List[UUID]
    +ceremonial_instrumental_balance: Optional[float]
    --
    +calculate_institutional_effectiveness(): float
    +get_institutional_type_classification(): str
    +assess_complementarity_strength(): float
    +conduct_integrated_ci_analysis(): Dict[str, Any]
    +integrate_with_matrix_ci_analysis(matrix_ci_data): Dict[str, Any]
  }

  class Policy <<CORECOLOR>> {
    +policy_type: Optional[str]
    +policy_domain: Optional[str]
    +implementation_status: Optional[str]
    +target_outcomes: List[str]
    +policy_instruments: List[UUID]
    +affected_actors: List[UUID]
    +policy_effectiveness: Optional[float]
    +unintended_consequences: List[str]
    --
    +assess_policy_effectiveness(): float
    +identify_policy_gaps(): List[str]
    +evaluate_policy_coherence(): float
  }

  class Resource <<CORECOLOR>> {
    +resource_type: ResourceType
    +unit_of_measure: Optional[str]
    +quantity: Optional[float]
    +quality_indicators: Dict[str, float]
    +availability_status: Optional[str]
    +depletion_rate: Optional[float]
    +regeneration_capacity: Optional[float]
    +ownership_structure: List[UUID]
    +access_rights: Dict[UUID, str]
    --
    +calculate_sustainability_index(): float
    +assess_resource_security(): float
    +evaluate_access_equity(): float
  }

  class Process <<CORECOLOR>> {
    +process_type: Optional[str]
    +process_stage: Optional[str]
    +inputs: List[UUID]
    +outputs: List[UUID]
    +transformation_rules: List[str]
    +efficiency_metrics: Dict[str, float]
    +quality_controls: List[str]
    +stakeholder_involvement: List[UUID]
    --
    +calculate_process_efficiency(): float
    +assess_quality_performance(): float
    +evaluate_stakeholder_satisfaction(): float
  }

  class Flow <<CORECOLOR>> {
    +nature: FlowNature
    +flow_type: FlowType
    +rate: Optional[float]
    +direction: Optional[str]
    +constraints: List[str]
    +flow_patterns: Dict[str, Any]
    +seasonal_variation: Optional[float]
    +volatility_measures: Dict[str, float]
    --
    +calculate_flow_stability(): float
    +assess_flow_sustainability(): float
    +identify_bottlenecks(): List[str]
  }
}

package "Specialized Analysis Layer" <<Rectangle>> {
  class SystemProperty <<SPECIALCOLOR>> {
    +property_type: SystemPropertyType
    +value: Any
    +unit: Optional[str]
    +timestamp: datetime
    +affected_nodes: List[UUID]
    +contributing_relationships: List[UUID]
  }

  class PolicyInstrument <<SPECIALCOLOR>> {
    +instrument_type: PolicyInstrumentType
    +target_behavior: Optional[str]
    +compliance_mechanism: Optional[str]
    +effectiveness_measure: Optional[float]
    --
    +__post_init__(): None
  }

  class MatrixCell <<SPECIALCOLOR>> {
    +institution_id: Optional[UUID]
    +criteria_id: Optional[UUID]
    +correlation_type: CorrelationType
    +correlation_strength: Optional[float]
    +correlation_scale: CorrelationScale
    +evidence_quality: EvidenceQuality
    +justification: Optional[str]
    +data_sources: List[str]
    +confidence_level: Optional[float]
    +last_updated: datetime
    +reviewed_by: Optional[str]
    +review_date: Optional[datetime]
    +deliveries_provided: List[Dict[str, Any]]
    +deliveries_received: List[Dict[str, Any]]
    +delivery_quality: Optional[float]
    +delivery_reliability: Optional[float]
  }

  class SFMCriteria <<SPECIALCOLOR>> {
    +criteria_type: CriteriaType
    +priority: CriteriaPriority
    +measurement_approach: MeasurementApproach
    +quantitative_indicators: List[str]
    +qualitative_descriptors: List[str]
    +threshold_values: Dict[str, float]
    +validation_methods: List[str]
  }

  class InstitutionalStructure <<SPECIALCOLOR>> {
    +structure_type: Optional[str]
    +hierarchy_level: Optional[int]
    +coordination_mechanisms: List[str]
    +decision_processes: List[str]
    +authority_distribution: Dict[str, float]
    +accountability_mechanisms: List[str]
  }

  class TransactionCost <<SPECIALCOLOR>> {
    +cost_type: Optional[str]
    +cost_category: Optional[str]
    +cost_estimate: Optional[float]
    +cost_drivers: List[str]
    +reduction_strategies: List[str]
    +measurement_method: Optional[str]
  }
}

package "Framework Components Layer" <<Rectangle>> {
  class SystemBoundary <<FRAMEWORKCOLOR>> {
    +boundary_type: Optional[str]
    +scope_definition: Optional[str]
    +inclusion_criteria: List[str]
    +exclusion_criteria: List[str]
    +boundary_spanners: List[UUID]
    +permeability_level: Optional[float]
    --
    +validate_boundary(): bool
    +assess_boundary_adequacy(): float
  }

  class ProblemDefinition <<FRAMEWORKCOLOR>> {
    +problem_type: Optional[str]
    +problem_scope: Optional[str]
    +stakeholder_perspectives: Dict[UUID, str]
    +problem_symptoms: List[str]
    +root_causes: List[str]
    +problem_evolution: List[Dict[str, Any]]
    --
    +analyze_problem_complexity(): float
    +identify_key_stakeholders(): List[UUID]
  }

  class SocialValueSystem <<FRAMEWORKCOLOR>> {
    +value_dimensions: List[ValueDimension]
    +value_priorities: Dict[str, float]
    +value_conflicts: List[Dict[str, Any]]
    +cultural_context: Optional[str]
    +evolution_patterns: List[Dict[str, Any]]
    --
    +assess_value_coherence(): float
    +identify_value_tensions(): List[str]
    +evaluate_value_alignment(): float
  }

  class IndicatorSystem <<FRAMEWORKCOLOR>> {
    +indicator_categories: List[str]
    +measurement_framework: Optional[str]
    +data_sources: List[str]
    +update_frequency: Optional[str]
    +quality_standards: List[str]
    --
    +validate_indicator_coherence(): bool
    +assess_measurement_quality(): float
  }

  class CriteriaFramework <<FRAMEWORKCOLOR>> {
    +evaluation_criteria: List[UUID]
    +weighting_scheme: Dict[str, float]
    +aggregation_method: Optional[str]
    +sensitivity_analysis: Optional[Dict[str, Any]]
    --
    +conduct_multi_criteria_analysis(): Dict[str, Any]
    +validate_criteria_consistency(): bool
  }
}

package "Domain-Specific Modules" <<Rectangle>> {
  class PowerAssessment <<DOMAINCOLOR>> {
    +power_dimensions: Dict[str, float]
    +power_sources: List[str]
    +power_relationships: List[UUID]
    +power_dynamics: Dict[str, Any]
    --
    +calculate_total_power(): float
    +analyze_power_distribution(): Dict[str, Any]
  }

  class StakeholderCoalition <<DOMAINCOLOR>> {
    +member_actors: List[UUID]
    +coalition_purpose: Optional[str]
    +coordination_mechanisms: List[str]
    +resource_pooling: Dict[str, float]
    +decision_rules: List[str]
    --
    +assess_coalition_stability(): float
    +calculate_collective_power(): float
  }

  class SocialFabricIndicator <<DOMAINCOLOR>> {
    +indicator_type: SocialFabricIndicatorType
    +measurement_scale: Optional[str]
    +current_value: Optional[float]
    +historical_values: List[Dict[str, Any]]
    +trend_analysis: Optional[Dict[str, Any]]
    --
    +calculate_trend(): float
    +assess_reliability(): float
  }

  class DemocraticSystem <<DOMAINCOLOR>> {
    +democratic_mechanisms: List[str]
    +participation_levels: Dict[str, float]
    +representation_quality: Optional[float]
    +accountability_structures: List[str]
    --
    +assess_democratic_quality(): float
    +evaluate_participation_effectiveness(): float
  }

  class ValueHierarchy <<DOMAINCOLOR>> {
    +parent_values: List[UUID]
    +priority_weight: Optional[float]
    +cultural_domain: Optional[str]
    +legitimacy_source: Optional[LegitimacySource]
  }

  class CeremonialBehavior <<DOMAINCOLOR>> {
    +rigidity_level: Optional[float]
    +tradition_strength: Optional[float]
    +resistance_to_change: Optional[float]
  }

  class InstrumentalBehavior <<DOMAINCOLOR>> {
    +efficiency_measure: Optional[float]
    +adaptability_score: Optional[float]
    +innovation_potential: Optional[float]
  }

  class ChangeProcess <<DOMAINCOLOR>> {
    +change_type: ChangeType
    +change_agents: List[UUID]
    +resistance_factors: List[UUID]
    +change_trajectory: List[TimeSlice]
    +success_probability: Optional[float]
    +temporal_dynamics: Optional[TemporalDynamics]
  }

  class CognitiveFramework <<DOMAINCOLOR>> {
    +framing_effects: Dict[str, str]
    +cognitive_biases: List[str]
    +information_filters: List[str]
    +learning_capacity: Optional[float]
  }

  class BehavioralPattern <<DOMAINCOLOR>> {
    +pattern_type: BehaviorPatternType
    +frequency: Optional[float]
    +predictability: Optional[float]
    +context_dependency: List[str]
  }

  class AdvancedDataValidator <<DOMAINCOLOR>> {
    +validation_schemas: Dict[str, ValidationSchema]
    +integrity_checkers: List[IntegrityChecker]
    +quality_analyzers: List[DataQualityAnalyzer]
    +validation_rules: List[ValidationRule]
    --
    +validate_comprehensive(): ValidationResult
    +assess_data_quality(): Dict[str, Any]
    +check_system_integrity(): bool
  }
}

package "Supporting Infrastructure" <<Rectangle>> {
  enum RelationshipKind <<SUPPORTCOLOR>> {
    INFLUENCES
    CONTROLS
    DEPENDS_ON
    PROVIDES_TO
    COMPETES_WITH
    COLLABORATES_WITH
    REGULATES
    SUPPORTS
  }

  enum ResourceType <<SUPPORTCOLOR>> {
    NATURAL
    HUMAN
    PHYSICAL
    FINANCIAL
    INFORMATIONAL
    INSTITUTIONAL
    CULTURAL
    TECHNOLOGICAL
  }

  enum FlowNature <<SUPPORTCOLOR>> {
    PHYSICAL
    FINANCIAL
    INFORMATIONAL
    ENERGY
    HUMAN_CAPITAL
    AUTHORITY
    LEGITIMACY
  }

  enum FlowType <<SUPPORTCOLOR>> {
    STOCK
    FLOW
    RATE
    ACCUMULATION
  }

  enum ParticipationLevel <<SUPPORTCOLOR>> {
    INFORMATION
    CONSULTATION
    INVOLVEMENT
    COLLABORATION
    EMPOWERMENT
  }

  enum BehaviorPatternType <<SUPPORTCOLOR>> {
    HABITUAL
    STRATEGIC
    ADAPTIVE
    RESISTANT
  }

  enum ChangeType <<SUPPORTCOLOR>> {
    EVOLUTIONARY
    REVOLUTIONARY
    CYCLICAL
  }

  class SFMException <<SUPPORTCOLOR>> {
    +error_code: Optional[str]
    +context: Dict[str, Any]
  }

  class ValidationResult <<SUPPORTCOLOR>> {
    +is_valid: bool
    +violations: List[str]
    +warnings: List[str]
    +context: ValidationContext
  }

  class TemporalDynamics <<SUPPORTCOLOR>> {
    +change_pattern: Optional[str]
    +rate_of_change: Optional[float]
    +stability_indicator: Optional[float]
    +periodicity: Optional[str]
  }

  class TimeSlice <<SUPPORTCOLOR>> {
    +label: str
    +start_date: Optional[datetime]
    +end_date: Optional[datetime]
    +duration: Optional[timedelta]
    +time_resolution: Optional[str]
  }

  class SpatialUnit <<SUPPORTCOLOR>> {
    +label: str
    +spatial_type: Optional[str]
    +boundaries: Optional[Dict[str, Any]]
    +coordinate_system: Optional[str]
  }

  class Scenario <<SUPPORTCOLOR>> {
    +scenario_type: ScenarioType
    +assumptions: List[str]
    +uncertainty_factors: List[str]
    +probability: Optional[float]
  }
}
}

' Inheritance relationships
Node <|-- Actor
Node <|-- Institution
Node <|-- Policy
Node <|-- Resource
Node <|-- Process
Node <|-- Flow
Node <|-- SystemProperty
Node <|-- PolicyInstrument
Node <|-- MatrixCell
Node <|-- SFMCriteria
Node <|-- InstitutionalStructure
Node <|-- TransactionCost
Node <|-- SystemBoundary
Node <|-- ProblemDefinition
Node <|-- SocialValueSystem
Node <|-- IndicatorSystem
Node <|-- CriteriaFramework
Node <|-- PowerAssessment
Node <|-- StakeholderCoalition
Node <|-- SocialFabricIndicator
Node <|-- DemocraticSystem
Node <|-- ValueHierarchy
Node <|-- CeremonialBehavior
Node <|-- InstrumentalBehavior
Node <|-- ChangeProcess
Node <|-- CognitiveFramework
Node <|-- BehavioralPattern
Node <|-- AdvancedDataValidator

' Key composition relationships
Actor "1" *-- "0..*" UUID : institutional_affiliations
Actor "1" *-- "0..*" UUID : coalition_memberships
Institution "1" *-- "0..*" UUID : institutional_complementarity
Policy "1" *-- "0..*" UUID : policy_instruments
Policy "1" *-- "0..*" UUID : affected_actors
Resource "1" *-- "0..*" UUID : ownership_structure
Process "1" *-- "0..*" UUID : inputs
Process "1" *-- "0..*" UUID : outputs
Process "1" *-- "0..*" UUID : stakeholder_involvement
MatrixCell "1" -- "1" UUID : institution_id
MatrixCell "1" -- "1" UUID : criteria_id
SystemBoundary "1" *-- "0..*" UUID : boundary_spanners
StakeholderCoalition "1" *-- "0..*" UUID : member_actors

' Association relationships
Relationship "0..*" -- "1" UUID : source_id
Relationship "0..*" -- "1" UUID : target_id
Relationship "1" -- "1" RelationshipKind : kind
Resource "1" -- "1" ResourceType : resource_type
Flow "1" -- "1" FlowNature : nature
Flow "1" -- "1" FlowType : flow_type
BehavioralPattern "1" -- "1" BehaviorPatternType : pattern_type
ChangeProcess "1" -- "1" ChangeType : change_type

' Framework integration relationships
CriteriaFramework "1" *-- "0..*" UUID : evaluation_criteria
PowerAssessment "1" *-- "0..*" UUID : power_relationships
SystemProperty "1" *-- "0..*" UUID : affected_nodes
SystemProperty "1" *-- "0..*" UUID : contributing_relationships
ChangeProcess "1" *-- "0..*" UUID : change_agents
ChangeProcess "1" *-- "0..*" UUID : resistance_factors
ValueHierarchy "1" *-- "0..*" UUID : parent_values

' Temporal and spatial context relationships
Relationship "0..*" -- "0..1" TimeSlice : time
Relationship "0..*" -- "0..1" SpatialUnit : space
Relationship "0..*" -- "0..1" Scenario : scenario
ChangeProcess "1" *-- "0..*" TimeSlice : change_trajectory

@enduml
```

## Module Dependencies and Relationships

### Core Dependencies
- **Base Infrastructure**: All other modules depend on `Node` and `Relationship` classes
- **Core Entities**: Specialized classes inherit from and reference core entities
- **Enums**: Provide controlled vocabularies used throughout the system
- **Exceptions**: Handle error conditions across all modules

### Key Integration Points
1. **UUID-based References**: All entities use UUID for cross-references, enabling loose coupling
2. **Metadata Integration**: All nodes support rich metadata and versioning
3. **Temporal Context**: Support for time-based analysis through `TimeSlice` and `TemporalDynamics`
4. **Spatial Context**: Geographic context through `SpatialUnit` references
5. **Scenario Analysis**: Support for multiple scenarios and comparative analysis

### Analysis Frameworks
- **Ceremonial-Instrumental Analysis**: Core analytical framework throughout the system
- **Power Analysis**: Stakeholder power assessment and coalition analysis
- **System Integration**: Comprehensive system coherence checking
- **Quality Assurance**: Data validation and quality measurement

## Key Design Patterns

1. **Dataclass Pattern**: All entities use Python dataclasses for clean structure
2. **UUID Identity**: Consistent UUID-based identity system
3. **Rich Metadata**: Comprehensive metadata support with versioning
4. **Enum-based Vocabularies**: Controlled vocabularies for consistent categorization
5. **Compositional Relationships**: Flexible composition through UUID references
6. **Analytical Methods**: Rich analytical methods on domain entities
7. **Validation Framework**: Comprehensive validation and quality assurance

## Usage Examples

The SFM framework enables complex institutional analysis following Hayden's methodology:

### Basic Entity Creation and Analysis
```python
# Create an actor with comprehensive power analysis
actor = Actor(
    label="Municipal Government",
    legal_form="Government Agency", 
    sector="Public",
    power_resources={
        "institutional_authority": 0.8,
        "economic_control": 0.6,
        "information_access": 0.7
    },
    network_centrality=0.7,
    agenda_setting_power=0.9,
    coalition_memberships=[uuid1, uuid2]
)

# Analyze actor's ceremonial-instrumental orientation
ci_analysis = actor.analyze_actor_ci_orientation()
print(f"CI Balance: {ci_analysis['orientation_balance']}")

# Assess transformation influence capacity
transformation_capacity = actor.assess_transformation_influence_capacity()
print(f"Role: {transformation_capacity['transformation_role_classification']}")
```

### Institution Analysis
```python
# Create institution with CI balance
institution = Institution(
    label="Environmental Protection Agency",
    formality_level="formal",
    enforcement_strength=0.7,
    legitimacy_score=0.8,
    ceremonial_instrumental_balance=0.3  # Slightly instrumental
)

# Conduct integrated CI analysis
ci_results = institution.conduct_integrated_ci_analysis()
effectiveness = institution.calculate_institutional_effectiveness()
```

### Relationship and Matrix Analysis
```python
# Create regulatory relationship
relationship = Relationship(
    source_id=actor.id,
    target_id=institution.id,
    kind=RelationshipKind.REGULATES,
    weight=0.8,
    certainty=0.9
)

# Create matrix cell for institution-criteria analysis
matrix_cell = MatrixCell(
    institution_id=institution.id,
    criteria_id=criteria.id,
    correlation_type=CorrelationType.POSITIVE,
    correlation_strength=0.7,
    correlation_scale=CorrelationScale.MODERATELY_POSITIVE,
    evidence_quality=EvidenceQuality.HIGH,
    justification="Strong institutional support for criteria"
)
```

### Behavioral and Change Analysis
```python
# Model change process
change_process = ChangeProcess(
    label="Digital Transformation Initiative",
    change_type=ChangeType.EVOLUTIONARY,
    change_agents=[actor.id, institution.id],
    success_probability=0.75
)

# Analyze behavioral patterns
behavior = BehavioralPattern(
    label="Collaborative Decision Making",
    pattern_type=BehaviorPatternType.STRATEGIC,
    frequency=0.8,
    predictability=0.6
)
```

### Power and Coalition Analysis
```python
# Create stakeholder coalition
coalition = StakeholderCoalition(
    label="Environmental Advocacy Coalition",
    member_actors=[actor1.id, actor2.id, actor3.id],
    coalition_purpose="Climate policy advocacy",
    resource_pooling={"funding": 500000, "expertise": 0.9}
)

# Assess coalition power
collective_power = coalition.calculate_collective_power()
stability = coalition.assess_coalition_stability()
```

This architecture supports comprehensive socio-economic system analysis following the Social Fabric Matrix methodology, enabling researchers and policymakers to model complex institutional relationships and analyze system-wide patterns.