"""
Advanced Data Validation Schema for Social Fabric Matrix Framework.

This module implements sophisticated validation schemas and integrity checking
for SFM data structures, ensuring data quality and consistency across the
entire framework.

Key Components:
- AdvancedDataValidator: Main validation engine with schema support
- ValidationSchema: Configurable validation schemas for different data types
- IntegrityChecker: Cross-reference and relationship integrity validation
- DataQualityAnalyzer: Comprehensive data quality assessment and scoring
"""

from __future__ import annotations

import uuid
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union, Callable, Type
from datetime import datetime, timedelta
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging

from models.base_nodes import Node
from models.core_nodes import Indicator, Actor, Institution, Flow, Relationship
from models.matrix_construction import MatrixCell, DeliveryMatrix
from models.sfm_enums import ValueCategory, InstitutionType, RelationshipKind
from models.realtime_data_integration import ValidationRule, ValidationSeverity, DataRecord

# Logging setup
logger = logging.getLogger(__name__)


class ValidationRuleType(Enum):
    """Types of validation rules."""
    
    REQUIRED = auto()           # Field must be present and non-empty
    TYPE = auto()              # Field must be of specific type
    RANGE = auto()             # Numeric field must be within range
    LENGTH = auto()            # String/list length constraints
    PATTERN = auto()           # Regex pattern matching
    ENUM = auto()              # Value must be from enumerated set
    CUSTOM = auto()            # Custom validation function
    REFERENCE = auto()         # Reference to another entity must exist
    DEPENDENCY = auto()        # Field dependency validation
    UNIQUENESS = auto()        # Value must be unique across dataset
    TEMPORAL = auto()          # Temporal consistency validation
    STATISTICAL = auto()       # Statistical outlier detection


class DataCategory(Enum):
    """Categories of data for different validation approaches."""
    
    INDICATOR = auto()         # Social/economic indicators
    INSTITUTION = auto()       # Institutional data
    ACTOR = auto()            # Actor/stakeholder data
    FLOW = auto()             # Flow relationships
    MATRIX_CELL = auto()      # Matrix cell data
    RELATIONSHIP = auto()     # Entity relationships
    TEMPORAL = auto()         # Time-series data
    METADATA = auto()         # Metadata and configuration


@dataclass
class ValidationContext:
    """Context information for validation processes."""
    
    data_category: DataCategory
    validation_timestamp: datetime = field(default_factory=datetime.now)
    source_system: str = ""
    validation_level: str = "standard"  # standard, strict, lenient
    
    # Context data
    related_entities: Dict[str, Set[uuid.UUID]] = field(default_factory=dict)
    historical_data: Dict[str, List[Any]] = field(default_factory=dict)
    statistical_context: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    is_valid: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 1.0
    confidence_level: float = 1.0
    
    # Detailed results
    field_results: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Performance metrics
    validation_time_ms: float = 0.0
    rules_applied: int = 0


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, rule_name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.rule_name = rule_name
        self.severity = severity
        self.description = ""
        
    @abstractmethod
    def validate(self, value: Any, context: ValidationContext) -> List[Dict[str, Any]]:
        """Validate a value and return violations."""
        pass
    
    def get_violation(self, message: str, **kwargs) -> Dict[str, Any]:
        """Create a standardized violation record."""
        return {
            'rule_name': self.rule_name,
            'severity': self.severity,
            'message': message,
            'timestamp': datetime.now(),
            **kwargs
        }


class RequiredFieldRule(ValidationRule):
    """Validation rule for required fields."""
    
    def __init__(self, field_name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(f"required_{field_name}", severity)
        self.field_name = field_name
        self.description = f"Field {field_name} is required"
    
    def validate(self, value: Any, context: ValidationContext) -> List[Dict[str, Any]]:
        violations = []
        
        if value is None or (isinstance(value, str) and value.strip() == ""):
            violations.append(self.get_violation(
                f"Required field '{self.field_name}' is missing or empty"
            ))
        
        return violations


class TypeValidationRule(ValidationRule):
    """Validation rule for type checking."""
    
    def __init__(self, field_name: str, expected_type: Type, 
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(f"type_{field_name}", severity)
        self.field_name = field_name
        self.expected_type = expected_type
        self.description = f"Field {field_name} must be of type {expected_type.__name__}"
    
    def validate(self, value: Any, context: ValidationContext) -> List[Dict[str, Any]]:
        violations = []
        
        if value is not None and not isinstance(value, self.expected_type):
            violations.append(self.get_violation(
                f"Field '{self.field_name}' must be of type {self.expected_type.__name__}, got {type(value).__name__}"
            ))
        
        return violations


class RangeValidationRule(ValidationRule):
    """Validation rule for numeric ranges."""
    
    def __init__(self, field_name: str, min_value: Optional[float] = None, 
                 max_value: Optional[float] = None, severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(f"range_{field_name}", severity)
        self.field_name = field_name
        self.min_value = min_value
        self.max_value = max_value
        self.description = f"Field {field_name} must be within specified range"
    
    def validate(self, value: Any, context: ValidationContext) -> List[Dict[str, Any]]:
        violations = []
        
        if value is not None and isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                violations.append(self.get_violation(
                    f"Field '{self.field_name}' value {value} is below minimum {self.min_value}"
                ))
            
            if self.max_value is not None and value > self.max_value:
                violations.append(self.get_violation(
                    f"Field '{self.field_name}' value {value} is above maximum {self.max_value}"
                ))
        
        return violations


class PatternValidationRule(ValidationRule):
    """Validation rule for regex pattern matching."""
    
    def __init__(self, field_name: str, pattern: str, 
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(f"pattern_{field_name}", severity)
        self.field_name = field_name
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        self.description = f"Field {field_name} must match pattern {pattern}"
    
    def validate(self, value: Any, context: ValidationContext) -> List[Dict[str, Any]]:
        violations = []
        
        if value is not None and isinstance(value, str):
            if not self.compiled_pattern.match(value):
                violations.append(self.get_violation(
                    f"Field '{self.field_name}' value '{value}' does not match required pattern"
                ))
        
        return violations


class ReferenceValidationRule(ValidationRule):
    """Validation rule for entity references."""
    
    def __init__(self, field_name: str, reference_type: str,
                 severity: ValidationSeverity = ValidationSeverity.ERROR):
        super().__init__(f"reference_{field_name}", severity)
        self.field_name = field_name
        self.reference_type = reference_type
        self.description = f"Field {field_name} must reference existing {reference_type}"
    
    def validate(self, value: Any, context: ValidationContext) -> List[Dict[str, Any]]:
        violations = []
        
        if value is not None:
            # Check if reference exists in context
            related_entities = context.related_entities.get(self.reference_type, set())
            
            if isinstance(value, uuid.UUID):
                if value not in related_entities:
                    violations.append(self.get_violation(
                        f"Field '{self.field_name}' references non-existent {self.reference_type}: {value}"
                    ))
            elif isinstance(value, list):
                for ref_id in value:
                    if isinstance(ref_id, uuid.UUID) and ref_id not in related_entities:
                        violations.append(self.get_violation(
                            f"Field '{self.field_name}' contains non-existent {self.reference_type} reference: {ref_id}"
                        ))
        
        return violations


class StatisticalOutlierRule(ValidationRule):
    """Validation rule for statistical outlier detection."""
    
    def __init__(self, field_name: str, z_score_threshold: float = 3.0,
                 severity: ValidationSeverity = ValidationSeverity.WARNING):
        super().__init__(f"outlier_{field_name}", severity)
        self.field_name = field_name
        self.z_score_threshold = z_score_threshold
        self.description = f"Field {field_name} statistical outlier detection"
    
    def validate(self, value: Any, context: ValidationContext) -> List[Dict[str, Any]]:
        violations = []
        
        if value is not None and isinstance(value, (int, float)):
            # Get historical data for statistical analysis
            historical_values = context.historical_data.get(self.field_name, [])
            
            if len(historical_values) >= 10:  # Need sufficient data for analysis
                mean_val = sum(historical_values) / len(historical_values)
                variance = sum((x - mean_val) ** 2 for x in historical_values) / len(historical_values)
                std_dev = variance ** 0.5
                
                if std_dev > 0:
                    z_score = abs(value - mean_val) / std_dev
                    
                    if z_score > self.z_score_threshold:
                        violations.append(self.get_violation(
                            f"Field '{self.field_name}' value {value} is a statistical outlier (z-score: {z_score:.2f})"
                        ))
        
        return violations


@dataclass
class ValidationSchema:
    """Configurable validation schema for different data types."""
    
    schema_name: str
    data_category: DataCategory
    rules: List[ValidationRule] = field(default_factory=list)
    
    # Schema configuration
    strict_mode: bool = False
    allow_additional_fields: bool = True
    required_fields: Set[str] = field(default_factory=set)
    optional_fields: Set[str] = field(default_factory=set)
    
    # Field-specific configurations
    field_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule to the schema."""
        self.rules.append(rule)
    
    def add_required_field(self, field_name: str, field_type: Type = str, **kwargs) -> None:
        """Add a required field with type validation."""
        self.required_fields.add(field_name)
        self.add_rule(RequiredFieldRule(field_name))
        self.add_rule(TypeValidationRule(field_name, field_type))
        
        # Add additional constraints if provided
        if 'min_value' in kwargs or 'max_value' in kwargs:
            self.add_rule(RangeValidationRule(field_name, 
                                            kwargs.get('min_value'), 
                                            kwargs.get('max_value')))
        
        if 'pattern' in kwargs:
            self.add_rule(PatternValidationRule(field_name, kwargs['pattern']))
    
    def validate_data(self, data: Dict[str, Any], context: ValidationContext) -> ValidationResult:
        """Validate data against this schema."""
        start_time = datetime.now()
        violations = []
        field_results = {}
        
        # Check for required fields
        for field_name in self.required_fields:
            if field_name not in data:
                violations.append({
                    'rule_name': f'required_{field_name}',
                    'severity': ValidationSeverity.ERROR,
                    'message': f'Required field {field_name} is missing',
                    'field_name': field_name
                })
                field_results[field_name] = False
        
        # Validate each field with applicable rules
        for rule in self.rules:
            field_name = getattr(rule, 'field_name', None)
            if field_name and field_name in data:
                field_violations = rule.validate(data[field_name], context)
                violations.extend(field_violations)
                
                if field_violations:
                    field_results[field_name] = False
                else:
                    field_results.setdefault(field_name, True)
        
        # Check for additional fields in strict mode
        if self.strict_mode and not self.allow_additional_fields:
            allowed_fields = self.required_fields | self.optional_fields
            for field_name in data.keys():
                if field_name not in allowed_fields:
                    violations.append({
                        'rule_name': 'additional_field',
                        'severity': ValidationSeverity.WARNING,
                        'message': f'Additional field {field_name} not allowed in strict mode',
                        'field_name': field_name
                    })
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(violations)
        
        # Calculate validation time
        validation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ValidationResult(
            is_valid=len([v for v in violations if v['severity'] in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0,
            violations=violations,
            quality_score=quality_score,
            field_results=field_results,
            validation_time_ms=validation_time,
            rules_applied=len(self.rules)
        )
    
    def _calculate_quality_score(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate quality score based on violations."""
        if not violations:
            return 1.0
        
        severity_weights = {
            ValidationSeverity.INFO: 0.0,
            ValidationSeverity.WARNING: 0.1,
            ValidationSeverity.ERROR: 0.3,
            ValidationSeverity.CRITICAL: 0.5
        }
        
        total_penalty = sum(severity_weights.get(v['severity'], 0.3) for v in violations)
        max_penalty = len(violations) * 0.5
        
        return max(0.0, 1.0 - (total_penalty / max_penalty)) if max_penalty > 0 else 1.0


@dataclass
class IntegrityChecker:
    """Cross-reference and relationship integrity validation."""
    
    entity_repositories: Dict[str, Set[uuid.UUID]] = field(default_factory=dict)
    relationship_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def register_entities(self, entity_type: str, entity_ids: Set[uuid.UUID]) -> None:
        """Register entities for reference validation."""
        self.entity_repositories[entity_type] = entity_ids
    
    def add_relationship_rule(self, source_type: str, target_type: str, 
                            relationship_type: str, required: bool = True) -> None:
        """Add a relationship integrity rule."""
        self.relationship_rules.append({
            'source_type': source_type,
            'target_type': target_type,
            'relationship_type': relationship_type,
            'required': required
        })
    
    def check_reference_integrity(self, entity_id: uuid.UUID, entity_type: str,
                                references: Dict[str, List[uuid.UUID]]) -> List[Dict[str, Any]]:
        """Check integrity of entity references."""
        violations = []
        
        for ref_type, ref_ids in references.items():
            if ref_type in self.entity_repositories:
                valid_ids = self.entity_repositories[ref_type]
                
                for ref_id in ref_ids:
                    if ref_id not in valid_ids:
                        violations.append({
                            'type': 'invalid_reference',
                            'entity_id': entity_id,
                            'entity_type': entity_type,
                            'reference_type': ref_type,
                            'invalid_reference_id': ref_id,
                            'message': f'{entity_type} {entity_id} has invalid {ref_type} reference: {ref_id}'
                        })
        
        return violations
    
    def check_relationship_integrity(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check integrity of relationships between entities."""
        violations = []
        
        # Group relationships by type
        relationship_map = {}
        for rel in relationships:
            rel_type = rel.get('relationship_type')
            if rel_type not in relationship_map:
                relationship_map[rel_type] = []
            relationship_map[rel_type].append(rel)
        
        # Check against relationship rules
        for rule in self.relationship_rules:
            if rule['required'] and rule['relationship_type'] not in relationship_map:
                violations.append({
                    'type': 'missing_relationship',
                    'source_type': rule['source_type'],
                    'target_type': rule['target_type'],
                    'relationship_type': rule['relationship_type'],
                    'message': f"Required relationship {rule['relationship_type']} between {rule['source_type']} and {rule['target_type']} is missing"
                })
        
        return violations


@dataclass
class DataQualityAnalyzer:
    """Comprehensive data quality assessment and scoring."""
    
    quality_dimensions: Dict[str, float] = field(default_factory=dict)
    benchmark_data: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize quality dimensions with default weights
        if not self.quality_dimensions:
            self.quality_dimensions = {
                'completeness': 0.2,      # How complete is the data
                'accuracy': 0.25,         # How accurate is the data
                'consistency': 0.2,       # Internal consistency
                'timeliness': 0.15,       # How current is the data
                'validity': 0.2           # Adherence to business rules
            }
    
    def analyze_completeness(self, data: Dict[str, Any], required_fields: Set[str]) -> float:
        """Analyze data completeness."""
        if not required_fields:
            return 1.0
        
        present_fields = sum(1 for field in required_fields if data.get(field) is not None)
        completeness_score = present_fields / len(required_fields)
        
        return completeness_score
    
    def analyze_accuracy(self, data: Dict[str, Any], validation_result: ValidationResult) -> float:
        """Analyze data accuracy based on validation results."""
        if validation_result.rules_applied == 0:
            return 1.0
        
        # Accuracy is inversely related to validation violations
        error_violations = len([v for v in validation_result.violations 
                              if v['severity'] in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]])
        
        accuracy_score = max(0.0, 1.0 - (error_violations / validation_result.rules_applied))
        return accuracy_score
    
    def analyze_consistency(self, data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> float:
        """Analyze internal consistency."""
        if not historical_data:
            return 1.0
        
        # Simple consistency check - compare field presence and types
        consistency_score = 1.0
        
        if historical_data:
            reference_data = historical_data[0]
            
            # Check field presence consistency
            current_fields = set(data.keys())
            reference_fields = set(reference_data.keys())
            
            field_similarity = len(current_fields & reference_fields) / len(current_fields | reference_fields)
            consistency_score *= field_similarity
            
            # Check type consistency for common fields
            common_fields = current_fields & reference_fields
            type_matches = 0
            
            for field in common_fields:
                if type(data[field]) == type(reference_data[field]):
                    type_matches += 1
            
            if common_fields:
                type_consistency = type_matches / len(common_fields)
                consistency_score *= type_consistency
        
        return consistency_score
    
    def analyze_timeliness(self, data: Dict[str, Any], timestamp_field: str = 'timestamp') -> float:
        """Analyze data timeliness."""
        timestamp = data.get(timestamp_field)
        
        if not timestamp:
            return 0.5  # Unknown timeliness
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                return 0.5
        
        if isinstance(timestamp, datetime):
            age = datetime.now() - timestamp
            
            # Score based on age (fresher data gets higher score)
            if age.total_seconds() < 3600:  # Less than 1 hour
                return 1.0
            elif age.total_seconds() < 24 * 3600:  # Less than 1 day
                return 0.8
            elif age.total_seconds() < 7 * 24 * 3600:  # Less than 1 week
                return 0.6
            elif age.total_seconds() < 30 * 24 * 3600:  # Less than 1 month
                return 0.4
            else:
                return 0.2
        
        return 0.5
    
    def analyze_validity(self, validation_result: ValidationResult) -> float:
        """Analyze data validity from validation results."""
        return validation_result.quality_score
    
    def calculate_overall_quality_score(self, data: Dict[str, Any], validation_result: ValidationResult,
                                      required_fields: Set[str] = None, 
                                      historical_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate comprehensive data quality score."""
        
        required_fields = required_fields or set()
        historical_data = historical_data or []
        
        # Calculate individual dimension scores
        completeness_score = self.analyze_completeness(data, required_fields)
        accuracy_score = self.analyze_accuracy(data, validation_result)
        consistency_score = self.analyze_consistency(data, historical_data)
        timeliness_score = self.analyze_timeliness(data)
        validity_score = self.analyze_validity(validation_result)
        
        # Calculate weighted overall score
        dimension_scores = {
            'completeness': completeness_score,
            'accuracy': accuracy_score,
            'consistency': consistency_score,
            'timeliness': timeliness_score,
            'validity': validity_score
        }
        
        overall_score = sum(score * self.quality_dimensions[dimension] 
                           for dimension, score in dimension_scores.items())
        
        return {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'quality_level': self._get_quality_level(overall_score),
            'recommendations': self._generate_recommendations(dimension_scores)
        }
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level from score."""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.7:
            return 'Good'
        elif score >= 0.5:
            return 'Fair'
        else:
            return 'Poor'
    
    def _generate_recommendations(self, dimension_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []
        
        for dimension, score in dimension_scores.items():
            if score < 0.7:
                if dimension == 'completeness':
                    recommendations.append("Improve data completeness by ensuring all required fields are populated")
                elif dimension == 'accuracy':
                    recommendations.append("Enhance data accuracy by implementing stricter validation rules")
                elif dimension == 'consistency':
                    recommendations.append("Improve consistency by standardizing data formats and structures")
                elif dimension == 'timeliness':
                    recommendations.append("Improve timeliness by increasing data refresh frequency")
                elif dimension == 'validity':
                    recommendations.append("Enhance validity by reviewing and updating business rules")
        
        return recommendations


@dataclass
class AdvancedDataValidator(Node):
    """
    Main validation engine with comprehensive schema support and quality analysis.
    
    This class orchestrates all validation activities including schema validation,
    integrity checking, and quality analysis for SFM data structures.
    """
    
    # Validation schemas for different data types
    schemas: Dict[str, ValidationSchema] = field(default_factory=dict)
    integrity_checker: IntegrityChecker = field(default_factory=IntegrityChecker)
    quality_analyzer: DataQualityAnalyzer = field(default_factory=DataQualityAnalyzer)
    
    # Validation configuration
    default_validation_level: str = "standard"
    enable_quality_analysis: bool = True
    enable_integrity_checking: bool = True
    
    # Validation statistics
    validations_performed: int = 0
    validation_errors: int = 0
    average_quality_score: float = 0.0
    
    def register_schema(self, schema: ValidationSchema) -> None:
        """Register a validation schema."""
        self.schemas[schema.schema_name] = schema
        logger.info(f"Registered validation schema: {schema.schema_name}")
    
    def register_entities_for_integrity_check(self, entity_type: str, entity_ids: Set[uuid.UUID]) -> None:
        """Register entities for integrity checking."""
        self.integrity_checker.register_entities(entity_type, entity_ids)
    
    def validate_record(self, data: Dict[str, Any], schema_name: str, 
                       context: Optional[ValidationContext] = None) -> Dict[str, Any]:
        """Validate a data record using specified schema."""
        
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown validation schema: {schema_name}")
        
        schema = self.schemas[schema_name]
        context = context or ValidationContext(data_category=schema.data_category)
        
        # Perform schema validation
        validation_result = schema.validate_data(data, context)
        
        # Perform quality analysis if enabled
        quality_analysis = None
        if self.enable_quality_analysis:
            quality_analysis = self.quality_analyzer.calculate_overall_quality_score(
                data, validation_result, schema.required_fields
            )
        
        # Update statistics
        self.validations_performed += 1
        if not validation_result.is_valid:
            self.validation_errors += 1
        
        # Update average quality score
        if quality_analysis:
            self.average_quality_score = (
                (self.average_quality_score * (self.validations_performed - 1) + 
                 quality_analysis['overall_score']) / self.validations_performed
            )
        
        return {
            'validation_result': validation_result,
            'quality_analysis': quality_analysis,
            'validation_timestamp': datetime.now(),
            'schema_used': schema_name
        }
    
    def create_standard_schemas(self) -> None:
        """Create standard validation schemas for SFM data types."""
        
        # Indicator validation schema
        indicator_schema = ValidationSchema(
            schema_name="sfm_indicator",
            data_category=DataCategory.INDICATOR
        )
        indicator_schema.add_required_field("label", str)
        indicator_schema.add_required_field("value_category", str)
        indicator_schema.add_required_field("current_value", (int, float))
        indicator_schema.add_required_field("measurement_unit", str)
        
        # Add range validation for common indicator types
        indicator_schema.add_rule(RangeValidationRule("current_value", -1000000, 1000000, 
                                                    ValidationSeverity.WARNING))
        
        self.register_schema(indicator_schema)
        
        # Institution validation schema
        institution_schema = ValidationSchema(
            schema_name="sfm_institution",
            data_category=DataCategory.INSTITUTION
        )
        institution_schema.add_required_field("label", str)
        institution_schema.add_required_field("institution_type", str)
        institution_schema.add_rule(PatternValidationRule("label", r"^[A-Za-z0-9\s\-_]+$"))
        
        self.register_schema(institution_schema)
        
        # Actor validation schema
        actor_schema = ValidationSchema(
            schema_name="sfm_actor",
            data_category=DataCategory.ACTOR
        )
        actor_schema.add_required_field("label", str)
        actor_schema.add_rule(PatternValidationRule("label", r"^[A-Za-z0-9\s\-_]+$"))
        
        self.register_schema(actor_schema)
        
        # Matrix cell validation schema
        matrix_cell_schema = ValidationSchema(
            schema_name="sfm_matrix_cell",
            data_category=DataCategory.MATRIX_CELL
        )
        matrix_cell_schema.add_required_field("row_label", str)
        matrix_cell_schema.add_required_field("column_label", str)
        matrix_cell_schema.add_required_field("cell_type", str)
        
        self.register_schema(matrix_cell_schema)
        
        logger.info("Created standard SFM validation schemas")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'validations_performed': self.validations_performed,
            'validation_errors': self.validation_errors,
            'error_rate': self.validation_errors / self.validations_performed if self.validations_performed > 0 else 0,
            'average_quality_score': self.average_quality_score,
            'registered_schemas': len(self.schemas),
            'schema_names': list(self.schemas.keys())
        }