"""
Enhanced temporal sequencing and coordination for Social Fabric Matrix modeling.

This module implements Hayden's temporal coordination methodology for managing
multi-stage processes, temporal dependencies, and time-based policy sequences
in the Social Fabric Matrix framework.

Key Components:
- TemporalSequence: Multi-stage temporal process coordination
- SequenceCoordinator: Manages coordination between multiple sequences
- TemporalConstraint: Time-based constraints and dependencies
- PolicySequence: Specific implementation for policy processes
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from datetime import datetime, timedelta
from enum import Enum, auto

from models.base_nodes import Node
from models.meta_entities import TimeSlice
from models.metadata_models import TemporalDynamics
from models.sfm_enums import (
    SequenceStage,
    AdjustmentType,
    ProblemSolvingStage,
    PolicyInstrumentType,
)

class SequenceStatus(Enum):
    """Status of a temporal sequence."""

    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class CoordinationType(Enum):
    """Types of coordination between sequences."""

    SEQUENTIAL = auto()      # One after another
    PARALLEL = auto()        # Simultaneous execution
    CONDITIONAL = auto()     # Depends on conditions
    SYNCHRONIZED = auto()    # Coordinated timing
    HIERARCHICAL = auto()    # Parent-child relationships

class ConstraintType(Enum):
    """Types of temporal constraints."""

    PRECEDENCE = auto()      # Must happen before/after
    DURATION = auto()        # Time duration limits
    DEADLINE = auto()        # Must complete by date
    RESOURCE = auto()        # Resource availability timing
    SYNCHRONIZATION = auto() # Must happen at same time
    FREQUENCY = auto()       # Rate of occurrence

@dataclass
class TemporalConstraint:
    """Represents temporal constraints between sequence stages or elements."""

    constraint_type: ConstraintType
    source_element: uuid.UUID  # What this constraint applies to
    target_element: Optional[uuid.UUID] = None  # What it relates to (if applicable)

    # Timing specifications
    min_duration: Optional[timedelta] = None
    max_duration: Optional[timedelta] = None
    exact_duration: Optional[timedelta] = None
    deadline: Optional[datetime] = None

    # Relationship specifications
    time_offset: Optional[timedelta] = None  # Offset from target element
    synchronization_window: Optional[timedelta] = None  # Allowed timing variance

    # Constraint properties
    is_hard_constraint: bool = True  # False for soft constraints
    priority: float = 1.0  # Constraint priority (0-1)
    violation_penalty: Optional[float] = None  # Cost of violating constraint

    # Metadata
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    description: str = ""

    def is_satisfied(self, start_time: datetime, end_time: Optional[datetime] = None) -> bool:
        """Check if this constraint is satisfied given timing."""
        now = datetime.now()

        if self.constraint_type == ConstraintType.DEADLINE:
            if self.deadline and (end_time or now) > self.deadline:
                return False

        elif self.constraint_type == ConstraintType.DURATION:
            if end_time:
                duration = end_time - start_time

                if self.min_duration and duration < self.min_duration:
                    return False
                if self.max_duration and duration > self.max_duration:
                    return False
                if self.exact_duration and abs(duration - self.exact_duration).total_seconds() > 60:
                    return False

        return True

    def calculate_violation_severity(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None) -> float:
        """Calculate severity of constraint violation (0 = satisfied, 1 = max violation)."""
        if self.is_satisfied(start_time, end_time):
            return 0.0

        severity = 0.5  # Default moderate violation

        if self.constraint_type == ConstraintType.DEADLINE and self.deadline:
            actual_end = end_time or datetime.now()
            if actual_end > self.deadline:
                delay = actual_end - self.deadline
                # Normalize to hours - 24 hours = max severity
                severity = min(delay.total_seconds() / (24 * 3600), 1.0)

        elif self.constraint_type == ConstraintType.DURATION and end_time:
            duration = end_time - start_time

            if self.min_duration and duration < self.min_duration:
                shortfall = self.min_duration - duration
                severity = min(shortfall.total_seconds() / self.min_duration.total_seconds(), 1.0)

            elif self.max_duration and duration > self.max_duration:
                excess = duration - self.max_duration
                severity = min(excess.total_seconds() / self.max_duration.total_seconds(), 1.0)

        return severity

@dataclass
class SequenceStageExecution:
    """Execution details for a specific stage in a temporal sequence."""

    stage: SequenceStage
    planned_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None

    status: SequenceStatus = SequenceStatus.NOT_STARTED
    progress: float = 0.0  # Completion percentage (0-1)

    # Resource and actor information
    assigned_actors: List[uuid.UUID] = field(default_factory=list)
    required_resources: List[uuid.UUID] = field(default_factory=list)
    allocated_resources: List[uuid.UUID] = field(default_factory=list)

    # Dependencies
    predecessor_stages: List[uuid.UUID] = field(default_factory=list)
    successor_stages: List[uuid.UUID] = field(default_factory=list)

    # Performance metrics
    efficiency_score: Optional[float] = None  # How well stage executed (0-1)
    quality_score: Optional[float] = None  # Quality of stage output (0-1)

    # Issues and adjustments
    encountered_issues: List[str] = field(default_factory=list)
    applied_adjustments: List[AdjustmentType] = field(default_factory=list)

    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def calculate_duration(self) -> Optional[timedelta]:
        """Calculate actual duration if both start and end are set."""
        if self.actual_start and self.actual_end:
            return self.actual_end - self.actual_start
        return None

    def calculate_delay(self) -> Optional[timedelta]:
        """Calculate delay compared to planned schedule."""
        if self.planned_end and self.actual_end:
            return self.actual_end - self.planned_end
        elif self.planned_end and not self.actual_end and self.status == SequenceStatus.IN_PROGRESS:
            return datetime.now() - self.planned_end
        return None

    def is_behind_schedule(self) -> bool:
        """Check if stage is behind planned schedule."""
        delay = self.calculate_delay()
        return delay is not None and delay.total_seconds() > 0

    def start_stage(self) -> None:
        """Mark stage as started."""
        self.actual_start = datetime.now()
        self.status = SequenceStatus.IN_PROGRESS

    def complete_stage(self, quality_score: Optional[float] = None) -> None:
        """Mark stage as completed."""
        self.actual_end = datetime.now()
        self.status = SequenceStatus.COMPLETED
        self.progress = 1.0
        if quality_score is not None:
            self.quality_score = quality_score

@dataclass
class TemporalSequence(Node):
    """Multi-stage temporal process with coordination capabilities."""

    sequence_type: str  # Type of sequence (policy, project, process, etc.)
    stages: Dict[uuid.UUID, SequenceStageExecution] = field(default_factory=dict)
    stage_order: List[uuid.UUID] = field(default_factory=list)  # Ordered list of stage IDs

    # Overall sequence properties
    status: SequenceStatus = SequenceStatus.NOT_STARTED
    priority: float = 0.5  # Sequence priority (0-1)

    # Timing
    planned_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None

    # Constraints and coordination
    temporal_constraints: List[TemporalConstraint] = field(default_factory=list)
    coordination_requirements: List[uuid.UUID] = field(default_factory=list)  # Other sequences to coordinate with

    # Performance tracking
    overall_progress: float = 0.0  # Overall completion (0-1)
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    # SFM integration
    affected_matrix_cells: List[uuid.UUID] = field(default_factory=list)
    institutional_context: List[uuid.UUID] = field(default_factory=list)

    def add_stage(self, stage: SequenceStageExecution, position: Optional[int] = None) -> None:
        """Add a stage to the sequence."""
        self.stages[stage.id] = stage

        if position is not None and 0 <= position <= len(self.stage_order):
            self.stage_order.insert(position, stage.id)
        else:
            self.stage_order.append(stage.id)

    def get_current_stage(self) -> Optional[SequenceStageExecution]:
        """Get the currently active stage."""
        for stage_id in self.stage_order:
            stage = self.stages.get(stage_id)
            if stage and stage.status == SequenceStatus.IN_PROGRESS:
                return stage
        return None

    def get_next_stage(self) -> Optional[SequenceStageExecution]:
        """Get the next stage to be executed."""
        for stage_id in self.stage_order:
            stage = self.stages.get(stage_id)
            if stage and stage.status == SequenceStatus.NOT_STARTED:
                return stage
        return None

    def advance_to_next_stage(self) -> bool:
        """Advance sequence to the next stage if possible."""
        current = self.get_current_stage()
        if current:
            current.complete_stage()

        next_stage = self.get_next_stage()
        if next_stage:
            # Check if prerequisites are met
            if self._can_start_stage(next_stage):
                next_stage.start_stage()
                return True

        # Check if sequence is complete
        if all(self.stages[sid].status == SequenceStatus.COMPLETED for sid in self.stage_order):
            self.status = SequenceStatus.COMPLETED
            self.actual_end = datetime.now()

        return False

    def _can_start_stage(self, stage: SequenceStageExecution) -> bool:
        """Check if a stage can be started based on prerequisites."""
        # Check predecessor stages
        for pred_id in stage.predecessor_stages:
            if pred_id in self.stages:
                pred_stage = self.stages[pred_id]
                if pred_stage.status != SequenceStatus.COMPLETED:
                    return False

        # Check temporal constraints
        for constraint in self.temporal_constraints:
            if constraint.source_element == stage.id:
                if not constraint.is_satisfied(datetime.now()):
                    return False

        # Check resource availability (simplified)
        # In a full implementation, this would check actual resource availability
        return len(stage.allocated_resources) >= len(stage.required_resources) * 0.8

    def calculate_overall_progress(self) -> float:
        """Calculate overall sequence progress."""
        if not self.stages:
            return 0.0

        total_progress = sum(stage.progress for stage in self.stages.values())
        self.overall_progress = total_progress / len(self.stages)
        return self.overall_progress

    def get_critical_path(self) -> List[uuid.UUID]:
        """Get the critical path through the sequence (simplified)."""
        # This is a simplified version - a full implementation would use
        # proper critical path method (CPM) calculations
        critical_path = []

        # Find the longest path through dependencies
        for stage_id in self.stage_order:
            stage = self.stages.get(stage_id)
            if stage and (not stage.predecessor_stages or
                         any(pid in critical_path for pid in stage.predecessor_stages)):
                critical_path.append(stage_id)

        return critical_path

    def add_temporal_constraint(self, constraint: TemporalConstraint) -> None:
        """Add a temporal constraint to the sequence."""
        self.temporal_constraints.append(constraint)

    def check_constraint_violations(self) -> List[Tuple[TemporalConstraint, float]]:
        """Check for constraint violations and return severity."""
        violations = []

        for constraint in self.temporal_constraints:
            if constraint.source_element in self.stages:
                stage = self.stages[constraint.source_element]
                severity = constraint.calculate_violation_severity(
                    stage.actual_start or datetime.now(),
                    stage.actual_end
                )
                if severity > 0:
                    violations.append((constraint, severity))

        return violations

    def estimate_completion_time(self) -> Optional[datetime]:
        """Estimate when the sequence will be completed."""
        if self.status == SequenceStatus.COMPLETED:
            return self.actual_end

        current_time = datetime.now()
        remaining_time = timedelta()

        for stage_id in self.stage_order:
            stage = self.stages.get(stage_id)
            if not stage:
                continue

            if stage.status == SequenceStatus.NOT_STARTED:
                # Estimate stage duration based on planned duration or average
                if stage.planned_start and stage.planned_end:
                    stage_duration = stage.planned_end - stage.planned_start
                else:
                    # Default estimate
                    stage_duration = timedelta(days=7)

                remaining_time += stage_duration

            elif stage.status == SequenceStatus.IN_PROGRESS:
                # Estimate remaining time based on progress
                if stage.planned_start and stage.planned_end:
                    total_duration = stage.planned_end - stage.planned_start
                    remaining_duration = total_duration * (1 - stage.progress)
                    remaining_time += remaining_duration
                else:
                    # Default remaining time estimate
                    remaining_time += timedelta(days=3)

        return current_time + remaining_time

@dataclass
class PolicySequence(TemporalSequence):
    """Specialized temporal sequence for policy implementation."""

    policy_id: uuid.UUID
    policy_instruments: List[PolicyInstrumentType] = field(default_factory=list)
    target_outcomes: List[str] = field(default_factory=list)

    # Policy-specific stages mapping
    problem_identification: Optional[uuid.UUID] = None
    policy_formulation: Optional[uuid.UUID] = None
    decision_making: Optional[uuid.UUID] = None
    implementation: Optional[uuid.UUID] = None
    evaluation: Optional[uuid.UUID] = None

    # Policy context
    stakeholder_groups: List[uuid.UUID] = field(default_factory=list)
    regulatory_context: List[uuid.UUID] = field(default_factory=list)

    def initialize_policy_stages(self) -> None:
        """Initialize standard policy implementation stages."""
        stages_data = [
            (ProblemSolvingStage.IDENTIFICATION, "Problem Identification"),
            (ProblemSolvingStage.ALTERNATIVE_GENERATION, "Policy Formulation"),
            (ProblemSolvingStage.ALTERNATIVE_EVALUATION, "Decision Making"),
            (ProblemSolvingStage.IMPLEMENTATION, "Implementation"),
            (ProblemSolvingStage.EVALUATION, "Evaluation")
        ]

        previous_stage_id = None

        for i, (stage_enum, stage_name) in enumerate(stages_data):
            stage_exec = SequenceStageExecution(
                stage=SequenceStage.INITIATION,  # Map to general sequence stage
                planned_start=datetime.now() + timedelta(days=i*30),
                planned_end=datetime.now() + timedelta(days=(i+1)*30)
            )

            if previous_stage_id:
                stage_exec.predecessor_stages.append(previous_stage_id)

            self.add_stage(stage_exec)

            # Set specific policy stage references
            if i == 0:
                self.problem_identification = stage_exec.id
            elif i == 1:
                self.policy_formulation = stage_exec.id
            elif i == 2:
                self.decision_making = stage_exec.id
            elif i == 3:
                self.implementation = stage_exec.id
            elif i == 4:
                self.evaluation = stage_exec.id

            previous_stage_id = stage_exec.id

@dataclass
class SequenceCoordinator(Node):
    """Manages coordination between multiple temporal sequences."""

    managed_sequences: Dict[uuid.UUID, TemporalSequence] = field(default_factory=dict)
    coordination_rules: List[CoordinationRule] = field(default_factory=list)

    # Coordination state
    active_coordinations: Dict[str, Any] = field(default_factory=dict)
    coordination_conflicts: List[str] = field(default_factory=list)

    # Performance metrics
    coordination_efficiency: Optional[float] = None
    sequence_synchronization_score: Optional[float] = None

    def add_sequence(self, sequence: TemporalSequence) -> None:
        """Add a sequence to be coordinated."""
        self.managed_sequences[sequence.id] = sequence

    def add_coordination_rule(self, rule: CoordinationRule) -> None:
        """Add a coordination rule."""
        self.coordination_rules.append(rule)

    def coordinate_sequences(self) -> Dict[str, Any]:
        """Execute coordination logic across all managed sequences."""
        coordination_results = {}

        # Check for conflicts
        conflicts = self._detect_conflicts()
        coordination_results['conflicts'] = conflicts

        # Apply coordination rules
        applied_rules = self._apply_coordination_rules()
        coordination_results['applied_rules'] = applied_rules

        # Calculate synchronization
        sync_score = self._calculate_synchronization_score()
        coordination_results['synchronization_score'] = sync_score
        self.sequence_synchronization_score = sync_score

        return coordination_results

    def _detect_conflicts(self) -> List[str]:
        """Detect conflicts between sequences."""
        conflicts = []

        sequences = list(self.managed_sequences.values())

        for i, seq1 in enumerate(sequences):
            for seq2 in sequences[i+1:]:
                # Check for resource conflicts
                seq1_resources = set()
                seq2_resources = set()

                for stage in seq1.stages.values():
                    seq1_resources.update(stage.required_resources)

                for stage in seq2.stages.values():
                    seq2_resources.update(stage.required_resources)

                common_resources = seq1_resources.intersection(seq2_resources)
                if common_resources:
                    conflicts.append(f"Resource conflict between {seq1.label} and {seq2.label}: {common_resources}")

        self.coordination_conflicts = conflicts
        return conflicts

    def _apply_coordination_rules(self) -> List[str]:
        """Apply coordination rules to managed sequences."""
        applied_rules = []

        for rule in self.coordination_rules:
            if rule.apply_rule(self.managed_sequences):
                applied_rules.append(f"Applied rule: {rule.description}")

        return applied_rules

    def _calculate_synchronization_score(self) -> float:
        """Calculate how well sequences are synchronized."""
        if len(self.managed_sequences) < 2:
            return 1.0

        # Calculate variance in progress across sequences
        progresses = [seq.calculate_overall_progress() for seq in self.managed_sequences.values()]

        if not progresses:
            return 0.0

        mean_progress = sum(progresses) / len(progresses)
        variance = sum((p - mean_progress) ** 2 for p in progresses) / len(progresses)

        # Convert variance to synchronization score (lower variance = better sync)
        max_variance = 0.25  # Maximum expected variance
        sync_score = max(0.0, 1.0 - (variance / max_variance))

        return sync_score

    def get_coordination_status(self) -> Dict[str, Any]:
        """Get overall coordination status."""
        return {
            'num_sequences': len(self.managed_sequences),
            'active_sequences': sum(1 for seq in self.managed_sequences.values()
                                  if seq.status == SequenceStatus.IN_PROGRESS),
            'completed_sequences': sum(1 for seq in self.managed_sequences.values()
                                     if seq.status == SequenceStatus.COMPLETED),
            'conflicts': len(self.coordination_conflicts),
            'synchronization_score': self.sequence_synchronization_score,
            'coordination_efficiency': self.coordination_efficiency
        }

@dataclass
class CoordinationRule:
    """Rule for coordinating between temporal sequences."""

    rule_type: CoordinationType
    source_sequence_id: uuid.UUID
    target_sequence_id: Optional[uuid.UUID] = None

    # Rule parameters
    time_offset: Optional[timedelta] = None
    condition_function: Optional[str] = None  # String representation of condition
    priority: float = 0.5

    # Rule metadata
    description: str = ""
    is_active: bool = True

    def apply_rule(self, sequences: Dict[uuid.UUID, TemporalSequence]) -> bool:
        """Apply this coordination rule to the given sequences."""
        if not self.is_active:
            return False

        source_seq = sequences.get(self.source_sequence_id)
        if not source_seq:
            return False

        if self.rule_type == CoordinationType.SEQUENTIAL:
            return self._apply_sequential_rule(source_seq, sequences)
        elif self.rule_type == CoordinationType.PARALLEL:
            return self._apply_parallel_rule(source_seq, sequences)
        elif self.rule_type == CoordinationType.SYNCHRONIZED:
            return self._apply_synchronized_rule(source_seq, sequences)

        return False

    def _apply_sequential_rule(self, source_seq: TemporalSequence,
                              sequences: Dict[uuid.UUID, TemporalSequence]) -> bool:
        """Apply sequential coordination rule."""
        if not self.target_sequence_id:
            return False

        target_seq = sequences.get(self.target_sequence_id)
        if not target_seq:
            return False

        # If source sequence is completed, start target sequence
        if (source_seq.status == SequenceStatus.COMPLETED and
            target_seq.status == SequenceStatus.NOT_STARTED):

            # Adjust target sequence start time
            start_time = datetime.now()
            if self.time_offset:
                start_time += self.time_offset

            target_seq.planned_start = start_time
            return True

        return False

    def _apply_parallel_rule(self, source_seq: TemporalSequence,
                            sequences: Dict[uuid.UUID, TemporalSequence]) -> bool:
        """Apply parallel coordination rule."""
        if not self.target_sequence_id:
            return False

        target_seq = sequences.get(self.target_sequence_id)
        if not target_seq:
            return False

        # Start both sequences at the same time if one starts
        if (source_seq.status == SequenceStatus.IN_PROGRESS and
            target_seq.status == SequenceStatus.NOT_STARTED):

            target_seq.planned_start = source_seq.actual_start
            return True

        return False

    def _apply_synchronized_rule(self, source_seq: TemporalSequence,
                                sequences: Dict[uuid.UUID, TemporalSequence]) -> bool:
        """Apply synchronized coordination rule."""
        # Synchronize sequence timing to maintain coordination
        # This is a simplified implementation
        if self.target_sequence_id:
            target_seq = sequences.get(self.target_sequence_id)
            if target_seq:
                # Adjust timing to maintain synchronization
                source_progress = source_seq.calculate_overall_progress()
                target_progress = target_seq.calculate_overall_progress()

                # If sequences are getting out of sync, adjust
                if abs(source_progress - target_progress) > 0.2:
                    return True

        return False
