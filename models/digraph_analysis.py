"""
Digraph analysis system for Social Fabric Matrix modeling.

This module implements Hayden's digraph methodology for analyzing institutional
dependencies, flows, and system structure in the Social Fabric Matrix framework.
The digraph represents the network of relationships between institutions and
criteria in the SFM.

Key Components:
- DigraphNode: Nodes in the SFM digraph representing institutions/criteria
- DigraphEdge: Edges representing dependencies and relationships
- SFMDigraph: The complete digraph structure with analysis methods
- NetworkAnalyzer: Tools for network analysis and centrality measures
"""

# pylint: disable=too-many-instance-attributes,too-many-public-methods
# Complex digraph analysis requires many attributes and methods for comprehensive network analysis

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any  # pylint: disable=unused-import
from datetime import datetime

from models.base_nodes import Node
from models.sfm_enums import (
    NetworkMetricType,
    DependencyStrength,
    DigraphNodeType,
    RelationshipKind,
)

@dataclass
class DigraphNode(Node):
    """Node in the SFM digraph representing institutions, criteria, or other entities."""

    node_type: DigraphNodeType = DigraphNodeType.INTERMEDIATE  # type: ignore[misc]
    influence_score: Optional[float] = None  # Overall influence in the system (0-1)
    dependency_level: Optional[float] = None  # How dependent this node is (0-1)
    centrality_measures: Dict[NetworkMetricType, float] = field(default_factory=dict)

    # Structural properties
    in_degree: int = 0  # Number of incoming edges
    out_degree: int = 0  # Number of outgoing edges
    reachability_score: Optional[float] = None  # How many nodes this can reach

    # SFM-specific attributes
    matrix_position: Optional[Tuple[int, int]] = None  # Position in SFM matrix
    delivery_capacity: Optional[float] = None  # Capacity to deliver to other nodes
    reception_capacity: Optional[float] = None  # Capacity to receive from other nodes

    def calculate_total_degree(self) -> int:
        """Calculate total degree (in + out)."""
        return self.in_degree + self.out_degree

    def calculate_degree_centrality(self, total_nodes: int) -> float:
        """Calculate degree centrality normalized by total nodes."""
        if total_nodes <= 1:
            return 0.0
        return self.calculate_total_degree() / (total_nodes - 1)

    def update_centrality_measure(self, metric_type: NetworkMetricType, value: float) -> None:
        """Update a specific centrality measure."""
        self.centrality_measures[metric_type] = value

    def get_centrality_measure(self, metric_type: NetworkMetricType) -> Optional[float]:
        """Get a specific centrality measure."""
        return self.centrality_measures.get(metric_type)

@dataclass
class DigraphEdge:
    """Edge in the SFM digraph representing dependencies and relationships."""

    source_id: uuid.UUID
    target_id: uuid.UUID
    relationship_kind: RelationshipKind
    dependency_strength: DependencyStrength = DependencyStrength.MODERATE  # type: ignore[misc]
    weight: float = 1.0  # Edge weight for analysis

    # Flow properties
    flow_capacity: Optional[float] = None  # Maximum flow through this edge
    current_flow: Optional[float] = None  # Current flow through this edge
    flow_efficiency: Optional[float] = None  # Efficiency of flow (0-1)

    # Temporal properties
    activation_time: Optional[datetime] = None  # When this relationship becomes active
    duration: Optional[float] = None  # How long this relationship lasts

    # SFM-specific attributes
    ceremonial_component: Optional[float] = None  # Ceremonial aspect of relationship
    instrumental_component: Optional[float] = None  # Instrumental aspect of relationship
    institutional_constraint: bool = False  # Whether this edge represents a constraint

    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def calculate_flow_utilization(self) -> Optional[float]:
        """Calculate flow utilization as percentage of capacity."""
        if self.flow_capacity is None or self.current_flow is None:
            return None
        if self.flow_capacity == 0:
            return 0.0
        return min(self.current_flow / self.flow_capacity, 1.0)

    def is_bottleneck(self, threshold: float = 0.9) -> bool:
        """Check if this edge is a bottleneck (high utilization)."""
        utilization = self.calculate_flow_utilization()
        return utilization is not None and utilization >= threshold

    def calculate_ceremonial_instrumental_balance(self) -> Optional[float]:
        """Calculate balance between ceremonial and instrumental components."""
        if self.ceremonial_component is None or self.instrumental_component is None:
            return None
        total = self.ceremonial_component + self.instrumental_component
        if total == 0:
            return 0.0
        return (self.instrumental_component - self.ceremonial_component) / total

@dataclass
class PathAnalysis:
    """Results of path analysis in the digraph."""

    source_id: uuid.UUID
    target_id: uuid.UUID
    path_nodes: List[uuid.UUID] = field(default_factory=list)
    path_length: int = 0
    total_weight: float = 0.0
    bottlenecks: List[uuid.UUID] = field(default_factory=list)  # Edge IDs that are bottlenecks

    # Path properties
    is_shortest_path: bool = False
    is_critical_path: bool = False  # Path that determines system timing
    path_efficiency: Optional[float] = None  # Overall efficiency of this path

    def calculate_average_weight(self) -> float:
        """Calculate average weight per edge in the path."""
        if self.path_length == 0:
            return 0.0
        return self.total_weight / self.path_length

@dataclass
class CentralityAnalysis:
    """Results of centrality analysis for the digraph."""

    node_centralities: Dict[uuid.UUID, Dict[NetworkMetricType, float]] = field(default_factory=dict)
    most_central_nodes: Dict[NetworkMetricType, uuid.UUID] = field(default_factory=dict)
    centrality_distribution: Dict[NetworkMetricType, Dict[str, float]] = field(default_factory=dict)

    def get_top_nodes(
        self,
        metric_type: NetworkMetricType,
        top_n: int = 5) -> List[Tuple[uuid.UUID, float]]:
        """Get top N nodes for a specific centrality metric."""
        node_scores = []
        for node_id, centralities in self.node_centralities.items():
            score = centralities.get(metric_type, 0.0)
            node_scores.append((node_id, score))

        node_scores.sort(key=lambda x: x[1], reverse=True)
        return node_scores[:top_n]

    def calculate_centralization(self, metric_type: NetworkMetricType) -> float:
        """Calculate overall centralization of the network for a metric."""
        if not self.node_centralities:
            return 0.0

        centralities = [
            centralities.get(metric_type, 0.0)
            for centralities in self.node_centralities.values()
        ]

        if not centralities:
            return 0.0

        max_centrality = max(centralities)
        sum_deviations = sum(max_centrality - c for c in centralities)

        n = len(centralities)
        max_possible_sum = (n - 1) * max_centrality

        if max_possible_sum == 0:
            return 0.0

        return sum_deviations / max_possible_sum

@dataclass
class SFMDigraph(Node):
    """Complete digraph structure for Social Fabric Matrix analysis."""

    nodes: Dict[uuid.UUID, DigraphNode] = field(default_factory=dict)
    edges: Dict[uuid.UUID, DigraphEdge] = field(default_factory=dict)

    # Analysis results
    centrality_analysis: Optional[CentralityAnalysis] = None
    path_analyses: List[PathAnalysis] = field(default_factory=list)

    # Matrix properties
    matrix_dimensions: Optional[Tuple[int, int]] = None  # (institutions, criteria)
    delivery_matrix: Optional[List[List[float]]] = None  # Matrix of delivery values

    # System properties
    system_coherence: Optional[float] = None  # Overall system coherence (0-1)
    institutional_density: Optional[float] = None  # Density of institutional connections
    feedback_loops: List[List[uuid.UUID]] = field(default_factory=list)  # Detected loops

    def add_node(self, node: DigraphNode) -> None:
        """Add a node to the digraph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: DigraphEdge) -> None:
        """Add an edge to the digraph."""
        self.edges[edge.id] = edge

        # Update node degrees
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].out_degree += 1
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].in_degree += 1

    def remove_node(self, node_id: uuid.UUID) -> None:
        """Remove a node and all connected edges."""
        if node_id not in self.nodes:
            return

        # Remove all edges connected to this node
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # Remove the node
        del self.nodes[node_id]

    def remove_edge(self, edge_id: uuid.UUID) -> None:
        """Remove an edge from the digraph."""
        if edge_id not in self.edges:
            return

        edge = self.edges[edge_id]

        # Update node degrees
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].out_degree -= 1
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].in_degree -= 1

        del self.edges[edge_id]

    def get_neighbors(self, node_id: uuid.UUID, direction: str = "out") -> List[uuid.UUID]:
        """Get neighboring nodes (out: targets, in: sources, both: all)."""
        neighbors = []

        for edge in self.edges.values():
            if direction in ["out", "both"] and edge.source_id == node_id:
                neighbors.append(edge.target_id)
            if direction in ["in", "both"] and edge.target_id == node_id:
                neighbors.append(edge.source_id)

        return list(set(neighbors))  # Remove duplicates

    def find_shortest_path(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID) -> Optional[PathAnalysis]:
        """Find shortest path between two nodes using Dijkstra's algorithm."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        # Dijkstra's algorithm implementation
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[source_id] = 0.0
        previous = {}
        unvisited = set(self.nodes.keys())

        while unvisited:
            current = min(unvisited, key=lambda x: distances[x])
            if distances[current] == float('inf'):
                break

            unvisited.remove(current)

            if current == target_id:
                break

            # Check all neighbors
            for edge in self.edges.values():
                if edge.source_id == current and edge.target_id in unvisited:
                    alt_distance = distances[current] + edge.weight
                    if alt_distance < distances[edge.target_id]:
                        distances[edge.target_id] = alt_distance
                        previous[edge.target_id] = current

        # Reconstruct path
        if target_id not in previous and source_id != target_id:
            return None  # No path found

        path = []
        current = target_id
        while current != source_id:
            path.append(current)
            if current not in previous:
                break
            current = previous[current]
        path.append(source_id)
        path.reverse()

        return PathAnalysis(
            source_id=source_id,
            target_id=target_id,
            path_nodes=path,
            path_length=len(path) - 1,
            total_weight=distances[target_id],
            is_shortest_path=True
        )

    def detect_cycles(self) -> List[List[uuid.UUID]]:
        """Detect cycles in the digraph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs_cycle_detect(node_id: uuid.UUID, path: List[uuid.UUID]) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            # Check all neighbors
            for edge in self.edges.values():
                if edge.source_id == node_id:
                    neighbor = edge.target_id

                    if neighbor in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        cycles.append(cycle)
                        return True
                    elif neighbor not in visited:
                        if dfs_cycle_detect(neighbor, path.copy()):
                            return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                dfs_cycle_detect(node_id, [])

        self.feedback_loops = cycles
        return cycles

    def calculate_system_metrics(self) -> Dict[str, float]:
        """Calculate overall system metrics."""
        if not self.nodes or not self.edges:
            return {}

        num_nodes = len(self.nodes)
        num_edges = len(self.edges)

        # Calculate density
        max_edges = num_nodes * (num_nodes - 1)  # Directed graph
        density = num_edges / max_edges if max_edges > 0 else 0.0

        # Calculate average degree
        total_degree = sum(node.calculate_total_degree() for node in self.nodes.values())
        avg_degree = total_degree / num_nodes if num_nodes > 0 else 0.0

        # Calculate clustering coefficient (simplified)
        clustering_coeff = self._calculate_clustering_coefficient()

        metrics = {
            'density': density,
            'average_degree': avg_degree,
            'clustering_coefficient': clustering_coeff,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_cycles': len(self.feedback_loops)
        }

        self.institutional_density = density
        return metrics

    def _calculate_clustering_coefficient(self) -> float:
        """Calculate the clustering coefficient of the graph."""
        if len(self.nodes) < 3:
            return 0.0

        total_clustering = 0.0
        valid_nodes = 0

        for node_id in self.nodes:
            neighbors = self.get_neighbors(node_id, "both")
            if len(neighbors) < 2:
                continue

            # Count edges between neighbors
            neighbor_edges = 0
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    # Check if there's an edge between neighbor1 and neighbor2
                    for edge in self.edges.values():
                        if ((edge.source_id == neighbor1 and edge.target_id == neighbor2) or
                            (edge.source_id == neighbor2 and edge.target_id == neighbor1)):
                            neighbor_edges += 1
                            break

            # Calculate clustering coefficient for this node
            k = len(neighbors)
            possible_edges = k * (k - 1) / 2  # Undirected edges between neighbors
            if possible_edges > 0:
                node_clustering = neighbor_edges / possible_edges
                total_clustering += node_clustering
                valid_nodes += 1

        return total_clustering / valid_nodes if valid_nodes > 0 else 0.0

@dataclass
class NetworkAnalyzer:
    """Tools for analyzing SFM digraph networks."""

    digraph: SFMDigraph

    def calculate_centrality_measures(self) -> CentralityAnalysis:
        """Calculate various centrality measures for all nodes."""
        analysis = CentralityAnalysis()

        for node_id in self.digraph.nodes:
            node_centralities = {}

            # Degree centrality
            degree_centrality = self.digraph.nodes[node_id].calculate_degree_centrality(
                len(self.digraph.nodes)
            )
            node_centralities[NetworkMetricType.DEGREE_CENTRALITY] = degree_centrality

            # Betweenness centrality (simplified)
            betweenness = self._calculate_betweenness_centrality(node_id)
            node_centralities[NetworkMetricType.BETWEENNESS_CENTRALITY] = betweenness

            # Closeness centrality (simplified)
            closeness = self._calculate_closeness_centrality(node_id)
            node_centralities[NetworkMetricType.CLOSENESS_CENTRALITY] = closeness

            analysis.node_centralities[node_id] = node_centralities

            # Update node centrality measures
            for metric_type, value in node_centralities.items():
                self.digraph.nodes[node_id].update_centrality_measure(metric_type, value)

        # Find most central nodes for each metric
        for metric_type in NetworkMetricType:
            max_centrality = 0.0
            most_central = None

            for node_id, centralities in analysis.node_centralities.items():
                centrality = centralities.get(metric_type, 0.0)
                if centrality > max_centrality:
                    max_centrality = centrality
                    most_central = node_id

            if most_central:
                analysis.most_central_nodes[metric_type] = most_central

        self.digraph.centrality_analysis = analysis
        return analysis

    def _calculate_betweenness_centrality(self, node_id: uuid.UUID) -> float:
        """Calculate betweenness centrality for a node (simplified version)."""
        if len(self.digraph.nodes) < 3:
            return 0.0

        betweenness = 0.0
        total_pairs = 0

        # For each pair of other nodes
        other_nodes = [nid for nid in self.digraph.nodes if nid != node_id]

        for i, source in enumerate(other_nodes):
            for target in other_nodes[i+1:]:
                # Find shortest path from source to target
                path = self.digraph.find_shortest_path(source, target)
                if path and node_id in path.path_nodes[1:-1]:  # Node is on the path (not source/target)
                    betweenness += 1.0
                total_pairs += 1

        return betweenness / total_pairs if total_pairs > 0 else 0.0

    def _calculate_closeness_centrality(self, node_id: uuid.UUID) -> float:
        """Calculate closeness centrality for a node."""
        if len(self.digraph.nodes) < 2:
            return 0.0

        total_distance = 0.0
        reachable_nodes = 0

        # Calculate distance to all other nodes
        for target_id in self.digraph.nodes:
            if target_id != node_id:
                path = self.digraph.find_shortest_path(node_id, target_id)
                if path:
                    total_distance += path.total_weight
                    reachable_nodes += 1

        if reachable_nodes == 0 or total_distance == 0:
            return 0.0

        # Closeness is inverse of average distance
        avg_distance = total_distance / reachable_nodes
        return 1.0 / avg_distance

    def analyze_institutional_dependencies(self) -> Dict[str, Any]:
        """Analyze institutional dependencies in the digraph."""
        dependencies = {}

        for node_id, node in self.digraph.nodes.items():
            if node.node_type == DigraphNodeType.INTERMEDIATE:  # type: ignore[misc]
                # Find what this institution depends on
                incoming_edges = [
                    edge for edge in self.digraph.edges.values()
                    if edge.target_id == node_id
                ]

                # Find what depends on this institution
                outgoing_edges = [
                    edge for edge in self.digraph.edges.values()
                    if edge.source_id == node_id
                ]

                dependencies[str(node_id)] = {
                    'depends_on': [str(edge.source_id) for edge in incoming_edges],
                    'supports': [str(edge.target_id) for edge in outgoing_edges],
                    'dependency_strength': sum(
                        1 if edge.dependency_strength == DependencyStrength.STRONG else
                        0.5 if edge.dependency_strength == DependencyStrength.MODERATE else 0.25  # type: ignore[misc]
                        for edge in incoming_edges
                    ),
                    'support_strength': sum(
                        1 if edge.dependency_strength == DependencyStrength.STRONG else
                        0.5 if edge.dependency_strength == DependencyStrength.MODERATE else 0.25  # type: ignore[misc]
                        for edge in outgoing_edges
                    )
                }

        return dependencies

    def identify_critical_paths(self) -> List[PathAnalysis]:
        """Identify critical paths in the system."""
        critical_paths = []

        # Find paths between key institutional nodes
        institutional_nodes = [
            node_id for node_id, node in self.digraph.nodes.items()
            if node.node_type == DigraphNodeType.INTERMEDIATE  # type: ignore[misc]
        ]

        for source in institutional_nodes:
            for target in institutional_nodes:
                if source != target:
                    path = self.digraph.find_shortest_path(source, target)
                    if path:
                        # Determine if this is a critical path
                        # (simplified: paths with high total weight or many bottlenecks)
                        if path.total_weight > 5.0 or len(path.bottlenecks) > 0:
                            path.is_critical_path = True
                            critical_paths.append(path)

        return critical_paths
