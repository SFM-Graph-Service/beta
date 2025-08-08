# Code Smell Analysis - Action Items

## Immediate Fixes Required (Critical Priority)

### 1. Critical Errors
- [ ] **Fix undefined variable 'Any'** in `models/stakeholder_power.py:280`
  - Action: Add `from typing import Any` import
  - Risk: Blocking - prevents code execution

- [ ] **Fix assignment from None** in `models/circular_causation.py:800`  
  - Action: Review logic and add proper error handling
  - Risk: Runtime errors, unpredictable behavior

### 2. Attribute Definition Issues
- [ ] **Move attributes to __init__** in `models/social_indicators.py`
  - Lines: 887, 929, 949, 1137
  - Attributes: `last_updated`, `overall_completeness`, `average_quality_score`, `matrix_integration_scores`
  - Risk: Unpredictable object state

## High Priority Refactoring Tasks

### 3. God Classes (Split into separate issues)
- [ ] **Split specialized_nodes.py** (3,019 lines)
  - Action: Extract node types into separate modules
  - Issue: Create dedicated refactoring task

- [ ] **Refactor social_indicators.py** (1,528 lines)  
  - Action: Separate StatisticalAnalyzer, IndicatorDatabase, IndicatorDashboard
  - Issue: Create dedicated refactoring task

- [ ] **Consider splitting sfm_enums.py** (3,991 lines)
  - Action: Group related enums into thematic modules
  - Issue: Create dedicated refactoring task

### 4. Long Methods
- [ ] **Refactor complex analysis methods**
  - `StatisticalAnalysisPipeline.calculate_matrix_integration_completeness`
  - `LegislativeProcessModeling.analyze_voting_coalition_dynamics`
  - Issue: Create dedicated refactoring tasks

### 5. Complex Conditional Logic
- [ ] **Simplify nested conditionals** in `social_indicators.py:1432`
  - Method: `IndicatorDashboard.check_alerts`
  - Action: Extract condition checking methods, use early returns

- [ ] **Reduce return statements** in `institutional_adjustment.py:355`
  - Current: 9 return statements
  - Target: ≤6 return statements

## Medium Priority Cleanup

### 6. Dead Code Removal
- [ ] **Clean up unused imports** (47+ instances across multiple files)
  - Files: system_boundary.py, sfm_system_integration.py, institutional_adjustment.py, multi_scale_integration.py
  - Tools: Use autoflake for automated cleanup

- [ ] **Remove unused code** in `advanced_data_validation.py`
  - Classes: `ReferenceValidationRule`, `StatisticalOutlierRule`  
  - Methods: `add_relationship_rule`, `check_reference_integrity`

### 7. Dictionary Iteration Optimization  
- [ ] **Fix dictionary iteration** in `political_action.py:1492`
  - Current: `for key in dict.keys()`
  - Better: `for key in dict`

## Separate GitHub Issues to Create

Based on this analysis, create the following separate issues for major refactoring tasks:

1. **Refactor specialized_nodes.py - Split god class into focused modules**
2. **Refactor social_indicators.py - Separate statistical analysis concerns**  
3. **Simplify complex analysis methods - Extract and simplify long methods**
4. **Clean up unused imports and dead code across codebase**
5. **Fix attribute definition anti-patterns in social_indicators.py**

## Metrics Summary

- **Total Issues Identified:** 74+
- **Critical Issues:** 3 (blocking errors)
- **High Priority Issues:** 8 (architectural/complexity)
- **Medium Priority Issues:** 63+ (cleanup/optimization)

## Next Steps

1. Address critical issues immediately (can be done in this PR)
2. Create separate GitHub issues for major refactoring tasks
3. Implement automated cleanup for unused imports
4. Establish coding standards to prevent future code smells