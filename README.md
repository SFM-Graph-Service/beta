# SFM Graph Service - Beta

Social Fabric Matrix (SFM) framework implementation with comprehensive data models and analytical tools.

## Overview

This repository contains the beta version of the SFM Graph Service, providing a comprehensive framework for modeling and analyzing complex socio-economic systems through institutional, political, and social indicators.

## Code Architecture

The codebase has been organized into focused, maintainable modules:

### Core Modules
- **`models/base_nodes.py`** - Base node definitions and core abstractions
- **`models/sfm_enums.py`** - Enumerations and type definitions for SFM framework

### Specialized Analysis Modules
- **`models/matrix_components.py`** - Core SFM matrix cells, criteria, and matrix configuration
- **`models/system_analysis.py`** - System-level properties, analysis, and institutional holarchy
- **`models/policy_framework.py`** - Policy instruments, value judgments, and problem-solving sequences
- **`models/institutional_analysis.py`** - Institutional structures and path dependency analysis
- **`models/economic_analysis.py`** - Transaction costs, coordination mechanisms, commons governance
- **`models/cultural_analysis.py`** - Ceremonial vs instrumental analysis, value systems, beliefs
- **`models/social_assessment.py`** - Social value assessment, fabric indicators, social costs
- **`models/technology_integration.py`** - Technology complexes and ecological systems
- **`models/network_analysis.py`** - Cross-impact analysis, delivery relationships, network analysis
- **`models/complex_analysis.py`** - Digraph analysis, circular causation, conflict detection
- **`models/methodological_framework.py`** - Inquiry frameworks, normative analysis, policy integration
- **`models/specialized_components.py`** - Additional specialized indicators and pathway analysis

### Unified Interface
- **`models/specialized_nodes.py`** - Provides unified access to all specialized components for backward compatibility

## Benefits of New Architecture

- ✅ **Improved maintainability**: Each module is focused and manageable (100-400 lines vs original 3020-line god class)
- ✅ **Better separation of concerns**: Each module has a single, well-defined responsibility
- ✅ **Easier testing**: Focused modules can be tested in isolation
- ✅ **Reduced cognitive load**: Developers can focus on specific aspects without overwhelming complexity
- ✅ **Maintained backward compatibility**: Existing imports continue to work seamlessly
- ✅ **No circular dependencies**: Clean module structure with clear dependencies

## Code Quality and Linting

This project uses strict code quality standards with automated linting and type checking.

### Linting Tools

- **Pylint**: Code analysis and style checking
- **Pyright**: Static type checking with strict mode enabled

### Running Linters

#### Pylint
```bash
# Run pylint on all models
pylint models/

# Run with specific configuration
pylint --rcfile=.pylintrc models/

# Check a specific file
pylint models/social_indicators.py
```

#### Pyright (Type Checking)
```bash
# Run pyright type checking
pyright models/

# For Pylance in VSCode, ensure pyrightconfig.json is configured
```

### Configuration Files

- **`.pylintrc`**: Pylint configuration focusing on functional errors over style preferences
- **`pyrightconfig.json`**: Pyright configuration with strict type checking enabled

### Linting Standards

#### Enabled Checks (High Priority)
- Syntax errors
- Undefined variables
- Import errors
- Missing members
- Unused imports and variables
- Security issues (dangerous defaults, bare except)

#### Disabled Checks (Design Choices)
- `too-many-instance-attributes`: SFM framework legitimately requires many attributes
- `too-many-lines`: Large files acceptable for comprehensive frameworks
- `line-too-long`: Style preference, not functional issue
- `too-few-public-methods`: Acceptable for data classes and enums

#### Ignored Issues (Documented)

Some linting errors are suppressed with `# pylint: disable=` comments when they represent:
1. **Unimplemented placeholder methods**: Methods that call non-existent private methods (intended for future implementation)
2. **Misplaced code sections**: Code that appears to belong to different classes but is documented as such
3. **Design patterns**: Patterns that are intentional but trigger linter warnings

### Adding New Code

When adding new code:
1. Run pylint to check for functional errors
2. Fix all `E` (Error) and critical `W` (Warning) issues
3. Use `# pylint: disable=` sparingly and with rationale comments
4. Ensure type annotations are present for public APIs
5. Add docstrings for public methods and classes


## Contributing

1. Follow the established code quality standards
2. Run linters before submitting changes
3. Document any intentional lint suppressions
4. Ensure type annotations for new public APIs