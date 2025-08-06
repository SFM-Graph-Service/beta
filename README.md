# SFM Graph Service - Beta

Social Fabric Matrix (SFM) framework implementation with comprehensive data models and analytical tools.

## Overview

This repository contains the beta version of the SFM Graph Service, providing a comprehensive framework for modeling and analyzing complex socio-economic systems through institutional, political, and social indicators.

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
- `missing-docstring`: Focus on functional issues first
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

### CI/CD Integration

Consider adding GitHub Actions workflow for automated linting:

```yaml
name: Code Quality
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install pylint pyright
      - run: pylint models/
      - run: pyright models/
```

## Development

### Prerequisites
- Python 3.12+
- pylint (`pip install pylint`)
- pyright (`pip install pyright`)

### Structure
- `models/`: Core SFM framework data models and analysis tools
- `.pylintrc`: Pylint configuration
- `pyrightconfig.json`: Type checking configuration

## Contributing

1. Follow the established code quality standards
2. Run linters before submitting changes
3. Document any intentional lint suppressions
4. Ensure type annotations for new public APIs