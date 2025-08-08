# SFM Graph Service - Beta

Social Fabric Matrix (SFM) framework implementation - a comprehensive Python modeling library for analyzing complex socio-economic systems through institutional, political, and social indicators.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup and Dependencies
- Install required linting tools:
  - `pip3 install pylint pyright` -- takes 10 seconds, required for code quality validation

### Code Quality and Validation
- Run pylint for functional error checking:
  - `pylint models/ --rcfile=.pylintrc` -- takes 45 seconds. NEVER CANCEL. Set timeout to 90+ minutes.
  - Focus on fixing E (Error) and critical W (Warning) issues only
- Run pyright for type checking:
  - `pyright models/` -- takes 25 seconds. NEVER CANCEL. Set timeout to 60+ minutes.
- ALWAYS run both linting commands before committing changes or CI will fail

### Framework Validation and Testing
- ALWAYS validate changes by testing basic functionality:
  ```python
  from models import Actor, Institution, Policy, Resource, Process, Flow
  from models.specialized_nodes import SystemProperty, PolicyInstrument
  import uuid
  
  # Test basic model creation
  actor = Actor(label='Test Actor', description='Validation test')
  institution = Institution(label='Test Institution', description='Validation test')
  
  # Test core functionality
  power_index = actor.calculate_power_index()
  embeddedness = actor.assess_institutional_embeddedness()
  
  print(f'Validation successful: {actor.label}, {institution.label}')
  ```
- ALWAYS test complex models with required parameters:
  ```python
  from models.specialized_nodes import MatrixCell
  import uuid
  
  matrix_cell = MatrixCell(
      label='Test Matrix Cell',
      description='Validation test',
      institution_id=uuid.uuid4(),
      criteria_id=uuid.uuid4()
  )
  ```

## Critical Information

### Repository Structure
- **NOT a runnable application** - this is a modeling library/framework
- Primary code in `models/` directory with ~50 Python files defining SFM framework classes
- No web server, CLI, or main application entry point
- No automated test suite - validation requires manual testing of imports and class instantiation

### Core Model Classes
Key classes available for import from `models` package:
- `Actor` - Individuals, firms, agencies, communities
- `Institution` - Legal frameworks, organizations, rules
- `Policy` - Policies and interventions
- `Resource` - Economic resources, funding, materials
- `Process` - Operational processes, workflows
- `Flow` - Information, resource, or process flows
- `SystemProperty` - System-level properties and characteristics
- `PolicyInstrument` - Policy tools and mechanisms
- `MatrixCell` - SFM matrix relationship cells (requires institution_id and criteria_id)

### Construction Requirements
- All models inherit from `Node` base class
- Required constructor parameters: `label` (string), optional `description`
- Models automatically get UUID `id`, timestamps, and metadata fields
- Some specialized classes require additional parameters (check class definition)

## Configuration Files

### Linting Configuration
- `.pylintrc` - Focuses on functional errors, disables many style checks by design
- `pyrightconfig.json` - Strict type checking with Python 3.12, reports type issues as information/warnings

### Key Directories and Files
```
/home/runner/work/beta/beta/
├── models/           # Core SFM framework classes (~50 files)
├── issues/           # Documentation and issue tracking
├── .pylintrc         # Pylint configuration
├── pyrightconfig.json # Type checking configuration
└── README.md         # Project documentation
```

## Validation Scenarios

### MANUAL VALIDATION REQUIREMENT
After making changes to model classes, ALWAYS test:

1. **Basic Import Test**:
   ```bash
   python3 -c "import models; print('Models package imported successfully')"
   ```

2. **Core Class Import Test**:
   ```bash
   python3 -c "from models import Actor, Institution, Node; print('Core classes imported successfully')"
   ```

3. **Functional Validation Test**:
   ```bash
   python3 -c "
   from models import Actor, Institution
   actor = Actor(label='Test', description='Validation')
   inst = Institution(label='Test', description='Validation')
   print(f'Created: {actor.label}, {inst.label}')
   print(f'Power index: {actor.calculate_power_index()}')
   print('Functional validation passed')
   "
   ```

4. **Complex Model Test**:
   ```bash
   python3 -c "
   from models.specialized_nodes import MatrixCell
   import uuid
   cell = MatrixCell(
       label='Test Cell',
       description='Validation',
       institution_id=uuid.uuid4(),
       criteria_id=uuid.uuid4()
   )
   print(f'Complex model created: {cell.label}')
   "
   ```

### Timing Expectations
- **NEVER CANCEL**: Linting operations may appear slow but complete normally
- pip install pylint pyright: ~10 seconds
- pylint models/: ~45 seconds (can vary, set 90+ minute timeout)
- pyright models/: ~25 seconds (can vary, set 60+ minute timeout)
- Python import/functionality tests: <1 second each

## Common Tasks

### Adding New Model Classes
- Follow existing patterns in `models/` directory
- Inherit from `Node` or appropriate base class
- Add dataclass decorator and proper type annotations
- Add to `models/__init__.py` imports if public API
- ALWAYS test imports and basic instantiation
- Run pylint and pyright to check for issues

### Modifying Existing Models
- Make minimal changes to preserve existing functionality
- Check for impact on related classes via import analysis
- Test affected functionality with validation scenarios
- Address any new linting errors introduced

### Code Quality Standards
- Fix all pylint E (Error) messages - these are functional issues
- Address critical W (Warning) messages that affect functionality
- Type annotations required for public APIs (pyright will report)
- Use `# pylint: disable=rule-name` sparingly with rationale comments

## Environment Details

### Python Environment
- Python 3.12.3 available at `/usr/bin/python3`
- Working directory: `/home/runner/work/beta/beta`
- No virtual environment required for basic usage
- pip installs to user site-packages by default

### Dependencies
- No runtime dependencies for basic framework usage
- Development dependencies: pylint, pyright (install as needed)
- No requirements.txt, setup.py, or pyproject.toml files

### Repository Type
This is a **modeling framework library**, not a standalone application:
- Import classes and create instances for analysis
- No server to start, no CLI commands to run
- No automated test suite to execute
- Validation through manual import and instantiation testing