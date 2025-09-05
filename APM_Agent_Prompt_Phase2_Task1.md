# Implementation Agent Prompt: Phase 2, Task 2.1

## Role Assignment
You are the **API Migration Implementation Agent** for the HA Realtime Voice Assistant project. Your expertise includes WebSocket protocols, OpenAI API integration, and backward compatibility design. You'll lead the migration from `gpt-4o-realtime-preview` to the new `gpt-realtime` model.

## Project Context - Phase 1 Complete
Phase 1 has successfully resolved all audio quality issues:
- Wake word detection: 85% → >95% ✅
- Processing latency: 30ms → <5ms ✅
- CPU usage: 45% → 15% ✅
- Audio pipeline: Fully optimized ✅

Now entering Phase 2: OpenAI Realtime API Migration

## Your Mission: Task 2.1 - Model Migration Implementation

### Objective
Migrate the assistant from `gpt-4o-realtime-preview` to `gpt-realtime`, capturing all performance improvements while maintaining backward compatibility for users who wish to remain on the preview model.

### New Model Advantages to Capture
Based on OpenAI's August 2025 release:

| Metric | Old Model | New Model | Improvement |
|--------|-----------|-----------|-------------|
| Big Bench Audio | 65.6% | 82.8% | +26% |
| Instruction Following | 20.6% | 30.5% | +48% |
| Function Calling | 49.7% | 66.5% | +34% |
| Cost | $40/$80 | $32/$64 | -20% |
| Voices | 8 | 10 (Cedar, Marin) | +2 new |

### Specifications

#### Core Requirements

1. **Model Configuration Update:**
   ```python
   # Current (src/config.py:18)
   model: str = "gpt-4o-realtime-preview"
   
   # New configuration with compatibility
   model: str = "gpt-realtime"  # Default to new model
   legacy_model: str = "gpt-4o-realtime-preview"  # Fallback option
   model_selection: str = "auto"  # auto, new, legacy
   ```

2. **New Voice Options:**
   ```python
   # Existing voices (8)
   existing_voices = ["alloy", "ash", "ballad", "coral", 
                      "echo", "sage", "shimmer", "verse"]
   
   # New exclusive voices (2)
   new_voices = ["cedar", "marin"]
   
   # Voice configuration
   voice_config = {
       "primary": "cedar",  # New voice with best quality
       "fallback": "alloy",  # If new voice unavailable
       "auto_select": True   # Choose based on model
   }
   ```

3. **WebSocket Protocol Updates:**
   - Update session initialization parameters
   - Handle new event types if any
   - Implement new error codes
   - Support asynchronous function calling improvements

4. **Backward Compatibility:**
   - Configuration flag for model selection
   - Automatic fallback on connection failure
   - Voice compatibility mapping
   - Feature degradation handling

5. **Performance Validation:**
   - Latency measurements
   - Function calling accuracy tests
   - Instruction following validation
   - Cost tracking implementation

### Implementation Guidelines

#### File Structure
Primary updates needed in:
```
src/
├── config.py                 # Model configuration
├── openai_client/
│   └── realtime.py          # WebSocket client updates
└── web/
    └── templates/
        └── config/
            └── yaml.html    # UI model selection
```

New files to create:
```
src/openai_client/
├── model_compatibility.py   # Compatibility layer
├── voice_manager.py         # Voice selection logic
└── performance_metrics.py  # Track improvements
```

#### Code Architecture
```python
class ModelMigration:
    """Handles migration to gpt-realtime with fallback support"""
    
    MODELS = {
        'gpt-realtime': {
            'endpoint': 'wss://api.openai.com/v1/realtime',
            'voices': ['alloy', 'ash', 'ballad', 'coral', 'echo', 
                      'sage', 'shimmer', 'verse', 'cedar', 'marin'],
            'features': ['async_functions', 'native_mcp', 'image_input'],
            'pricing': {'input': 32, 'output': 64}  # per 1M tokens
        },
        'gpt-4o-realtime-preview': {
            'endpoint': 'wss://api.openai.com/v1/realtime',
            'voices': ['alloy', 'ash', 'ballad', 'coral', 'echo',
                      'sage', 'shimmer', 'verse'],
            'features': [],
            'pricing': {'input': 40, 'output': 80}
        }
    }
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_model = None
        self.metrics = PerformanceMetrics()
        
    async def connect(self, preferred_model: str = None):
        # Attempt connection with new model
        # Fallback to legacy if needed
        
    def select_voice(self, preferred: str) -> str:
        # Smart voice selection based on model
        
    def track_usage(self, tokens: dict):
        # Cost tracking with model-specific pricing
```

#### Migration Strategy

1. **Phase A - Preparation:**
   - Add new model configuration
   - Update voice list
   - Create compatibility layer

2. **Phase B - Implementation:**
   - Update WebSocket client
   - Add model selection logic
   - Implement performance tracking

3. **Phase C - Validation:**
   - Test both models side-by-side
   - Validate voice quality
   - Confirm cost reduction

4. **Phase D - Deployment:**
   - Default to new model
   - Monitor for issues
   - Collect metrics

### Acceptance Criteria
- [ ] Successfully connects to `gpt-realtime` model
- [ ] All 10 voices accessible (including Cedar, Marin)
- [ ] Backward compatibility with preview model
- [ ] Configuration UI shows model selection
- [ ] Cost reduction verified (20% lower)
- [ ] Performance metrics collected and logged
- [ ] Automatic fallback works on failure
- [ ] No breaking changes for existing users
- [ ] Documentation updated with migration guide

### Resources Available

#### OpenAI Documentation
From `docs/references/20250829_OpenAI_Update/`:
- Model announcement and capabilities
- New voice samples and characteristics
- Pricing structure changes
- Performance benchmarks

#### Current Implementation
```python
# src/openai_client/realtime.py
# Current WebSocket connection setup
async def connect(self):
    self.ws_url = "wss://api.openai.com/v1/realtime"
    # Headers, authentication, etc.
```

#### Configuration System
```python
# Leverage existing config structure
from src.config import load_config, save_config
config = load_config()
config.openai.model = "gpt-realtime"
save_config(config)
```

### Success Metrics
- Connection success rate: >99%
- Voice quality improvement: Subjective but noticeable
- Function calling accuracy: >66% (from 49.7%)
- Cost per interaction: <$0.008 (from $0.010)
- User satisfaction: No degradation

### Deliverables
1. **Updated Core Files:**
   - `src/config.py` - Model configuration
   - `src/openai_client/realtime.py` - Client updates

2. **New Modules:**
   - `src/openai_client/model_compatibility.py`
   - `src/openai_client/voice_manager.py`
   - `src/openai_client/performance_metrics.py`

3. **Configuration:**
   - Updated `config.yaml.example` with new options
   - Migration settings template

4. **Documentation:**
   - `docs/OPENAI_MIGRATION.md` - Migration guide
   - Updated `README.md` with model information

5. **Testing:**
   - `tests/test_model_migration.py`
   - Performance comparison results

### Timeline
**Duration:** 2 days
- Day 1: Core implementation and compatibility layer
- Day 2: Testing, validation, and documentation

### Critical Considerations

#### Breaking Changes Risk
- New voices won't work with old model
- Some features exclusive to new model
- Different error responses possible

#### Rollback Strategy
```yaml
# Quick rollback configuration
openai:
  model_selection: "legacy"  # Forces old model
  disable_new_features: true
```

#### Cost Tracking
```python
# Implement usage monitoring
def calculate_cost(tokens: dict, model: str) -> float:
    rates = MODELS[model]['pricing']
    input_cost = (tokens['input'] / 1_000_000) * rates['input']
    output_cost = (tokens['output'] / 1_000_000) * rates['output']
    return input_cost + output_cost
```

### Getting Started
1. Review OpenAI announcement in docs/references/
2. Update configuration schema
3. Modify WebSocket client for model selection
4. Implement voice compatibility layer
5. Add performance tracking
6. Test with both models
7. Create migration documentation

### Questions to Address
- Should we auto-migrate existing users or require opt-in?
- How to handle users who prefer old voices?
- Should we expose cost savings in the UI?
- What metrics are most important to track?

## Your Response Should Include
1. Acknowledgment of Phase 1 completion
2. Understanding of new model capabilities
3. Migration approach and risk mitigation
4. Backward compatibility strategy
5. Testing plan for validation

Remember: This migration represents a significant improvement in capabilities and cost. The implementation must be seamless for users while capturing all benefits of the new model. Focus on zero-downtime migration with robust fallback options.

---

*Reference the APM_Memory_Bank.md and APM_Phase1_Completion_Report.md for project context.*