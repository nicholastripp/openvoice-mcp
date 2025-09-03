# APM Implementation Plan: HA Realtime Voice Assistant

## Project Metadata
- **Project Name:** Home Assistant Realtime Voice Assistant
- **Current Version:** v1.1.5
- **APM Framework Version:** 1.0
- **Plan Created:** 2025-09-03
- **Project Status:** Production (with critical audio quality concerns)
- **Estimated Completion:** 65% (core features complete, optimization needed)

## Executive Summary
This Implementation Plan guides the systematic improvement of the HA Realtime Voice Assistant using the Agentic Project Management (APM) framework. The project is a production-ready voice assistant but requires critical work on audio quality and API modernization before continuing feature development.

## Project Objectives
1. **Primary Goal:** Create a reliable, low-latency voice interface for Home Assistant
2. **Critical Issue:** Audio pipeline distortion affecting wake word detection and transcription
3. **Opportunity:** Leverage OpenAI's new `gpt-realtime` model for significant improvements
4. **Success Criteria:** <600ms latency, >95% wake word detection, >98% transcription accuracy

## Phase Structure Overview

### Phase 1: Audio Input Diagnostics & Validation
**Duration:** 2 weeks  
**Priority:** CRITICAL  
**Dependencies:** None  
**Goal:** Establish audio quality baseline and fix distortion issues

### Phase 2: OpenAI Realtime API Migration  
**Duration:** 2 weeks  
**Priority:** HIGH  
**Dependencies:** Phase 1 completion  
**Goal:** Upgrade to `gpt-realtime` model with new features

### Phase 3: System Stabilization
**Duration:** 1 week  
**Priority:** HIGH  
**Dependencies:** Phase 2 completion  
**Goal:** Fix connection and playback stability issues

### Phase 4: Feature Enhancement
**Duration:** 2 weeks  
**Priority:** MEDIUM  
**Dependencies:** Phase 3 completion  
**Goal:** Add user-requested features and improvements

### Phase 5: Performance Optimization
**Duration:** 1 week  
**Priority:** LOW  
**Dependencies:** Phase 4 completion  
**Goal:** Optimize for resource-constrained hardware

---

## Phase 1: Audio Input Diagnostics & Validation

### Objectives
- Identify and eliminate audio distortion in the processing pipeline
- Create comprehensive diagnostic tools for audio quality assessment
- Establish optimal configuration baselines for different hardware

### Key Tasks

#### Task 1.1: Audio Pipeline Analysis Tool Development
**Agent:** Audio DSP Agent  
**Duration:** 3 days  
**Dependencies:** None

**Specifications:**
- Build diagnostic tool analyzing signal at 7 critical stages
- Generate visual representations (waveforms, spectrograms)
- Output comprehensive metrics report
- Support real-time monitoring mode

**Acceptance Criteria:**
- [ ] Tool captures audio at each pipeline stage
- [ ] Metrics include RMS, peak, THD, SNR, clipping ratio
- [ ] Visual output shows frequency response
- [ ] CSV export for metrics comparison

#### Task 1.2: Resampling Quality Assessment
**Agent:** Audio DSP Agent  
**Duration:** 2 days  
**Dependencies:** Task 1.1

**Specifications:**
- Compare scipy.signal.resample with alternatives
- Test different chunk sizes (1200, 2400, 4800 samples)
- Measure frequency response 20Hz-12kHz
- Document aliasing artifacts

**Acceptance Criteria:**
- [ ] Quantitative comparison of resampling methods
- [ ] Optimal chunk size identified
- [ ] Frequency response plots generated
- [ ] Aliasing below -60dB

#### Task 1.3: Gain Stage Optimization
**Agent:** Testing Agent  
**Duration:** 3 days  
**Dependencies:** Task 1.1

**Specifications:**
- Test matrix: 3 mic types × 5 gain levels × 3 volume levels
- Map cumulative gain through pipeline
- Create automatic calibration routine
- Generate device-specific profiles

**Acceptance Criteria:**
- [ ] No clipping at normal speech (60-65 dB SPL)
- [ ] Wake word detection >95% accuracy
- [ ] Calibration completes in <30 seconds
- [ ] Profiles for USB, HAT, conference mics

#### Task 1.4: Audio Format Validation
**Agent:** Audio DSP Agent  
**Duration:** 2 days  
**Dependencies:** Task 1.1

**Specifications:**
- Test PCM16 conversion with different normalization
- Verify bit depth preservation
- Check for DC bias introduction
- Validate symmetric waveform conversion

**Acceptance Criteria:**
- [ ] Zero DC bias in output
- [ ] Symmetric clipping behavior
- [ ] No quantization noise above -90dB
- [ ] Proper headroom management

#### Task 1.5: Documentation and Reporting
**Agent:** Documentation Agent  
**Duration:** 2 days  
**Dependencies:** Tasks 1.2-1.4

**Deliverables:**
- Audio pipeline diagnostic tool (`tools/audio_pipeline_diagnostic.py`)
- Audio quality baseline report (`docs/audio_baseline_report.md`)
- Device configuration profiles (`config/audio_profiles/*.yaml`)
- Updated tuning guide (`docs/AUDIO_SETUP.md`)

---

## Phase 2: OpenAI Realtime API Migration

### Objectives
- Migrate to `gpt-realtime` model for improved performance
- Implement new API features (MCP, image input, async functions)
- Reduce operational costs by 20%

### Key Tasks

#### Task 2.1: Model Migration Implementation
**Agent:** API Migration Agent  
**Duration:** 2 days  
**Dependencies:** Phase 1 complete

**Specifications:**
- Update model identifier to `gpt-realtime`
- Implement new voice options (Cedar, Marin)
- Update WebSocket protocol handling
- Maintain backward compatibility flag

**Acceptance Criteria:**
- [ ] Successful connection to new model
- [ ] All 10 voices functional
- [ ] Existing features preserved
- [ ] Rollback mechanism tested

#### Task 2.2: Native MCP Integration
**Agent:** Integration Agent  
**Duration:** 3 days  
**Dependencies:** Task 2.1

**Specifications:**
- Replace custom MCP bridge with native support
- Configure remote MCP server connection
- Update tool discovery mechanism
- Implement approval workflows

**Acceptance Criteria:**
- [ ] Native MCP tools discovered
- [ ] Function calling success >90%
- [ ] Reduced latency vs bridge
- [ ] Tool approval UI functional

#### Task 2.3: Image Input Support
**Agent:** API Migration Agent  
**Duration:** 2 days  
**Dependencies:** Task 2.1

**Specifications:**
- Implement image capture/upload
- Add base64 encoding pipeline
- Create UI for image attachment
- Support screenshot integration

**Acceptance Criteria:**
- [ ] Images successfully processed
- [ ] Multiple format support (PNG, JPEG)
- [ ] Size optimization implemented
- [ ] Context awareness improved

#### Task 2.4: Performance Validation
**Agent:** Testing Agent  
**Duration:** 2 days  
**Dependencies:** Tasks 2.1-2.3

**Specifications:**
- Benchmark against current model
- Measure latency improvements
- Test accuracy metrics
- Calculate cost savings

**Acceptance Criteria:**
- [ ] Latency <600ms achieved
- [ ] Function calling >66% accuracy
- [ ] Instruction following >30% accuracy
- [ ] 20% cost reduction verified

#### Task 2.5: Migration Documentation
**Agent:** Documentation Agent  
**Duration:** 1 day  
**Dependencies:** Task 2.4

**Deliverables:**
- Migration guide (`docs/openai_migration_guide.md`)
- Performance comparison report
- Breaking changes documentation
- Configuration examples

---

## Phase 3: System Stabilization

### Objectives
- Fix SSE/MCP connection reliability issues
- Resolve audio playback interruptions
- Implement comprehensive error recovery

### Key Tasks

#### Task 3.1: Connection Management Improvements
**Agent:** Integration Agent  
**Duration:** 2 days  
**Dependencies:** Phase 2 complete

**Specifications:**
- Implement exponential backoff (1s, 2s, 4s, 8s, max 60s)
- Add connection health monitoring
- Create fallback to cached data
- Implement circuit breaker pattern

**Acceptance Criteria:**
- [ ] 99% uptime over 24 hours
- [ ] Automatic recovery within 60s
- [ ] No data loss during reconnection
- [ ] Health status in dashboard

#### Task 3.2: Audio Playback Stabilization
**Agent:** Audio DSP Agent  
**Duration:** 2 days  
**Dependencies:** Phase 2 complete

**Specifications:**
- Fix buffer underrun issues
- Resolve stuck audio timeout
- Improve queue management
- Handle edge cases

**Acceptance Criteria:**
- [ ] Zero audio interruptions
- [ ] Proper timeout handling
- [ ] Smooth multi-turn flow
- [ ] No zombie audio sessions

#### Task 3.3: Error Recovery Framework
**Agent:** Implementation Agent  
**Duration:** 2 days  
**Dependencies:** Tasks 3.1-3.2

**Deliverables:**
- Robust error handling throughout codebase
- User-friendly error messages
- Automatic recovery mechanisms
- Error reporting system

---

## Phase 4: Feature Enhancement

### Objectives
- Improve user experience with requested features
- Add advanced capabilities for power users
- Enhance configuration management

### Key Tasks

#### Task 4.1: Wake Word Improvements
**Duration:** 3 days  
**Dependencies:** Phase 3 complete

- Custom wake word training UI
- Performance metrics dashboard
- Multiple wake word support
- Context-aware activation

#### Task 4.2: User Interface Enhancements
**Duration:** 3 days  
**Dependencies:** Phase 3 complete

- Real-time audio visualization
- Conversation history with search
- Settings backup/restore
- Multi-user profiles

#### Task 4.3: Integration Features
**Duration:** 2 days  
**Dependencies:** Phase 3 complete

- Conversation context persistence
- Smart home scene integration
- Routine automation support
- Third-party webhooks

---

## Phase 5: Performance Optimization

### Objectives
- Support resource-constrained hardware
- Reduce latency and resource usage
- Add intelligent caching and prediction

### Key Tasks

#### Task 5.1: Hardware Optimization
**Duration:** 3 days  
**Dependencies:** Phase 4 complete

- Raspberry Pi Zero W 2 support
- Memory usage reduction
- CPU optimization
- Power consumption tuning

#### Task 5.2: Intelligent Features
**Duration:** 2 days  
**Dependencies:** Phase 4 complete

- Predictive command completion
- Local intent caching
- Offline fallback modes
- Edge ML integration

---

## Agent Assignment Matrix

| Agent Type | Primary Phases | Key Responsibilities |
|------------|---------------|----------------------|
| Audio DSP Agent | 1, 3 | Signal processing, codec optimization, audio quality |
| API Migration Agent | 2 | OpenAI API updates, protocol changes, model migration |
| Testing Agent | 1, 2, 4 | Test automation, metrics collection, validation |
| Integration Agent | 2, 3 | MCP protocol, Home Assistant, connection management |
| Documentation Agent | All | Guides, API docs, configuration examples |
| Implementation Agent | 3, 4, 5 | Core feature development, bug fixes |

## Risk Management

### Critical Risks
1. **Audio Quality Regression** - Mitigate with comprehensive testing before deployment
2. **API Breaking Changes** - Maintain compatibility flag and rollback capability
3. **Hardware Limitations** - Test on minimum spec devices throughout
4. **User Data Loss** - Implement backup/restore before major changes

### Contingency Plans
- Phase 1 extends: Prioritize wake word detection over perfect audio
- Phase 2 blocked: Continue with current API, backlog new features
- Performance issues: Create "lite" mode with reduced features

## Success Metrics

### Phase Completion Criteria
- **Phase 1:** Audio metrics baseline established, zero clipping
- **Phase 2:** New API integrated, cost reduction achieved
- **Phase 3:** 99% uptime, zero critical errors
- **Phase 4:** User satisfaction score >4.5/5
- **Phase 5:** 50% resource usage reduction

### Overall Project Success
- Response latency: <600ms (from 800ms)
- Wake word accuracy: >95% (from ~85%)
- Transcription accuracy: >98% (from ~95%)
- User retention: >80% after 30 days
- Cost per interaction: <$0.01

## Communication Plan

### Progress Reporting
- Daily: Agent status updates in Memory Bank
- Weekly: Phase progress summary
- Phase completion: Comprehensive report with metrics

### Stakeholder Updates
- GitHub releases for major milestones
- Community forum updates for testing needs
- Documentation updates with each phase

## Appendices

### A. Technical Specifications
- See `docs/technical_specs.md`

### B. Testing Matrices
- See `tests/test_matrices.md`

### C. Configuration Templates
- See `config/templates/`

---

*This Implementation Plan is a living document. Updates will be reflected in the Memory Bank as the project progresses.*