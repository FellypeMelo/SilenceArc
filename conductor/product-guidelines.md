# Silence Arc - Product Guidelines (AI-XP Integrated)

## 🎨 User Experience (UX) & Design
- **Efficiency-First Interaction:** Minimize clicks for toggling suppression or switching profiles.
- **Information Density:** Rich, real-time telemetry and signal visualization (latency, memory, GPU util) without clutter.
- **High Aesthetics:** Modern, polished visual identity reflecting Intel Arc's high performance.

## ✍️ Communication & Tone
- **Technical Precision:** Use direct, precise language. Avoid ambiguity or "fluff".
- **Data-Driven Status:** All reporting should be backed by specific hardware and audio metrics.

## 🏗️ Engineering Standards (AI-XP Core)
- **TDD Mandatory (Red-Green-Refactor):** No production code changes without a prior failing test. 
- **Clean Architecture:** Domain logic must remain pure and decoupled from infrastructure (Audio APIs, GPU Drivers, GPU backends, GUI Frameworks).
- **SOLID Compliance:** Every class has exactly one reason to change (SRP). Dependencies are injected via constructor (DIP).
- **Algorithmic Elegance:** Functions < 15 lines. Cyclomatic complexity ≤ 15. Nesting depth ≤ 2.
- **Security Native:** Treat all audio sources and configuration files as untrusted inputs. Implement fail-secure error handling.

## 🚀 Development Operations
- **Silent Recovery:** Prioritize seamless continuity for transient hardware/audio issues.
- **Confinement Protocol:** Formalize RED phase proofs before any implementation.
- **YAGNI + KISS:** No "just-in-case" features. Minimalist implementations that fulfill the current track only.
- **Big-O Awareness:** Every DSP algorithm and buffer operation must have proven asymptotic complexity.
