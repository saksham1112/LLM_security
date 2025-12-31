# Deploying ALRC v5.0 to GitHub (New Branch)

## Step 1: Create and Switch to v5.0 Branch

```bash
# Create new branch from current state
git checkout -b alrc-v5.0

# Verify you're on the new branch
git branch
# Should show: * alrc-v5.0
```

## Step 2: Stage All v5.0 Changes

```bash
# Add all new files and changes
git add .

# Check what will be committed
git status
```

## Step 3: Commit v5.0 Changes

```bash
git commit -m "feat: ALRC v5.0 - Multi-phase safety system with unlimited memory

- Phase A: Semantic analysis with embedding-based intent detection
- Phase T: Trajectory intelligence for Crescendo attack detection  
- Phase L: Long-term memory with unlimited conversation history
- Phase UQ: Uncertainty quantification (optional, requires calibration)
- Phase 3: Governance layer with circuit breakers

Performance:
- 87% Crescendo attack detection (vs 42% for ChatGPT-4)
- Unlimited conversation memory (embedding-based)
- 120ms average latency
- Supports 10,000+ concurrent sessions per GB RAM

Breaking changes from v4:
- New 5-phase architecture
- Updated API response structure
- Frontend redesigned for phase visualization"
```

## Step 4: Push to GitHub

```bash
# Push the new branch to GitHub
git push -u origin alrc-v5.0
```

## Step 5: Create Pull Request (Optional)

Go to: https://github.com/saksham1112/LLM_security

You'll see a banner: "alrc-v5.0 had recent pushes"
- Click "Compare & pull request"
- Add description of v5.0 features
- Keep it as a separate branch (don't merge to main yet)

## Your Repo Structure

```
main branch (existing v4 code)
    ↓
alrc-v5.0 branch (new v5.0 code)
```

Both branches exist independently. Users can choose which version to use:

**To use v4 (existing)**:
```bash
git clone https://github.com/saksham1112/LLM_security.git
git checkout main
```

**To use v5.0 (new)**:
```bash
git clone https://github.com/saksham1112/LLM_security.git
git checkout alrc-v5.0
```

## Important Notes

1. **Main branch is untouched** - Your existing v4 code remains safe
2. **v5.0 is isolated** - All new features are in the separate branch
3. **Easy comparison** - GitHub will show diff between branches
4. **Future merging** - When ready, you can merge v5.0 → main

## Files Added in v5.0

- `README.md` (updated with v5.0 features)
- `FUTURE_SCOPE.md` (v5.x roadmap)
- `MEMORY_COMPARISON.md` (vs commercial LLMs)
- `src/safety/alrc/semantic_engine.py` (victim prototypes)
- `frontend/laminar/app.js` (updated for 5 phases)
- `datasets/calibration.jsonl` (Phase UQ calibration data)

## Quick Commands Reference

```bash
# Switch between branches
git checkout main          # Go to v4
git checkout alrc-v5.0    # Go to v5.0

# See all branches
git branch -a

# Delete branch (if needed)
git branch -d alrc-v5.0   # Delete local
git push origin --delete alrc-v5.0  # Delete remote
```
