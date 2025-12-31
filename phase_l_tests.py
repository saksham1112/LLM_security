"""
Tests for Phase L Long-Term Memory.

Run with: pytest phase_l_tests.py -v
"""

import pytest
import numpy as np
import time


# ===== SESSION MEMORY TESTS =====

from src.safety.alrc.phase_l.session_memory import SessionMemory


@pytest.fixture
def session_memory():
    return SessionMemory(redis_url="redis://localhost:6379", db=15)


@pytest.fixture
def sample_embedding():
    return np.random.rand(384).astype(np.float32)


@pytest.fixture
def sample_intent_profile():
    return {'educational': 0.7, 'operational': 0.2, 'instructional': 0.1, 'malicious': 0.0}


class TestSessionMemory:
    
    def test_store_and_retrieve(self, session_memory, sample_embedding, sample_intent_profile):
        session_id = "test_001"
        session_memory.store_turn(session_id, sample_embedding, 0.5, sample_intent_profile)
        trajectory = session_memory.get_trajectory(session_id)
        assert len(trajectory) == 1
        session_memory.delete_session(session_id)
    
    def test_ttl_expiry(self, session_memory, sample_embedding, sample_intent_profile):
        session_id = "test_ttl"
        session_memory.TTL_SECONDS = 2
        session_memory.store_turn(session_id, sample_embedding, 0.3, sample_intent_profile)
        assert session_memory.session_exists(session_id)
        time.sleep(3)
        assert not session_memory.session_exists(session_id)
        session_memory.TTL_SECONDS = 1800


# ===== USER PROTOTYPE TESTS =====

from src.safety.alrc.phase_l.prototype import UserPrototype


class TestUserPrototype:
    
    def test_initialization(self):
        prototype = UserPrototype(embedding_dim=384, alpha=0.05)
        assert prototype.count == 0
        assert prototype.variance == 0.0
    
    def test_first_update(self):
        prototype = UserPrototype(embedding_dim=384, alpha=0.05)
        emb = np.random.rand(384).astype(np.float32)
        p, var = prototype.update(emb)
        assert prototype.count == 1
        assert np.allclose(p, emb)
        assert var == 0.0
    
    def test_momentum_update(self):
        prototype = UserPrototype(embedding_dim=384, alpha=0.1)
        emb1 = np.ones(384, dtype=np.float32)
        emb2 = np.ones(384, dtype=np.float32) * 2
        
        prototype.update(emb1)
        p, var = prototype.update(emb2)
        
        # p should be (1-α)*emb1 + α*emb2 = 0.9*1 + 0.1*2 = 1.1
        assert np.allclose(p[0], 1.1, atol=0.01)
        assert prototype.count == 2
    
    def test_drift_magnitude(self):
        prototype = UserPrototype(embedding_dim=384, alpha=0.1)
        emb1 = np.zeros(384, dtype=np.float32)
        emb2 = np.ones(384, dtype=np.float32)
        
        prototype.update(emb1)
        prototype.update(emb2)
        
        drift = prototype.get_drift_magnitude()
        assert drift > 0, "Drift should be positive when embeddings change"
    
    def test_oscillation_detection(self):
        prototype = UserPrototype(embedding_dim=384, alpha=0.1)
        
        # Alternating embeddings (oscillation signature)
        for i in range(10):
            if i % 2 == 0:
                emb = np.zeros(384, dtype=np.float32)
            else:
                emb = np.ones(384, dtype=np.float32)
            prototype.update(emb)
        
        # Should have high variance
        assert prototype.variance > 0.1, "Oscillating should increase variance"


# ===== DRIFT TRACKER TESTS =====

from src.safety.alrc.phase_l.drift import DriftTracker


class TestDriftTracker:
    
    def test_initialization(self):
        hazards = [np.ones(384, dtype=np.float32)]
        tracker = DriftTracker(hazard_centroids=hazards, decay=0.9)
        assert tracker.drift_accumulator == 0.0
    
    def test_drift_toward_hazard(self):
        hazard = np.ones(384, dtype=np.float32)
        tracker = DriftTracker(hazard_centroids=[hazard], decay=0.9)
        
        # Start away from hazard
        p1 = np.zeros(384, dtype=np.float32)
        drift1 = tracker.compute_drift(p1)
        
        # Move toward hazard
        p2 = np.ones(384, dtype=np.float32) * 0.5
        drift2 = tracker.compute_drift(p2)
        
        # Should accumulate positive drift
        assert drift2 > drift1, "Drift should increase when moving toward hazard"
    
    def test_leaky_integration(self):
        hazard = np.ones(384, dtype=np.float32)
        tracker = DriftTracker(hazard_centroids=[hazard], decay=0.5)
        
        # Move toward hazard
        p1 = np.zeros(384, dtype=np.float32)
        tracker.compute_drift(p1)
        
        p2 = np.ones(384, dtype=np.float32) * 0.5
        drift2 = tracker.compute_drift(p2)
        
        # Stay still (no more drift)
        drift3 = tracker.compute_drift(p2)
        
        # Drift should decay
        assert drift3 < drift2, "Drift should decay when no movement"


# ===== INTEGRATION TESTS =====

from src.safety.alrc.phase_l.integration import LongTermMemory


class TestLongTermMemory:
    
    def test_disabled_mode(self):
        ltm = LongTermMemory(enabled=False)
        emb = np.random.rand(384).astype(np.float32)
        drift = ltm.update("session", emb, 0.5, {})
        assert drift == 0.0, "Should return neutral when disabled"
    
    def test_end_to_end(self):
        hazards = [np.ones(384, dtype=np.float32)]
        ltm = LongTermMemory(
            redis_url="redis://localhost:6379",
            hazard_centroids=hazards,
            enabled=True
        )
        
        session_id = "test_e2e"
        
        # Simulate trajectory toward hazard
        for i in range(5):
            emb = np.ones(384, dtype=np.float32) * (i / 10.0)
            intent = {'educational': 0.5, 'operational': 0.3, 'instructional': 0.2, 'malicious': 0.0}
            drift = ltm.update(session_id, emb, 0.0, intent)
            print(f"Turn {i}: drift={drift:.4f}")
        
        # Drift should accumulate
        prototype = ltm.get_prototype(session_id)
        assert prototype is not None
        
        # Cleanup
        ltm.reset_session(session_id)
    
    def test_stats(self):
        ltm = LongTermMemory(enabled=True)
        stats = ltm.get_stats()
        assert stats['enabled'] == True
        assert 'active_sessions' in stats
