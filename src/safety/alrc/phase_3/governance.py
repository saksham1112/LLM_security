"""Governance Approval: Human-in-the-loop for critical actions."""

from typing import Dict, List, Optional
import uuid
import time
import logging

logger = logging.getLogger(__name__)


class GovernanceApproval:
    """
    Human-in-the-loop approval for critical actions.
    
    Prevents unsafe self-modification.
    """
    
    REQUIRES_APPROVAL = [
        "adapt_baseline",      # Recalibration
        "harden_thresholds",   # Permanent threshold change
        "disable_phase",       # Disable safety component
        "reset_drift_window",  # Reset drift detector
    ]
    
    def __init__(
        self,
        auto_approve_timeout: int = 3600,  # 1 hour
        enabled: bool = True
    ):
        self.enabled = enabled
        self.pending_approvals: Dict[str, Dict] = {}
        self.auto_approve_timeout = auto_approve_timeout
        
        logger.info(f"GovernanceApproval initialized: enabled={enabled}")
    
    def request_approval(self, action: str, context: Dict) -> str:
        """
        Submit action for approval.
        
        Returns:
            approval_id
        """
        if not self.enabled:
            # Auto-approve if governance disabled
            return "auto_approved"
        
        approval_id = str(uuid.uuid4())[:8]
        
        self.pending_approvals[approval_id] = {
            "action": action,
            "context": context,
            "requested_at": time.time(),
            "status": "pending",
        }
        
        logger.warning(
            f"Approval requested: id={approval_id}, action={action}, "
            f"context={context}"
        )
        
        # In production: send Slack/Email notification
        # send_slack_notification(f"Approval required: {action}", context)
        
        return approval_id
    
    def check_approval(self, approval_id: str) -> bool:
        """Check if action was approved."""
        if not self.enabled:
            return True
        
        if approval_id == "auto_approved":
            return True
        
        if approval_id not in self.pending_approvals:
            return False
        
        approval = self.pending_approvals[approval_id]
        
        # Auto-approve after timeout (configurable)
        if time.time() - approval["requested_at"] > self.auto_approve_timeout:
            logger.warning(f"Auto-approving {approval['action']} after timeout")
            approval["status"] = "approved"
            approval["auto_approved"] = True
        
        return approval["status"] == "approved"
    
    def approve(self, approval_id: str, approver: str = "admin"):
        """Manual approval."""
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id]["status"] = "approved"
            self.pending_approvals[approval_id]["approver"] = approver
            self.pending_approvals[approval_id]["approved_at"] = time.time()
            
            logger.info(f"Action approved by {approver}: {approval_id}")
    
    def reject(self, approval_id: str, rejector: str = "admin", reason: str = ""):
        """Manual rejection."""
        if approval_id in self.pending_approvals:
            self.pending_approvals[approval_id]["status"] = "rejected"
            self.pending_approvals[approval_id]["rejector"] = rejector
            self.pending_approvals[approval_id]["rejected_at"] = time.time()
            self.pending_approvals[approval_id]["reject_reason"] = reason
            
            logger.info(f"Action rejected by {rejector}: {approval_id}, reason={reason}")
    
    def get_pending(self) -> List[Dict]:
        """Get list of pending approvals."""
        return [
            {"id": aid, **approval}
            for aid, approval in self.pending_approvals.items()
            if approval["status"] == "pending"
        ]
    
    def clear_old(self, max_age_seconds: int = 86400):
        """Clear old approvals (older than 24h by default)."""
        now = time.time()
        to_remove = []
        
        for aid, approval in self.pending_approvals.items():
            if now - approval["requested_at"] > max_age_seconds:
                to_remove.append(aid)
        
        for aid in to_remove:
            del self.pending_approvals[aid]
        
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old approvals")
