"""
Enterprise Tenant Manager (SQL Edition) - Config-driven multi-tenant isolation.

Generalized from Driscoll fork - domain validation now config-driven.

Tables created:
- cog_tenants: User registry with vault paths and permissions
- cog_sessions: Active sessions for token validation
- cog_usage: Token usage for expense reporting

Usage:
    from enterprise_tenant import EnterpriseTenantManager, TenantContext
    
    manager = EnterpriseTenantManager(
        connection_string=conn_str,
        allowed_domains=["company.com", "subsidiary.com"]
    )
    await manager.initialize()
    
    tenant = await manager.get_or_create_tenant("alice@company.com")

Version: 2.0.0 (enterprise-generic)
"""

import os
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TenantContext:
    """Scoped context for a single tenant."""
    user_id: str
    email: str
    tenant_folder: Path
    zone: Optional[str] = None
    role: str = "user"
    division: str = "default"
    direct_reports: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "tenant_folder": str(self.tenant_folder),
            "zone": self.zone,
            "role": self.role,
            "division": self.division,
        }


class EnterpriseTenantManager:
    """
    Config-driven tenant management using MS SQL Server.
    
    No hardcoded domains - pass allowed_domains at init or via config.
    """
    
    # Schema for tenant tables
    SCHEMA_SQL = """
    -- Tenant registry
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='cog_tenants' AND xtype='U')
    CREATE TABLE cog_tenants (
        user_id VARCHAR(32) PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE,
        zone VARCHAR(50),
        division VARCHAR(50) DEFAULT 'default',
        role VARCHAR(50) DEFAULT 'user',
        direct_reports VARCHAR(MAX),
        created_at DATETIME2 DEFAULT GETDATE(),
        last_active DATETIME2 DEFAULT GETDATE(),
        is_active BIT DEFAULT 1
    );
    
    -- Session tokens
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='cog_sessions' AND xtype='U')
    CREATE TABLE cog_sessions (
        token VARCHAR(64) PRIMARY KEY,
        user_id VARCHAR(32) NOT NULL,
        created_at DATETIME2 DEFAULT GETDATE(),
        expires_at DATETIME2 NOT NULL,
        is_valid BIT DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES cog_tenants(user_id)
    );
    
    -- Usage logging
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='cog_usage' AND xtype='U')
    CREATE TABLE cog_usage (
        id INT IDENTITY(1,1) PRIMARY KEY,
        user_id VARCHAR(32) NOT NULL,
        timestamp DATETIME2 DEFAULT GETDATE(),
        tokens_in INT NOT NULL,
        tokens_out INT NOT NULL,
        estimated_cost DECIMAL(10,6),
        division VARCHAR(50),
        query_type VARCHAR(50),
        model VARCHAR(100),
        FOREIGN KEY (user_id) REFERENCES cog_tenants(user_id)
    );
    
    -- Indexes
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_usage_timestamp')
    CREATE INDEX idx_usage_timestamp ON cog_usage(timestamp);
    
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name='idx_usage_user')
    CREATE INDEX idx_usage_user ON cog_usage(user_id);
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        vault_base: Path = Path("./vault"),
        allowed_domains: Optional[List[str]] = None,
        division_patterns: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize enterprise tenant manager.
        
        Args:
            connection_string: ODBC connection string (or from env)
            vault_base: Base path for tenant folders on filesystem
            allowed_domains: List of allowed email domains (e.g., ["company.com"])
            division_patterns: Email patterns to detect division
        """
        # Try multiple env vars for backward compat
        self.conn_str = connection_string or os.environ.get(
            "ENTERPRISE_SQL_CONN",
            os.environ.get("DRISCOLL_SQL_CONN")
        )
        if not self.conn_str:
            raise ValueError(
                "Connection string required. Set ENTERPRISE_SQL_CONN or pass directly."
            )
        
        self.vault_base = Path(vault_base)
        self.shared_path = self.vault_base / "shared"
        self.tenants_path = self.vault_base / "tenants"
        
        # Domain validation - config-driven, not hardcoded
        self.allowed_domains: Set[str] = set()
        if allowed_domains:
            self.allowed_domains = {d.lower().strip() for d in allowed_domains}
        
        # Division detection patterns - customizable per deployment
        self.division_patterns = division_patterns or {
            "transportation": ["transport", "driver", "fleet", "dispatch", "logistics"],
            "operations": ["ops", "operations", "warehouse", "inventory"],
            "hr": ["hr", "human", "people"],
            "sales": ["sales", "account", "rep"],
        }
        
        # Ensure filesystem paths exist
        self.shared_path.mkdir(parents=True, exist_ok=True)
        self.tenants_path.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self):
        """Get database connection."""
        return pyodbc.connect(self.conn_str)
    
    def initialize(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for statement in self.SCHEMA_SQL.split(';'):
                statement = statement.strip()
                if statement:
                    try:
                        cursor.execute(statement)
                    except pyodbc.Error as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Schema statement warning: {e}")
            conn.commit()
        logger.info("Enterprise tenant tables initialized")
    
    def is_allowed_user(self, email: str) -> bool:
        """
        Check if email is from an allowed domain.
        
        Config-driven - no hardcoded domains.
        """
        if not self.allowed_domains:
            # No restrictions configured = allow all
            return True
        
        email_lower = email.lower().strip()
        return any(
            email_lower.endswith(f"@{domain}")
            for domain in self.allowed_domains
        )
    
    def _hash_user_id(self, email: str) -> str:
        """Create stable hash from email."""
        return hashlib.sha256(email.lower().strip().encode()).hexdigest()[:12]
    
    def _detect_division(self, email: str) -> str:
        """Detect division from email pattern using configured patterns."""
        email_lower = email.lower()
        
        for division, patterns in self.division_patterns.items():
            if any(p in email_lower for p in patterns):
                return division
        
        return "default"
    
    def _create_tenant_folders(self, user_id: str):
        """Create filesystem folders for tenant."""
        tenant_folder = self.tenants_path / f"usr_{user_id}"
        tenant_folder.mkdir(parents=True, exist_ok=True)
        (tenant_folder / "episodic").mkdir(exist_ok=True)
        (tenant_folder / "artifacts").mkdir(exist_ok=True)
    
    def get_or_create_tenant(
        self,
        email: str,
        zone: Optional[str] = None,
        division: Optional[str] = None,
    ) -> TenantContext:
        """
        Get existing tenant or create new one.
        
        Args:
            email: User's email
            zone: Optional zone assignment
            division: Optional division override
            
        Returns:
            TenantContext for this user
        """
        user_id = self._hash_user_id(email)
        tenant_folder = self.tenants_path / f"usr_{user_id}"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT user_id, email, zone, division, role, direct_reports, created_at, last_active "
                "FROM cog_tenants WHERE user_id = ?",
                user_id
            )
            row = cursor.fetchone()
            
            if row is None:
                # Create new tenant
                detected_division = division or self._detect_division(email)
                
                cursor.execute(
                    "INSERT INTO cog_tenants (user_id, email, zone, division, role, direct_reports) "
                    "VALUES (?, ?, ?, ?, 'user', '[]')",
                    user_id, email.lower(), zone, detected_division
                )
                conn.commit()
                
                self._create_tenant_folders(user_id)
                
                logger.info(f"Created new tenant: {email} -> usr_{user_id}")
                
                return TenantContext(
                    user_id=user_id,
                    email=email.lower(),
                    tenant_folder=tenant_folder,
                    zone=zone,
                    role="user",
                    division=detected_division,
                    direct_reports=[],
                    created_at=datetime.now(),
                    last_active=datetime.now(),
                )
            else:
                # Update last_active
                cursor.execute(
                    "UPDATE cog_tenants SET last_active = GETDATE() WHERE user_id = ?",
                    user_id
                )
                conn.commit()
                
                import json
                direct_reports = json.loads(row[5]) if row[5] else []
                
                return TenantContext(
                    user_id=row[0],
                    email=row[1],
                    tenant_folder=tenant_folder,
                    zone=row[2],
                    division=row[3],
                    role=row[4],
                    direct_reports=direct_reports,
                    created_at=row[6],
                    last_active=row[7],
                )
    
    def get_vault_paths(self, tenant: TenantContext) -> List[Path]:
        """Get all vault paths this tenant can read."""
        paths = []
        
        # 1. Everyone gets shared
        if self.shared_path.exists():
            paths.append(self.shared_path)
        
        # 2. Their own episodic folder
        own_episodic = tenant.tenant_folder / "episodic"
        if own_episodic.exists():
            paths.append(own_episodic)
        
        # 3. Leaders get direct reports
        if tenant.role in ("zone_leader", "manager") and tenant.direct_reports:
            for report_id in tenant.direct_reports:
                report_folder = self.tenants_path / f"usr_{report_id}" / "episodic"
                if report_folder.exists():
                    paths.append(report_folder)
        
        # 4. Admins get everything
        if tenant.role == "admin":
            for tenant_folder in self.tenants_path.iterdir():
                if tenant_folder.is_dir():
                    episodic = tenant_folder / "episodic"
                    if episodic.exists():
                        paths.append(episodic)
        
        return paths
    
    def get_write_path(self, tenant: TenantContext) -> Path:
        """Get path where tenant can write memories."""
        return tenant.tenant_folder / "episodic"
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def create_session(self, email: str, hours_valid: int = 24) -> str:
        """Create a session token for a user."""
        user_id = self._hash_user_id(email)
        token = secrets.token_urlsafe(48)
        expires_at = datetime.now() + timedelta(hours=hours_valid)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO cog_sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                token, user_id, expires_at
            )
            conn.commit()
        
        return token
    
    def validate_session(self, token: str) -> Optional[TenantContext]:
        """Validate a session token."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT s.user_id, t.email, t.zone, t.division, t.role "
                "FROM cog_sessions s "
                "JOIN cog_tenants t ON s.user_id = t.user_id "
                "WHERE s.token = ? AND s.is_valid = 1 AND s.expires_at > GETDATE()",
                token
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            tenant_folder = self.tenants_path / f"usr_{row[0]}"
            
            return TenantContext(
                user_id=row[0],
                email=row[1],
                tenant_folder=tenant_folder,
                zone=row[2],
                division=row[3],
                role=row[4],
            )
    
    def invalidate_session(self, token: str):
        """Invalidate a session token."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE cog_sessions SET is_valid = 0 WHERE token = ?",
                token
            )
            conn.commit()
    
    # =========================================================================
    # USAGE LOGGING
    # =========================================================================
    
    def log_usage(
        self,
        user_id: str,
        tokens_in: int,
        tokens_out: int,
        division: str = "default",
        query_type: str = "chat",
        model: str = "grok-4-fast-reasoning",
    ):
        """Log token usage to database."""
        estimated_cost = (tokens_in * 0.000002) + (tokens_out * 0.000006)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO cog_usage (user_id, tokens_in, tokens_out, estimated_cost, division, query_type, model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                user_id, tokens_in, tokens_out, estimated_cost, division, query_type, model
            )
            conn.commit()
    
    def get_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get usage summary for expense reporting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(tokens_in + tokens_out) as total_tokens,
                    SUM(estimated_cost) as total_cost,
                    division
                FROM cog_usage
                WHERE 1=1
            """
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " GROUP BY division"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            by_division = {}
            total_requests = 0
            total_tokens = 0
            total_cost = 0.0
            
            for row in rows:
                by_division[row[3]] = {
                    "requests": row[0],
                    "tokens": row[1],
                    "cost": float(row[2]) if row[2] else 0.0,
                }
                total_requests += row[0]
                total_tokens += row[1] or 0
                total_cost += float(row[2]) if row[2] else 0.0
            
            return {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 2),
                "by_division": by_division,
            }
    
    # =========================================================================
    # ADMIN
    # =========================================================================
    
    def promote_to_manager(self, user_id: str, direct_reports: List[str]):
        """Promote user to manager role."""
        import json
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE cog_tenants SET role = 'manager', direct_reports = ? WHERE user_id = ?",
                json.dumps(direct_reports), user_id
            )
            conn.commit()
    
    def list_tenants(self) -> List[Dict[str, Any]]:
        """List all tenants."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT user_id, email, zone, division, role, created_at, last_active "
                "FROM cog_tenants WHERE is_active = 1"
            )
            return [
                {
                    "user_id": row[0],
                    "email": row[1],
                    "zone": row[2],
                    "division": row[3],
                    "role": row[4],
                    "created_at": row[5],
                    "last_active": row[6],
                }
                for row in cursor.fetchall()
            ]


# =============================================================================
# FACTORY FUNCTION - Load from config
# =============================================================================

def create_tenant_manager_from_config(config: Dict[str, Any]) -> EnterpriseTenantManager:
    """
    Create tenant manager from config dict.
    
    Args:
        config: Parsed enterprise_config.yaml
        
    Returns:
        Configured EnterpriseTenantManager
    """
    deployment = config.get("deployment", {})
    auth = deployment.get("auth", {})
    
    return EnterpriseTenantManager(
        allowed_domains=auth.get("allowed_domains", []),
        vault_base=Path(config.get("tenant", {}).get("base_dir", "./vault")),
    )


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if not PYODBC_AVAILABLE:
        print("Error: pyodbc required. Run: pip install pyodbc")
        sys.exit(1)
    
    # Test with example domains
    manager = EnterpriseTenantManager(
        allowed_domains=["example.com", "test.com"]
    )
    
    # Test domain validation
    print("Domain validation test:")
    print(f"  alice@example.com: {manager.is_allowed_user('alice@example.com')}")  # True
    print(f"  bob@other.com: {manager.is_allowed_user('bob@other.com')}")  # False
    
    # If SQL connection available, test full flow
    if manager.conn_str:
        manager.initialize()
        print(f"\nTenants: {len(manager.list_tenants())}")