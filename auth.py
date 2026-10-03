# =============================================================
# auth.py — SSO login + role-based access for the dashboard
#
# Login uses Streamlit's built-in OpenID Connect (st.login): Google, Microsoft
# Entra ID, Okta, Auth0 … anything with an OIDC discovery URL. Configure it in
# .streamlit/secrets.toml (see .streamlit/secrets.example.toml):
#
#   [auth]   redirect_uri, cookie_secret, client_id, client_secret, server_metadata_url
#   [access] who may see what:
#       admins           = ["ops-head@company.com"]          # everything + Ask PISA
#       viewers          = ["analyst@company.com"]           # all hubs, read-only, no Ask PISA
#       allow_any_viewer = false                             # true: any verified login is a viewer
#       [access.hub_managers]                                # only their own hub
#       WH_DEL_01 = ["delhi.manager@company.com"]
#
# Without an [auth] section the app runs in open demo mode (with a banner),
# unless PISA_ENV=production, where it refuses to start instead.
# =============================================================

import os
from collections.abc import Mapping
from dataclasses import dataclass

import streamlit as st

from config import WAREHOUSES

WAREHOUSE_NAME = {w["id"]: w["name"] for w in WAREHOUSES}


@dataclass(frozen=True)
class User:
    email: str | None
    name: str
    role: str                  # "admin" | "hub_manager" | "viewer"
    hub: str | None = None     # warehouse name a hub manager is limited to
    demo: bool = False

    @property
    def can_ask(self):
        # Ask PISA spends the company's LLM budget and its SQL can read every hub
        return self.role == "admin"


def resolve_role(email, access):
    """Pure role lookup (testable without Streamlit). Returns (role, warehouse_id) or None for no access."""
    email = (email or "").strip().lower()
    if not email:
        return None
    norm = lambda xs: {x.strip().lower() for x in (xs or [])}
    if email in norm(access.get("admins")):
        return "admin", None
    managers = access.get("hub_managers") or {}
    for wh_id, emails in (managers.items() if isinstance(managers, Mapping) else []):
        if email in norm(emails) and wh_id in WAREHOUSE_NAME:
            return "hub_manager", wh_id
    if email in norm(access.get("viewers")) or access.get("allow_any_viewer", False):
        return "viewer", None
    return None


def _auth_configured():
    try:
        return "auth" in st.secrets
    except FileNotFoundError:  # no secrets.toml at all
        return False


def _brand(subtitle):
    st.markdown(f'<div class="mast-mark" style="margin-top:2rem">PISA</div>'
                f'<p class="lede" style="margin-top:.75rem">{subtitle}</p>', unsafe_allow_html=True)


def require_user():
    """Returns the signed-in User, or renders a login / access-denied page and stops the script."""
    if not _auth_configured():
        if os.getenv("PISA_ENV") == "production":
            _brand("Sign-in is not configured for this deployment.")
            st.error("Add an [auth] section to the app's secrets (see .streamlit/secrets.example.toml).")
            st.stop()
        return User(email=None, name="Demo visitor", role="admin", demo=True)

    access = st.secrets.get("access", {})
    if not st.user.get("is_logged_in", False):
        _brand("Sign in with your company account to see today's stock and spoilage alerts.")
        provider = access.get("provider")
        if st.button("Sign in", type="primary"):
            st.login(provider) if provider else st.login()
        st.stop()

    email = st.user.get("email")
    found = resolve_role(email, access) if st.user.get("email_verified", True) else None
    if found is None:
        _brand(f"{email or 'This account'} does not have access to PISA.")
        st.caption("Ask your PISA admin to add you, then sign in again.")
        if st.button("Sign out"):
            st.logout()
        st.stop()

    role, wh_id = found
    return User(email=email, name=st.user.get("name") or email, role=role,
                hub=WAREHOUSE_NAME.get(wh_id))
