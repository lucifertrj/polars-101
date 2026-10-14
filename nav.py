"""Shared sidebar navigation.

Streamlit's auto-generated multipage nav is disabled in
``.streamlit/config.toml`` (it showed the raw ``ui`` filename), so every page
renders this labeled link list instead. Keep the paths in sync with the files
in ``pages/`` and the entrypoint ``ui.py``.
"""

import streamlit as st


def render_nav() -> None:
    """Render the custom page links at the top of the sidebar."""
    st.sidebar.page_link("ui.py", label="Chat", icon="💬")
    st.sidebar.page_link("pages/1_📈_Report.py", label="Report", icon="📈")
    st.sidebar.page_link("pages/2_🔍_Data_Explorer.py", label="Data Explorer", icon="🔍")
    st.sidebar.divider()
